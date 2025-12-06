use crate::responsive::Responsive;
use crate::Node;
use crate::{as_entity::AsEntity, entity::CalculateCollisions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The main struct you will interact with. This is a k-d tree containing all of your gravitational
/// entities.
#[cfg_attr(feature = "bevy_ecs", derive(::bevy_ecs::prelude::Component))]
#[derive(Serialize, Deserialize)]
#[must_use]
pub struct GravTree<T: AsEntity + Responsive + Clone> {
    /// A GravTree consists of a root [[Node]]. A [[Node]] is a recursive binary tree data structure.
    /// Tragically must be public for now for testing reasons. Perhaps could be replaced by various
    /// getter methods later.
    pub(crate) root: Node<T>,
    /// Arena storage: entities stored in a contiguous flat array for cache-efficient access
    /// and zero-copy mutations. The tree structure references these entities.
    entities: Vec<T>,
    /// This is just the number of entities in the tree. This is used in testing to verify that no
    /// entities are being dropped.
    number_of_entities: usize,
    /// This coefficient determines the granularity of the simulation, i.e. how much each frame of
    /// the simulation actually moves the individual entities.
    time_step: f64, // the time coefficient; how large each simulation frame is time-wise.
    /// The maximum number of entities to be contained within any leaf node. Defaults to 3 but is
    /// configurable. This is _not_ the maximum number of entities in the simulation. A higher
    /// number here will result in lower simulation granularity.
    max_entities: i32,
    /// `theta` is how far away a node has to be before the simulation starts approximating its
    /// contained entities by treating them as one large node instead of individually addressing
    /// them.
    /// More specifically, this is the tolerance for the distance from an entity to the center of mass of an entity
    /// If the distance is beyond this threshold, we treat the entire node as one giant
    /// entity instead of recursing into it.
    theta: f64,
    /// Whether or not to calculate collisions when time stepping
    calculate_collisions: CalculateCollisions,
    /// Tracks if the tree structure is stale and needs rebuilding.
    /// Set to true after time_step_mut modifies entities in place.
    tree_needs_rebuild: bool,
}

impl<T: AsEntity + Responsive + Clone + Send + Sync> GravTree<T> {
    pub fn new(
        pts: &[T],
        time_step: f64,
        max_entities: i32,
        theta: f64,
        calculate_collisions: CalculateCollisions,
    ) -> GravTree<T>
    where
        T: AsEntity,
    {
        let size_of_vec = pts.len();
        // Handle the case where a grav tree is initialized without any points...
        if size_of_vec == 0 {
            return GravTree {
                root: Node::new(),
                entities: Vec::new(),
                number_of_entities: size_of_vec,
                time_step,
                max_entities,
                theta,
                calculate_collisions,
                tree_needs_rebuild: false,
            };
        }

        // Store entities in the arena
        let entities = pts.to_vec();

        // Because of the tree's recursive gravity calculation, there needs to be a parent node
        // that "contains" the _real_ root node. This "phantom_parent" serves no purpose other than
        // to hold a pointer to the real root node. Perhaps not the most ideal situation for now,
        // and can be made more elegant in the future, if need be.
        // The real root of the tree is therefore tree.root.left
        let mut phantom_parent = Node::new();
        phantom_parent.left = Some(Box::new(Node::<T>::new_root_node(&entities, max_entities)));
        phantom_parent.points = Some(Vec::new());

        GravTree {
            root: phantom_parent,
            entities,
            number_of_entities: size_of_vec,
            time_step,
            max_entities,
            theta,
            calculate_collisions,
            tree_needs_rebuild: false,
        }
    }
    /// Sets the `theta` value of the simulation.
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }

    /// Traverses the tree and returns a vector of all entities in the tree.
    ///
    /// With arena pattern, this simply clones from the entity storage.
    pub fn as_vec(&self) -> Vec<T> {
        self.entities.clone()
    }

    /// Returns an iterator over all entities in the tree without cloning them.
    /// This is a zero-copy alternative to `as_vec()`.
    ///
    /// With the arena pattern, this directly iterates over the entity storage
    /// for maximum efficiency.
    ///
    /// # Example
    /// ```rust
    /// # use bigbang::{GravTree, Entity, CalculateCollisions};
    /// # let entities = vec![Entity::default()];
    /// let tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    /// for entity in tree.iter() {
    ///     // Work with borrowed entity without cloning
    ///     println!("Entity at ({}, {}, {})", entity.x, entity.y, entity.z);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.entities.iter()
    }
    /// Gets the total number of entities contained by this tree.
    pub fn get_number_of_entities(&self) -> usize {
        self.number_of_entities
    }

    /// This function creates a new GravTree with updated entity positions after applying gravity.
    /// Returns a new GravTree.
    ///
    /// This method is provided for backward compatibility and immutable workflows.
    /// For better performance with large simulations, use `time_step_mut()` instead,
    /// which avoids allocating a new tree.
    pub fn time_step(&self) -> GravTree<T> {
        if self.entities.is_empty() {
            return GravTree::<T>::new(
                &[],
                self.time_step,
                self.max_entities,
                self.theta,
                self.calculate_collisions,
            );
        }

        // Calculate new positions for all entities in parallel using arena entities
        let updated_entities: Vec<T> = self.entities
            .par_iter()
            .map(|entity| {
                let entity_as_entity = entity.as_entity();
                let accel = match self.calculate_collisions {
                    CalculateCollisions::Yes => {
                        entity_as_entity.get_acceleration_and_collisions(&self.root, self.theta)
                    }
                    CalculateCollisions::No => {
                        entity_as_entity.get_acceleration_without_collisions(&self.root, self.theta)
                    }
                };
                entity.respond(accel, self.time_step)
            })
            .collect();

        // Construct a new tree with the updated entities
        GravTree::<T>::new(
            &updated_entities,
            self.time_step,
            self.max_entities,
            self.theta,
            self.calculate_collisions,
        )
    }

    /// This function updates the tree in-place by applying gravity to all entities.
    /// This is more efficient than `time_step()` as it avoids allocation by
    /// mutating entities directly in the arena using `respond_mut()`.
    ///
    /// After calling this method, the tree structure becomes stale and will be
    /// automatically rebuilt on the next call to `time_step_mut()`. For more control,
    /// use `rebuild_tree()` explicitly.
    ///
    /// # Example
    /// ```rust
    /// # use bigbang::{GravTree, Entity, CalculateCollisions};
    /// # let entities = vec![Entity::default()];
    /// let mut tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    /// tree.time_step_mut(); // Updates entities in place (zero-copy!)
    /// ```
    pub fn time_step_mut(&mut self) {
        if self.entities.is_empty() {
            self.root = Node::new();
            self.number_of_entities = 0;
            self.tree_needs_rebuild = false;
            return;
        }

        // CRITICAL: Rebuild tree BEFORE calculating accelerations if entities moved in previous step
        // This ensures spatial queries use accurate positions, maintaining physics correctness.
        // The tree becomes stale after entities are mutated (below), but is rebuilt here
        // at the START of the next step, so accelerations are always calculated with fresh data.
        if self.tree_needs_rebuild {
            self.rebuild_tree();
        }

        // Calculate accelerations for all entities in parallel
        // We extract just the acceleration values to avoid lifetime issues with SimulationResult
        let accelerations: Vec<(f64, f64, f64)> = self.entities
            .par_iter()
            .map(|entity| {
                let entity_as_entity = entity.as_entity();
                let result = match self.calculate_collisions {
                    CalculateCollisions::Yes => {
                        entity_as_entity.get_acceleration_and_collisions(&self.root, self.theta)
                    }
                    CalculateCollisions::No => {
                        entity_as_entity.get_acceleration_without_collisions(&self.root, self.theta)
                    }
                };
                result.gravitational_acceleration
            })
            .collect();

        // Apply accelerations to entities in parallel using respond_mut (TRUE ZERO-COPY!)
        // We create SimulationResults on the fly with empty collision vectors
        self.entities
            .par_iter_mut()
            .zip(accelerations.par_iter())
            .for_each(|(entity, &accel)| {
                use crate::SimulationResult;
                let sim_result = SimulationResult {
                    collisions: Vec::new(),
                    gravitational_acceleration: accel,
                };
                entity.respond_mut(sim_result, self.time_step);
            });

        // Mark tree as stale since entities have moved to new positions.
        // The tree will be rebuilt at the START of the next time_step_mut() call (see above),
        // ensuring accelerations are always calculated with accurate spatial data.
        // This deferred rebuild strategy achieves zero-copy while maintaining correctness.
        self.tree_needs_rebuild = true;
    }

    /// Explicitly rebuilds the tree structure from current entity positions.
    ///
    /// This is automatically called when needed by `time_step_mut()`, but you can
    /// call it manually for more control over when tree reconstruction happens.
    ///
    /// # Example
    /// ```rust
    /// # use bigbang::{GravTree, Entity, CalculateCollisions};
    /// # let entities = vec![Entity::default()];
    /// let mut tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    /// tree.time_step_mut();
    /// // ... do other work ...
    /// tree.rebuild_tree(); // Rebuild now for next operation
    /// ```
    pub fn rebuild_tree(&mut self) {
        if self.entities.is_empty() {
            self.root = Node::new();
            self.root.points = Some(Vec::new());
            self.tree_needs_rebuild = false;
            return;
        }

        let mut phantom_parent = Node::new();
        phantom_parent.left = Some(Box::new(Node::<T>::new_root_node(&self.entities, self.max_entities)));
        phantom_parent.points = Some(Vec::new());

        self.root = phantom_parent;
        self.tree_needs_rebuild = false;
    }
}
