use crate::responsive::Responsive;
use crate::Node;
use crate::{as_entity::AsEntity, entity::CalculateCollisions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The main struct you will interact with. This is a k-d tree containing all of your gravitational
/// entities.
#[cfg_attr(feature = "bevy_ecs", derive(::bevy_ecs::prelude::Component))]
#[derive(Serialize, Deserialize)]
pub struct GravTree<T: AsEntity + Responsive + Clone> {
    /// A GravTree consists of a root [[Node]]. A [[Node]] is a recursive binary tree data structure.
    /// Tragically must be public for now for testing reasons. Perhaps could be replaced by various
    /// getter methods later.
    pub(crate) root: Node<T>,
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
                number_of_entities: size_of_vec,
                time_step,
                max_entities,
                theta,
                calculate_collisions,
            };
        }

        // Because of the tree's recursive gravity calculation, there needs to be a parent node
        // that "contains" the _real_ root node. This "phantom_parent" serves no purpose other than
        // to hold a pointer to the real root node. Perhaps not the most ideal situation for now,
        // and can be made more elegant in the future, if need be.
        // The real root of the tree is therefore tree.root.left
        let mut phantom_parent = Node::new();
        phantom_parent.left = Some(Box::new(Node::<T>::new_root_node(pts, max_entities)));
        phantom_parent.points = Some(Vec::new());

        GravTree {
            root: phantom_parent,
            number_of_entities: size_of_vec,
            time_step,
            max_entities,
            theta,
            calculate_collisions,
        }
    }
    /// Sets the `theta` value of the simulation.
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }

    /// Traverses the tree and returns a vector of all entities in the tree.
    pub fn as_vec(&self) -> Vec<T> {
        let node = self.root.clone();
        let mut to_return: Vec<T> = Vec::new();
        if let Some(node) = &node.left {
            to_return.append(&mut node.traverse_tree_helper());
        }
        if let Some(node) = &node.right {
            to_return.append(&mut node.traverse_tree_helper());
        } else {
            to_return.append(
                &mut (node
                    .points
                    .as_ref()
                    .expect("unexpected null node #9")
                    .clone()),
            );
        }
        to_return
    }

    /// Returns an iterator over all entities in the tree without cloning them.
    /// This is a zero-copy alternative to `as_vec()`.
    ///
    /// # Example
    /// ```rust,ignore
    /// # use bigbang::{GravTree, Entity, CalculateCollisions};
    /// # let entities = vec![Entity::default()];
    /// let tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    /// for entity in tree.iter() {
    ///     // Work with borrowed entity without cloning
    ///     println!("Entity at ({}, {}, {})", entity.x, entity.y, entity.z);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let mut iters: Vec<Box<dyn Iterator<Item = &T> + '_>> = Vec::new();

        if let Some(ref left) = self.root.left {
            iters.push(Box::new(left.iter()));
        }
        if let Some(ref right) = self.root.right {
            iters.push(Box::new(right.iter()));
        }
        if let Some(ref points) = self.root.points {
            iters.push(Box::new(points.iter()));
        }

        iters.into_iter().flatten()
    }
    /// Gets the total number of entities contained by this tree.
    pub fn get_number_of_entities(&self) -> usize {
        self.number_of_entities
    }

    /// This function creates a vector of all entities from the tree and applies gravity to them.
    /// Returns a new GravTree.
    // of note: The c++ implementation of this just stores a vector of
    // accelerations and matches up the
    // indexes with the indexes of the entities, and then applies them. That way
    // some memory is saved.
    // I am not sure if this will be necessary or very practical in the rust
    // implementation (I would have to implement indexing in my GravTree struct).
    pub fn time_step(&self) -> GravTree<T> {
        // Use iterator to collect entity references (zero-copy)
        let entities: Vec<&T> = self.iter().collect();

        // Calculate new positions for all entities in parallel
        let updated_entities: Vec<T> = entities
            .par_iter()
            .map(|&x| {
                let x_entity = x.as_entity();
                let accel = match self.calculate_collisions {
                    CalculateCollisions::Yes => {
                        x_entity.get_acceleration_and_collisions(&self.root, self.theta)
                    }
                    CalculateCollisions::No => {
                        x_entity.get_acceleration_without_collisions(&self.root, self.theta)
                    }
                };
                x.respond(accel, self.time_step)
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
    /// This is more efficient than `time_step()` as it avoids an extra allocation by
    /// mutating the tree directly rather than returning a new one.
    ///
    /// # Example
    /// ```rust,ignore
    /// # use bigbang::{GravTree, Entity, CalculateCollisions};
    /// # let entities = vec![Entity::default()];
    /// let mut tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    /// tree.time_step_mut(); // Updates tree in place
    /// ```
    pub fn time_step_mut(&mut self) {
        // Phase 1: Collect entity references (zero-copy via iterator)
        // We still need to clone here because we need the tree structure for calculations
        // but also need owned entities to rebuild the tree
        let entities: Vec<&T> = self.iter().collect();

        if entities.is_empty() {
            self.root = Node::new();
            self.number_of_entities = 0;
            return;
        }

        // Phase 2: Calculate accelerations in parallel using references to entities
        // and references to the existing tree structure
        let updated_entities: Vec<T> = entities
            .par_iter()
            .map(|&x| {
                let x_entity = x.as_entity();
                let accel = match self.calculate_collisions {
                    CalculateCollisions::Yes => {
                        x_entity.get_acceleration_and_collisions(&self.root, self.theta)
                    }
                    CalculateCollisions::No => {
                        x_entity.get_acceleration_without_collisions(&self.root, self.theta)
                    }
                };
                x.respond(accel, self.time_step)
            })
            .collect();

        // Phase 3: Rebuild the tree structure with updated entities
        let number_of_entities = updated_entities.len();


        let mut phantom_parent = Node::new();
        phantom_parent.left = Some(Box::new(Node::<T>::new_root_node(&updated_entities, self.max_entities)));
        phantom_parent.points = Some(Vec::new());

        self.root = phantom_parent;
        self.number_of_entities = number_of_entities;
    }
}
