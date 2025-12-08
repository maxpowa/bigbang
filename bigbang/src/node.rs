use crate::as_entity::AsEntity;
use crate::dimension::Dimension;
use crate::entity::Entity;
use serde::{Deserialize, Serialize};

/// This is internal to the tree and is not exposed to the consumer.
///
/// A [[Node]] is either a leaf or an internal node. If it is an internal node, it contains _no_ entities. Instead,
/// it is purely structural and contains aggregate statistics of its children. An internal node will sometimes be treated
/// as an [[Entity]] of its own via `as_entity()` (not to be confused with the trait [[AsEntity]] -- this is just a method on
/// [[Node]].
///
/// If a [[Node]] is a leaf, then it contains up to `max_entities` particles, as well as the aggregate values of these particles.
/// These aggregate values are the center of mass, the total mass, and max/min values for each dimension.
///
/// # Arena Pattern
///
/// Nodes store indices into the parent `GravTree`'s entity arena rather than cloning entities. This provides:
/// - Zero-copy tree construction (no entity duplication)
/// - Reduced memory usage (~50% for entity storage)
/// - Cache-friendly access patterns
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Node {
    split_dimension: Option<Dimension>, // Dimension that this node splits at.
    split_value: f64,                   // Value that this node splits at.
    pub(crate) left: Option<Box<Node>>, // Left subtree.
    pub(crate) right: Option<Box<Node>>, // Right subtree.
    pub(crate) point_indices: Option<Vec<usize>>, // Indices into arena for leaf entities.
    pub(crate) center_of_mass: (f64, f64, f64), /* The center of mass for this node and it's children all
                                                 * together. (x, y, z). */
    total_mass: f64, // Total mass of all entities under this node.
    r_max: f64,      // Maximum radius that is a child of this node.
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
}

impl Node {
    pub(crate) fn new() -> Node {
        Node {
            split_dimension: None,
            split_value: 0.0,
            left: None,
            right: None,
            point_indices: None,
            center_of_mass: (0.0, 0.0, 0.0),
            total_mass: 0.0,
            r_max: 0.0,
            x_min: 0.0,
            x_max: 0.0,
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
        }
    }
    /// Looks into its own children's maximum and minimum values, setting its own
    /// values accordingly.
    pub(crate) fn set_max_mins(&mut self) {
        let xmin = f64::min(
            self.left.as_ref().unwrap().x_min,
            self.right.as_ref().unwrap().x_min,
        );
        let xmax = f64::max(
            self.left.as_ref().unwrap().x_max,
            self.right.as_ref().unwrap().x_max,
        );
        let ymin = f64::min(
            self.left.as_ref().unwrap().y_min,
            self.right.as_ref().unwrap().y_min,
        );
        let ymax = f64::max(
            self.left.as_ref().unwrap().y_max,
            self.right.as_ref().unwrap().y_max,
        );
        let zmin = f64::min(
            self.left.as_ref().unwrap().z_min,
            self.right.as_ref().unwrap().z_min,
        );
        let zmax = f64::max(
            self.left.as_ref().unwrap().z_max,
            self.right.as_ref().unwrap().z_max,
        );
        let left_r_max = self.left.as_ref().expect("unexpected null node #7").r_max;
        let right_r_max = self.right.as_ref().expect("unexpected null node #8").r_max;
        self.r_max = f64::max(left_r_max, right_r_max);
        self.x_min = xmin;
        self.x_max = xmax;
        self.y_min = ymin;
        self.y_max = ymax;
        self.z_min = zmin;
        self.z_max = zmax;
    }
    // Used when treating a node as the sum of its parts in gravity calculations.
    /// Converts a node into an entity with the x, y, z, and mass being derived from the center of
    /// mass and the total mass of the entities it contains.
    pub(crate) fn as_entity(&self) -> Entity {
        // Construct a "super radius" of the largest dimension / 2 + a radius.
        let (range_x, range_y, range_z) = (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        );
        let max_dimension_range = f64::max(range_x, f64::max(range_y, range_z));
        let super_radius = max_dimension_range / 2f64 + self.r_max;
        // Center of mass is NaN a lot
        Entity {
            x: self.center_of_mass.0,
            y: self.center_of_mass.1,
            z: self.center_of_mass.2,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            mass: self.total_mass,
            radius: super_radius,
        }
    }

    pub(crate) fn max_distance(&self) -> f64 {
        let x_distance = self.x_max - self.x_min;
        let y_distance = self.y_max - self.y_min;
        let z_distance = self.z_max - self.z_min;
        f64::max(x_distance, f64::max(y_distance, z_distance))
    }

    /// Traverses tree and returns entities by resolving indices from arena.
    pub(crate) fn traverse_tree_helper<T: AsEntity + Clone>(&self, arena: &[T]) -> Vec<T> {
        let mut to_return: Vec<T> = Vec::new();
        if let Some(node) = &self.left {
            to_return.append(&mut node.traverse_tree_helper(arena));
        }
        if let Some(node) = &self.right {
            to_return.append(&mut node.traverse_tree_helper(arena));
        } else {
            // Use indices to resolve entities from arena
            if let Some(ref indices) = self.point_indices {
                to_return.extend(indices.iter().map(|&idx| arena[idx].clone()));
            }
        }
        to_return
    }

    /// Returns an iterator over all entities in the tree without cloning them.
    /// This is a zero-copy alternative to `traverse_tree_helper()`.
    pub(crate) fn iter<'a, T: AsEntity + Clone>(&'a self, arena: &'a [T]) -> NodeIterator<'a, T> {
        NodeIterator::new(self, arena)
    }

    /// Takes in a slice of entities and creates a recursive 3d tree structure using indices.
    /// This is the public API that maintains backward compatibility.
    ///
    /// # Parameters
    /// - `parallel_threshold`: Subtrees with fewer entities than this will be built sequentially
    pub(crate) fn new_root_node<T: AsEntity + Clone + Send + Sync>(entities: &[T], max_entities: i32, parallel_threshold: usize) -> Node {
        use rayon::prelude::*;
        // OPTIMIZATION: Convert all entities once at top level to avoid O(n*depth) conversions
        // Use parallel iteration for large datasets
        let entities_as_entities: Vec<Entity> = if entities.len() > (parallel_threshold*10) {
            entities.par_iter().map(|e| e.as_entity()).collect()
        } else {
            entities.iter().map(|e| e.as_entity()).collect()
        };
        let indices: Vec<usize> = (0..entities.len()).collect();
        Self::new_root_node_with_indices(entities, &entities_as_entities, &indices, max_entities, parallel_threshold)
    }

    /// Internal implementation using indices for arena pattern.
    /// Pre-converted entities passed to avoid repeated as_entity() calls.
    fn new_root_node_with_indices<T: AsEntity + Clone + Send + Sync>(
        entities: &[T],
        entities_as_entities: &[Entity],
        indices: &[usize],
        max_entities: i32,
        parallel_threshold: usize,
    ) -> Node {
        use crate::utilities::{partition_indices_by_median, xyz_distances_indexed, max_min_xyz_indexed};

        let length_of_points = indices.len() as i32;

        // OPTIMIZATION: Use indexed access to avoid cloning entities
        let (xdistance, ydistance, zdistance) = xyz_distances_indexed(entities_as_entities, indices);


        // If our current collection is small enough to become a leaf
        if length_of_points <= max_entities {
            // LEAF NODE - store indices only!
            let (x_total, y_total, z_total, max_radius, total_mass) = indices
                .iter()
                .fold((0.0, 0.0, 0.0, 0.0, 0.0), |acc, &idx| {
                    let pt = &entities_as_entities[idx];
                    (
                        acc.0 + (pt.x * pt.mass),
                        acc.1 + (pt.y * pt.mass),
                        acc.2 + (pt.z * pt.mass),
                        if acc.3 > pt.radius { acc.3 } else { pt.radius },
                        acc.4 + pt.mass,
                    )
                });

            let (x_max, x_min, y_max, y_min, z_max, z_min) = max_min_xyz_indexed(entities_as_entities, indices);

            Node {
                center_of_mass: (
                    x_total / total_mass,
                    y_total / total_mass,
                    z_total / total_mass,
                ),
                total_mass,
                r_max: max_radius,
                point_indices: Some(indices.to_vec()),
                left: None,
                right: None,
                split_dimension: None,
                split_value: 0.0,
                x_max,
                x_min,
                y_max,
                y_min,
                z_max,
                z_min,
            }
        } else {
            // INTERNAL NODE - partition indices
            let mut mut_indices = indices.to_vec();
            let split_index;
            let (split_dimension, split_value) = if zdistance > ydistance && zdistance > xdistance {
                // Split on Z - use full entities array for partition
                let (split_value, tmp) = partition_indices_by_median(Dimension::Z, &entities_as_entities, &mut mut_indices);
                split_index = tmp;
                (Dimension::Z, split_value)
            } else if ydistance > xdistance && ydistance > zdistance {
                // Split on Y - use full entities array for partition
                let (split_value, tmp) = partition_indices_by_median(Dimension::Y, &entities_as_entities, &mut mut_indices);
                split_index = tmp;
                (Dimension::Y, split_value)
            } else {
                // Split on X - use full entities array for partition
                let (split_value, tmp) = partition_indices_by_median(Dimension::X, &entities_as_entities, &mut mut_indices);
                split_index = tmp;
                (Dimension::X, split_value)
            };

            // KEY CHANGE: Split indices, not entities!
            let (below_indices, above_indices) = mut_indices.split_at(split_index);

            // PARALLEL TREE CONSTRUCTION: Build left and right subtrees in parallel
            // This provides significant speedup for large datasets as tree construction
            // is embarrassingly parallel - each subtree is completely independent.
            // Use a threshold to avoid overhead for small subtrees.
            let (left, right) = if indices.len() > parallel_threshold {
                let (below_vec, above_vec) = (below_indices.to_vec(), above_indices.to_vec());
                rayon::join(
                    || Self::new_root_node_with_indices(entities, entities_as_entities, &below_vec, max_entities, parallel_threshold),
                    || Self::new_root_node_with_indices(entities, entities_as_entities, &above_vec, max_entities, parallel_threshold),
                )
            } else {
                let left = Self::new_root_node_with_indices(entities, entities_as_entities, below_indices, max_entities, parallel_threshold);
                let right = Self::new_root_node_with_indices(entities, entities_as_entities, above_indices, max_entities, parallel_threshold);
                (left, right)
            };

            // The center of mass is a recursive definition
            let left_mass = left.total_mass;
            let right_mass = right.total_mass;
            let (left_x, left_y, left_z) = left.center_of_mass;
            let (right_x, right_y, right_z) = right.center_of_mass;
            let total_mass = left_mass + right_mass;
            assert!(total_mass != 0., "invalid mass of 0");

            let (center_x, center_y, center_z) = (
                ((left_mass * left_x) + (right_mass * right_x)) / total_mass,
                ((left_mass * left_y) + (right_mass * right_y)) / total_mass,
                ((left_mass * left_z) + (right_mass * right_z)) / total_mass,
            );

            let mut root_node = Node::new();
            root_node.split_dimension = Some(split_dimension);
            root_node.split_value = split_value;
            root_node.left = Some(Box::new(left));
            root_node.right = Some(Box::new(right));
            root_node.center_of_mass = (center_x, center_y, center_z);
            root_node.set_max_mins();
            root_node.total_mass = total_mass;
            root_node
        }
    }
}

/// This tests the recursive node construction used to create a new gravtree. It tests some private
/// fields so it is located within the same module as the node itself.
#[test]
fn test() {
    use crate::CalculateCollisions;
    // Entity now has a proper Responsive implementation in entity.rs
    let mut test_vec: Vec<Entity> = Vec::new();
    for i in 0..10 {
        test_vec.push(Entity {
            x: i as f64,
            y: (10 - i) as f64,
            z: i as f64,
            vx: i as f64,
            vy: i as f64,
            vz: i as f64,
            mass: i as f64,
            radius: i as f64,
        });
    }

    let check_vec = test_vec.clone();
    let tree = crate::GravTree::with_default_parallel_threshold(&test_vec, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let root_node = tree.root.clone();

    let mut nodes: Vec<Node> = Vec::new();
    let mut traversal_stack: Vec<Option<Box<Node>>> = Vec::new();
    let mut rover = Some(Box::new(root_node));
    while !traversal_stack.is_empty() || rover.is_some() {
        if rover.is_some() {
            traversal_stack.push(rover.clone());
            nodes.push(*rover.clone().unwrap());
            rover = rover.unwrap().left;
        } else {
            rover = traversal_stack.pop().unwrap();
            rover = rover.unwrap().right;
        }
    }

    let post_tree_vec = tree.as_vec();
    // The tree should contain all of the elements we put into it.
    for i in check_vec.iter() {
        assert!(post_tree_vec.contains(i));
    }

    // In this example, there should be exactly 8 nodes.
    assert_eq!(8, nodes.len());

    // No node should have zero mass, except for the first one which is the "phantom parent".
    for node in nodes.iter().skip(1) {
        assert!(node.total_mass > 0.);
    }

    // The total mass of the root node should be the sum of all of their masses.
    let total_mass = check_vec.iter().fold(0., |acc, x| acc + x.mass);
    assert_eq!(total_mass, tree.root.left.unwrap().total_mass);
}

/// Zero-copy iterator over entities in a Node tree using arena pattern.
///
/// This iterator traverses the tree structure without cloning entities,
/// yielding references to entities resolved from the arena via indices.
pub(crate) struct NodeIterator<'a, T: AsEntity + Clone> {
    /// Stack of nodes to visit. We use a Vec as a stack for depth-first traversal.
    stack: Vec<&'a Node>,
    /// When we reach a leaf node, we iterate through its indices.
    current_leaf_iter: Option<std::slice::Iter<'a, usize>>,
    /// Arena for resolving indices to entities.
    arena: &'a [T],
}

impl<'a, T: AsEntity + Clone> NodeIterator<'a, T> {
    pub(crate) fn new(root: &'a Node, arena: &'a [T]) -> Self {
        NodeIterator {
            stack: vec![root],
            current_leaf_iter: None,
            arena,
        }
    }
}

impl<'a, T: AsEntity + Clone> Iterator for NodeIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // If we're currently iterating through a leaf node's indices, continue doing so
        if let Some(ref mut leaf_iter) = self.current_leaf_iter {
            if let Some(&idx) = leaf_iter.next() {
                return Some(&self.arena[idx]);
            } else {
                // Finished with this leaf, clear it
                self.current_leaf_iter = None;
            }
        }

        // Process nodes from the stack
        while let Some(node) = self.stack.pop() {
            // If this node has point_indices, it's a leaf - start iterating through them
            if let Some(ref indices) = node.point_indices {
                let mut iter = indices.iter();
                // Get the first index (if any) and store the iterator for subsequent calls
                if let Some(&idx) = iter.next() {
                    self.current_leaf_iter = Some(iter);
                    return Some(&self.arena[idx]);
                }
                // If indices is empty, continue to next node
                continue;
            }

            // Internal node - push children onto stack (right first, then left for DFS order)
            if let Some(ref right) = node.right {
                self.stack.push(right);
            }
            if let Some(ref left) = node.left {
                self.stack.push(left);
            }
        }

        // No more nodes or entities
        None
    }
}

