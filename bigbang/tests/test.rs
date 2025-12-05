extern crate bigbang;
use bigbang::{AsEntity, CalculateCollisions, Entity, GravTree, Responsive, SimulationResult};

#[derive(Clone, PartialEq)]
struct MyEntity {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    radius: f64,
}

impl AsEntity for MyEntity {
    fn as_entity(&self) -> Entity {
        return Entity {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
            radius: self.radius,
            mass: if self.radius < 1. { 0.5 } else { 105. },
        };
    }
}

impl Responsive for MyEntity {
    fn respond(&self, simulation_result: SimulationResult<MyEntity>, time_step: f64) -> Self {
        let (ax, ay, _az) = simulation_result.gravitational_acceleration;
        let (x, y, z) = (self.x, self.y, self.z);
        let (mut vx, mut vy, mut vz) = (self.vx, self.vy, self.vz);
        let self_mass = if self.radius < 1. { 0.5 } else { 105. };
        // calculate the collisions
        for other in simulation_result.collisions.clone() {
            let other_mass = if other.radius < 1. { 0.5 } else { 105. };
            let mass_coefficient_v1 = (self_mass - other_mass) / (self_mass + other_mass);
            let mass_coefficient_v2 = (2f64 * other_mass) / (self_mass + other_mass);
            vx = (mass_coefficient_v1 * vx) + (mass_coefficient_v2 * other.vx);
            vy = (mass_coefficient_v1 * vy) + (mass_coefficient_v2 * other.vy);
            vz = (mass_coefficient_v1 * vz) + (mass_coefficient_v2 * other.vz);
        }
        vx += ax * time_step;
        vy += ay * time_step;
        MyEntity {
            vx,
            vy,
            vz,
            x: x + (vx * time_step),
            y: y + (vy * time_step),
            z: z + (vz * time_step),
            radius: self.radius,
        }
    }
}

impl MyEntity {
    pub fn random_entity() -> MyEntity {
        MyEntity {
            vx: 0f64,
            vy: 0f64,
            vz: 0f64,
            x: rand::random::<f64>() * 50f64,
            y: rand::random::<f64>() * 50f64,
            z: rand::random::<f64>() * 50f64,
            radius: rand::random::<f64>() / 10f64,
        }
    }
}

#[test]
fn test_traversal() {
    let mut vec: Vec<MyEntity> = Vec::new();
    for _ in 0..100 {
        let entity = MyEntity::random_entity();
        vec.push(entity);
    }
    let vec_clone = vec.clone();
    let tree = GravTree::new(&vec, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let traversed_vec = tree.as_vec();
    let mut all_found = true;
    for i in vec_clone {
        if !traversed_vec.contains(&i) {
            all_found = false;
        }
    }

    assert!(all_found);
}

#[test]
fn test_time_step() {
    let mut vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = Vec::new();
    for _ in 0..1000 {
        let entity = MyEntity::random_entity();
        vec_that_wants_to_be_a_kdtree.push(entity);
    }

    let test_tree = GravTree::new(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let after_time_step = test_tree.time_step();
    assert_eq!(after_time_step.as_vec().len(), 1000);
}

#[test]
fn test_time_step_mut() {
    let mut vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = Vec::new();
    for _ in 0..1000 {
        let entity = MyEntity::random_entity();
        vec_that_wants_to_be_a_kdtree.push(entity);
    }

    let mut test_tree = GravTree::new(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
    test_tree.time_step_mut();
    assert_eq!(test_tree.as_vec().len(), 1000);
    assert_eq!(test_tree.get_number_of_entities(), 1000);
}

#[test]
fn test_time_step_consistency() {
    // Verify that time_step_mut produces the same results as time_step
    let mut vec: Vec<MyEntity> = Vec::new();
    for i in 0..100 {
        vec.push(MyEntity {
            x: (i as f64) * 2.0,
            y: (i as f64) * 2.0,
            z: (i as f64) * 2.0,
            vx: 0.1,
            vy: 0.1,
            vz: 0.1,
            radius: 1.0,
        });
    }

    let tree1 = GravTree::new(&vec, 0.2, 3, 0.2, CalculateCollisions::No);
    let tree2 = GravTree::new(&vec, 0.2, 3, 0.2, CalculateCollisions::No);

    // Using immutable time_step
    let result1 = tree1.time_step();
    let entities1 = result1.as_vec();

    // Using mutable time_step_mut
    let mut tree2_mut = tree2;
    tree2_mut.time_step_mut();
    let entities2 = tree2_mut.as_vec();

    // Both should have the same number of entities
    assert_eq!(entities1.len(), entities2.len());
    assert_eq!(result1.get_number_of_entities(), tree2_mut.get_number_of_entities());

    // The positions should be approximately the same
    // (We can't do exact equality due to potential floating point differences in tree traversal order)
    let sum_x1: f64 = entities1.iter().map(|e| e.x).sum();
    let sum_x2: f64 = entities2.iter().map(|e| e.x).sum();
    assert!((sum_x1 - sum_x2).abs() < 0.001, "X positions differ: {} vs {}", sum_x1, sum_x2);
}

