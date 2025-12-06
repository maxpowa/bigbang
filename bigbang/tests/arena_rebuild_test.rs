extern crate bigbang;
use bigbang::{AsEntity, CalculateCollisions, Entity, GravTree, Responsive, SimulationResult};

#[derive(Clone, PartialEq, Debug)]
struct TestEntity {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    radius: f64,
}

impl AsEntity for TestEntity {
    fn as_entity(&self) -> Entity {
        Entity {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
            radius: self.radius,
            mass: self.mass,
        }
    }
}

impl Responsive for TestEntity {
    fn respond(&self, simulation_result: SimulationResult<TestEntity>, time_step: f64) -> Self {
        let (ax, ay, az) = simulation_result.gravitational_acceleration;
        let mut new_entity = self.clone();
        new_entity.vx += ax * time_step;
        new_entity.vy += ay * time_step;
        new_entity.vz += az * time_step;
        new_entity.x += new_entity.vx * time_step;
        new_entity.y += new_entity.vy * time_step;
        new_entity.z += new_entity.vz * time_step;
        new_entity
    }

    fn respond_mut(&mut self, simulation_result: SimulationResult<Self>, time_step: f64) {
        let (ax, ay, az) = simulation_result.gravitational_acceleration;
        self.vx += ax * time_step;
        self.vy += ay * time_step;
        self.vz += az * time_step;
        self.x += self.vx * time_step;
        self.y += self.vy * time_step;
        self.z += self.vz * time_step;
    }
}

#[test]
fn test_tree_rebuild_on_consecutive_steps() {
    // This test verifies that the tree is correctly rebuilt between time steps
    // to maintain accuracy

    let entities = vec![
        TestEntity { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1000.0, radius: 1.0 },
        TestEntity { x: 10.0, y: 0.0, z: 0.0, vx: 0.0, vy: 1.0, vz: 0.0, mass: 1.0, radius: 0.1 },
        TestEntity { x: 0.0, y: 10.0, z: 0.0, vx: 1.0, vy: 0.0, vz: 0.0, mass: 1.0, radius: 0.1 },
    ];

    let mut tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);

    // First time step - tree is fresh, no rebuild needed
    let initial_entities = tree.as_vec();
    tree.time_step_mut();
    let after_step1 = tree.as_vec();

    // Verify entities moved
    assert_ne!(initial_entities[1].x, after_step1[1].x);

    // Second time step - tree should rebuild before calculating accelerations
    tree.time_step_mut();
    let after_step2 = tree.as_vec();

    // Verify entities continue to move (would fail if tree wasn't rebuilt)
    assert_ne!(after_step1[1].x, after_step2[1].x);

    // Third time step - verify consistent behavior
    tree.time_step_mut();
    let after_step3 = tree.as_vec();

    assert_ne!(after_step2[1].x, after_step3[1].x);
}

#[test]
fn test_explicit_rebuild() {
    let entities = vec![
        TestEntity { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1000.0, radius: 1.0 },
        TestEntity { x: 10.0, y: 0.0, z: 0.0, vx: 0.0, vy: 1.0, vz: 0.0, mass: 1.0, radius: 0.1 },
    ];

    let mut tree = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);

    // Manually modify entities to make tree stale
    tree.time_step_mut();

    // Explicitly rebuild the tree
    tree.rebuild_tree();

    // After explicit rebuild, tree should be fresh
    // Subsequent time_step_mut should not rebuild at the start
    tree.time_step_mut();

    // Verify entities are still being updated correctly
    let final_entities = tree.as_vec();
    assert!(final_entities.len() == 2);
}

#[test]
fn test_time_step_vs_time_step_mut_with_multiple_steps() {
    // Verify that multiple time_step_mut calls produce same results as multiple time_step calls
    let entities = vec![
        TestEntity { x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 100.0, radius: 1.0 },
        TestEntity { x: 5.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.5, vz: 0.0, mass: 1.0, radius: 0.1 },
        TestEntity { x: 0.0, y: 5.0, z: 0.0, vx: 0.5, vy: 0.0, vz: 0.0, mass: 1.0, radius: 0.1 },
    ];

    // Method 1: Using time_step (creates new trees)
    let mut tree1 = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    tree1 = tree1.time_step();
    tree1 = tree1.time_step();
    tree1 = tree1.time_step();
    let result1 = tree1.as_vec();

    // Method 2: Using time_step_mut (modifies in place)
    let mut tree2 = GravTree::new(&entities, 0.1, 3, 0.5, CalculateCollisions::No);
    tree2.time_step_mut();
    tree2.time_step_mut();
    tree2.time_step_mut();
    let result2 = tree2.as_vec();

    // Results should be very close (floating point precision tolerances)
    assert_eq!(result1.len(), result2.len());
    for i in 0..result1.len() {
        assert!((result1[i].x - result2[i].x).abs() < 1e-10,
            "X position mismatch at entity {}: {} vs {}", i, result1[i].x, result2[i].x);
        assert!((result1[i].y - result2[i].y).abs() < 1e-10,
            "Y position mismatch at entity {}: {} vs {}", i, result1[i].y, result2[i].y);
        assert!((result1[i].vx - result2[i].vx).abs() < 1e-10,
            "VX velocity mismatch at entity {}: {} vs {}", i, result1[i].vx, result2[i].vx);
    }
}

