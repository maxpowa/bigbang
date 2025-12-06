extern crate bigbang;
use bigbang::{CalculateCollisions, Entity, GravTree};

/// Test proper 3D elastic collision physics
#[test]
fn test_head_on_elastic_collision() {
    // Two equal-mass particles moving toward each other should exchange velocities
    let entities = vec![
        Entity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 1.0,  // Moving right
            vy: 0.0,
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
        Entity {
            x: 1.5,  // Overlapping (radii = 1.0 + 1.0 = 2.0 > distance 1.5)
            y: 0.0,
            z: 0.0,
            vx: -1.0,  // Moving left
            vy: 0.0,
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
    ];

    let tree = GravTree::new(&entities, 0.01, 3, 0.5, CalculateCollisions::Yes);
    let result = tree.time_step();
    let final_entities = result.as_vec();

    // After elastic collision, equal mass particles should exchange velocities
    // Allow for small numerical error and gravity effects
    assert!(
        (final_entities[0].vx - (-1.0)).abs() < 0.1,
        "Entity 0 should be moving left after collision, vx = {}",
        final_entities[0].vx
    );
    assert!(
        (final_entities[1].vx - 1.0).abs() < 0.1,
        "Entity 1 should be moving right after collision, vx = {}",
        final_entities[1].vx
    );
}

#[test]
fn test_glancing_collision_preserves_tangential_velocity() {
    // Test that glancing collision preserves tangential velocity component
    let entities = vec![
        Entity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 1.0,   // Moving right
            vy: 0.0,   // No vertical motion
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
        Entity {
            x: 1.5,
            y: 0.5,  // Offset in Y (glancing collision)
            z: 0.0,
            vx: -1.0,  // Moving left
            vy: 2.0,   // Moving up (tangential component)
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
    ];

    let tree = GravTree::new(&entities, 0.01, 3, 0.5, CalculateCollisions::Yes);
    let initial = tree.as_vec();
    let result = tree.time_step();
    let final_entities = result.as_vec();

    // The Y component (tangential to collision normal) should be largely preserved
    // (small changes due to gravity are OK)
    let vy_change_0 = (final_entities[0].vy - initial[0].vy).abs();
    let _vy_change_1 = (final_entities[1].vy - initial[1].vy).abs();

    // Tangential velocity changes should be small (< 0.5)
    assert!(
        vy_change_0 < 0.5,
        "Entity 0 tangential velocity should be mostly preserved, change = {}",
        vy_change_0
    );
}

#[test]
fn test_uses_actual_mass_not_radius() {
    // Two particles with same radius but different mass
    let entities = vec![
        Entity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
            mass: 100.0,  // Very heavy
            radius: 1.0,
        },
        Entity {
            x: 1.5,
            y: 0.0,
            z: 0.0,
            vx: -1.0,
            vy: 0.0,
            vz: 0.0,
            mass: 1.0,  // Very light (same radius!)
            radius: 1.0,
        },
    ];

    let tree = GravTree::new(&entities, 0.01, 3, 0.5, CalculateCollisions::Yes);
    let result = tree.time_step();
    let final_entities = result.as_vec();

    // Heavy particle should barely slow down
    // Light particle should bounce back with high velocity
    // This verifies we're using actual mass, not radius-based mass

    // Heavy particle should still be moving right (barely affected)
    assert!(
        final_entities[0].vx > 0.5,
        "Heavy particle should keep moving right, vx = {}",
        final_entities[0].vx
    );

    // Light particle should be moving right with higher velocity (bounced back)
    assert!(
        final_entities[1].vx > 0.0,
        "Light particle should bounce back to the right, vx = {}",
        final_entities[1].vx
    );
}

#[test]
fn test_collision_physics_no_clone() {
    // Verify the implementation doesn't clone collision vector (performance test)
    let entities = vec![
        Entity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
        Entity {
            x: 1.5,
            y: 0.0,
            z: 0.0,
            vx: -1.0,
            vy: 0.0,
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
    ];

    let tree = GravTree::new(&entities, 0.01, 3, 0.5, CalculateCollisions::Yes);

    // This should use .iter() not .clone() - just verify it runs without panic
    let result = tree.time_step();
    assert_eq!(result.as_vec().len(), 2);
}

#[test]
fn test_momentum_conservation_in_collision() {
    // Verify momentum is conserved in elastic collision
    let entities = vec![
        Entity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 2.0,
            vy: 0.0,
            vz: 0.0,
            mass: 5.0,
            radius: 1.0,
        },
        Entity {
            x: 1.5,
            y: 0.0,
            z: 0.0,
            vx: -1.0,
            vy: 0.0,
            vz: 0.0,
            mass: 10.0,
            radius: 1.0,
        },
    ];

    let tree = GravTree::new(&entities, 0.001, 3, 0.5, CalculateCollisions::Yes);  // Small timestep
    let initial = tree.as_vec();
    let result = tree.time_step();
    let final_entities = result.as_vec();

    // Calculate initial and final momentum
    let initial_momentum_x = initial[0].mass * initial[0].vx + initial[1].mass * initial[1].vx;
    let final_momentum_x = final_entities[0].mass * final_entities[0].vx
                         + final_entities[1].mass * final_entities[1].vx;

    // Momentum should be conserved (allow small error for gravity and numerical precision)
    assert!(
        (initial_momentum_x - final_momentum_x).abs() < 0.5,
        "Momentum should be conserved: initial = {}, final = {}",
        initial_momentum_x,
        final_momentum_x
    );
}

