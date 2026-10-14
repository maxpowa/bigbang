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
        return Entity::new(
            self.vx,
            self.vy,
            self.vz,
            self.x,
            self.y,
            self.z,
            self.radius,
            if self.radius < 1. { 0.5 } else { 105. },
        );
    }
}

impl Responsive for MyEntity {
    fn respond(&self, simulation_result: SimulationResult<MyEntity>, time_step: f64) -> Self {
        let (ax, ay, _az) = simulation_result.gravitational_acceleration;
        let (x, y, z) = (self.x, self.y, self.z);
        let (mut vx, mut vy, mut vz) = (self.vx, self.vy, self.vz);
        
        // Get actual mass from as_entity (proper mass calculation)
        let entity = self.as_entity();
        let self_mass = entity.mass;
        
        // Handle collisions with proper 3D elastic collision physics
        for other in simulation_result.collisions.iter() {
            let other_entity = other.as_entity();
            let other_mass = other_entity.mass;
            
            let dx = other.x - self.x;
            let dy = other.y - self.y;
            let dz = other.z - self.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            
            if dist < 1e-10 {
                continue;
            }
            
            let nx = dx / dist;
            let ny = dy / dist;
            let nz = dz / dist;
            
            let v1n = vx * nx + vy * ny + vz * nz;
            let v2n = other.vx * nx + other.vy * ny + other.vz * nz;
            
            let mass_sum = self_mass + other_mass;
            let v1n_new = ((self_mass - other_mass) * v1n + 2.0 * other_mass * v2n) / mass_sum;
            
            let delta_vn = v1n_new - v1n;
            vx += delta_vn * nx;
            vy += delta_vn * ny;
            vz += delta_vn * nz;
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
    let tree = GravTree::with_default_parallel_threshold(&vec, 0.2, 3, 0.2, CalculateCollisions::Yes);
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

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
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

    let mut test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
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

    let tree1 = GravTree::with_default_parallel_threshold(&vec, 0.2, 3, 0.2, CalculateCollisions::No);
    let tree2 = GravTree::with_default_parallel_threshold(&vec, 0.2, 3, 0.2, CalculateCollisions::No);

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

// Regression test for C2: time_step_mut() must deliver the collision partners it
// computed in SimulationResult.collisions to Responsive::respond_mut(). Before the fix,
// the collision list was discarded during the in-place step, so colliding bodies
// passed through each other and never saw a collision.

#[derive(Clone, PartialEq, Debug)]
struct CollisionCountingEntity {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    radius: f64,
    mass: f64,
    /// Number of collision partners this entity saw during the most recent step.
    collisions_seen: usize,
}

impl AsEntity for CollisionCountingEntity {
    fn as_entity(&self) -> Entity {
        Entity::new(
            self.vx,
            self.vy,
            self.vz,
            self.x,
            self.y,
            self.z,
            self.radius,
            self.mass,
        )
    }
}

impl Responsive for CollisionCountingEntity {
    fn respond(&self, simulation_result: SimulationResult<CollisionCountingEntity>, time_step: f64) -> Self {
        let mut new_self = self.clone();
        new_self.collisions_seen = simulation_result.collisions.len();

        // Equal-mass elastic collision response: swap the velocity components along the
        // collision normal for each partner. (Mirrors the physics in MyEntity::respond.)
        for other in &simulation_result.collisions {
            let dx = other.x - self.x;
            let dy = other.y - self.y;
            let dz = other.z - self.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1e-10 {
                continue;
            }
            let nx = dx / dist;
            let ny = dy / dist;
            let nz = dz / dist;
            let v1n = self.vx * nx + self.vy * ny + self.vz * nz;
            let v2n = other.vx * nx + other.vy * ny + other.vz * nz;
            // Equal masses: the normal components are exchanged.
            let delta_vn = v2n - v1n;
            new_self.vx += delta_vn * nx;
            new_self.vy += delta_vn * ny;
            new_self.vz += delta_vn * nz;
        }

        // Euler integration of gravity and position.
        let (ax, ay, az) = simulation_result.gravitational_acceleration;
        new_self.vx += ax * time_step;
        new_self.vy += ay * time_step;
        new_self.vz += az * time_step;
        new_self.x += new_self.vx * time_step;
        new_self.y += new_self.vy * time_step;
        new_self.z += new_self.vz * time_step;
        new_self
    }
}

#[test]
fn test_time_step_mut_reports_collisions() {
    // Two bodies moving toward each other, overlapping (distance 1.5 < sum of radii 2.0),
    // so they collide. With the C2 fix, time_step_mut() must apply the collision:
    // an equal-mass elastic swap flips both vx signs. Without the fix, collisions were
    // discarded and the bodies just kept their original velocities (plus a small gravity
    // kick), so every assertion below would fail.
    let a = CollisionCountingEntity {
        x: 0.0, y: 0.0, z: 0.0,
        vx: 1.0, vy: 0.0, vz: 0.0,
        radius: 1.0, mass: 10.0,
        collisions_seen: 0,
    };
    let b = CollisionCountingEntity {
        x: 1.5, y: 0.0, z: 0.0,
        vx: -1.0, vy: 0.0, vz: 0.0,
        radius: 1.0, mass: 10.0,
        collisions_seen: 0,
    };
    let entities = vec![a, b];

    let mut tree = GravTree::with_default_parallel_threshold(&entities, 0.01, 3, 0.5, CalculateCollisions::Yes);
    tree.time_step_mut();

    let out = tree.as_vec();
    // Identify by position rather than arena order (A stays near x=0, B stays near x=1.5).
    let a_out = out.iter().find(|e| e.x < 1.0).expect("entity A (x < 1.0)");
    let b_out = out.iter().find(|e| e.x > 1.0).expect("entity B (x > 1.0)");

    // Post-fix: the collision flipped each body's velocity (elastic swap dominates gravity).
    assert!(
        a_out.vx < 0.0,
        "A.vx should be negative after collision swap, got {} (C2: collisions discarded in time_step_mut?)",
        a_out.vx,
    );
    assert!(
        b_out.vx > 0.0,
        "B.vx should be positive after collision swap, got {} (C2: collisions discarded in time_step_mut?)",
        b_out.vx,
    );
    // Post-fix: each entity saw exactly one collision partner during the step.
    assert_eq!(
        a_out.collisions_seen, 1,
        "A should have seen 1 collision, saw {} (C2: collisions discarded in time_step_mut?)",
        a_out.collisions_seen,
    );
    assert_eq!(
        b_out.collisions_seen, 1,
        "B should have seen 1 collision, saw {} (C2: collisions discarded in time_step_mut?)",
        b_out.collisions_seen,
    );
}

