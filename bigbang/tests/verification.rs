/// Formal verification tests for the bigbang n-body simulation library
///
/// This test suite compares the Barnes-Hut tree approximation against:
/// 1. A direct O(n²) brute-force n-body simulator (reference implementation)
/// 2. Known analytical solutions for simple cases
/// 3. Physical conservation laws (energy, momentum, angular momentum)

extern crate bigbang;
use bigbang::{AsEntity, CalculateCollisions, Entity, GravTree, Responsive, SimulationResult};

// The bigbang library uses normalized units where G = 1
// This is a common approach in astrophysical simulations to avoid numerical issues
const G: f64 = 1.0;
const EPSILON: f64 = 1e-6;   // Small value to avoid division by zero

#[derive(Clone, PartialEq, Debug)]
struct VerifiableEntity {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    radius: f64,
    mass: f64,
}

impl AsEntity for VerifiableEntity {
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

impl Responsive for VerifiableEntity {
    fn respond(&self, simulation_result: SimulationResult<Self>, time_step: f64) -> Self {
        let (ax, ay, az) = simulation_result.gravitational_acceleration;

        // Simple Euler integration
        let vx = self.vx + ax * time_step;
        let vy = self.vy + ay * time_step;
        let vz = self.vz + az * time_step;

        VerifiableEntity {
            x: self.x + vx * time_step,
            y: self.y + vy * time_step,
            z: self.z + vz * time_step,
            vx,
            vy,
            vz,
            radius: self.radius,
            mass: self.mass,
        }
    }
}

impl VerifiableEntity {
    fn new(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64, radius: f64) -> Self {
        VerifiableEntity {
            x, y, z, vx, vy, vz, mass, radius,
        }
    }

    fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
    }

    fn momentum(&self) -> (f64, f64, f64) {
        (self.mass * self.vx, self.mass * self.vy, self.mass * self.vz)
    }

    fn distance_to(&self, other: &VerifiableEntity) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Direct O(n²) n-body simulation - reference implementation
/// This is guaranteed to be correct (no approximations) but slow
struct DirectNBodySimulator {
    entities: Vec<VerifiableEntity>,
    time_step: f64,
}

impl DirectNBodySimulator {
    fn new(entities: Vec<VerifiableEntity>, time_step: f64) -> Self {
        DirectNBodySimulator { entities, time_step }
    }

    /// Calculate acceleration on entity i due to all other entities
    fn calculate_acceleration(&self, i: usize) -> (f64, f64, f64) {
        let mut ax = 0.0;
        let mut ay = 0.0;
        let mut az = 0.0;

        let entity = &self.entities[i];

        for (j, other) in self.entities.iter().enumerate() {
            if i == j {
                continue;
            }

            let dx = other.x - entity.x;
            let dy = other.y - entity.y;
            let dz = other.z - entity.z;

            let dist_sq = dx * dx + dy * dy + dz * dz + EPSILON * EPSILON;
            let dist = dist_sq.sqrt();

            // F = G * m1 * m2 / r²
            // a = F / m1 = G * m2 / r²
            let force_magnitude = G * other.mass / dist_sq;

            // Acceleration components
            ax += force_magnitude * dx / dist;
            ay += force_magnitude * dy / dist;
            az += force_magnitude * dz / dist;
        }

        (ax, ay, az)
    }

    /// Perform one time step using direct n-body calculation
    fn time_step(&mut self) {
        let mut accelerations = Vec::new();

        // Calculate all accelerations
        for i in 0..self.entities.len() {
            accelerations.push(self.calculate_acceleration(i));
        }

        // Update all entities
        for (i, (ax, ay, az)) in accelerations.iter().enumerate() {
            let entity = &mut self.entities[i];
            entity.vx += ax * self.time_step;
            entity.vy += ay * self.time_step;
            entity.vz += az * self.time_step;
            entity.x += entity.vx * self.time_step;
            entity.y += entity.vy * self.time_step;
            entity.z += entity.vz * self.time_step;
        }
    }

    fn get_entities(&self) -> &[VerifiableEntity] {
        &self.entities
    }
}

/// Calculate total momentum of a system
fn total_momentum(entities: &[VerifiableEntity]) -> (f64, f64, f64) {
    entities.iter().fold((0.0, 0.0, 0.0), |(px, py, pz), e| {
        let (epx, epy, epz) = e.momentum();
        (px + epx, py + epy, pz + epz)
    })
}

/// Calculate total kinetic energy of a system
fn total_kinetic_energy(entities: &[VerifiableEntity]) -> f64 {
    entities.iter().map(|e| e.kinetic_energy()).sum()
}

/// Calculate total potential energy of a system
fn total_potential_energy(entities: &[VerifiableEntity]) -> f64 {
    let mut potential = 0.0;

    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let dist = entities[i].distance_to(&entities[j]) + EPSILON;
            potential -= G * entities[i].mass * entities[j].mass / dist;
        }
    }

    potential
}

/// Calculate total energy (kinetic + potential)
fn total_energy(entities: &[VerifiableEntity]) -> f64 {
    total_kinetic_energy(entities) + total_potential_energy(entities)
}

/// Calculate angular momentum of a system
fn total_angular_momentum(entities: &[VerifiableEntity]) -> (f64, f64, f64) {
    entities.iter().fold((0.0, 0.0, 0.0), |(lx, ly, lz), e| {
        // L = r × p = r × (m * v)
        let lpx = e.mass * (e.y * e.vz - e.z * e.vy);
        let lpy = e.mass * (e.z * e.vx - e.x * e.vz);
        let lpz = e.mass * (e.x * e.vy - e.y * e.vx);
        (lx + lpx, ly + lpy, lz + lpz)
    })
}

// ============================================================================
// VERIFICATION TESTS
// ============================================================================

#[test]
fn test_two_body_problem_against_direct() {
    // Two-body problem: two equal masses
    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0, 1.0),
        VerifiableEntity::new(100.0, 0.0, 0.0, 0.0, -1.0, 0.0, 100.0, 1.0),
    ];

    let time_step = 0.1;
    let num_steps = 10;

    // Run direct simulation
    let mut direct_sim = DirectNBodySimulator::new(entities.clone(), time_step);
    for _ in 0..num_steps {
        direct_sim.time_step();
    }

    // Run Barnes-Hut simulation with very small theta (should be nearly exact)
    let mut tree = GravTree::new(&entities, time_step, 3, 0.01, CalculateCollisions::No);
    for _ in 0..num_steps {
        tree = tree.time_step();
    }

    let direct_entities = direct_sim.get_entities();
    let tree_entities = tree.as_vec();

    // Compare positions and velocities
    for i in 0..2 {
        let pos_diff = ((tree_entities[i].x - direct_entities[i].x).powi(2)
            + (tree_entities[i].y - direct_entities[i].y).powi(2)
            + (tree_entities[i].z - direct_entities[i].z).powi(2))
            .sqrt();

        let vel_diff = ((tree_entities[i].vx - direct_entities[i].vx).powi(2)
            + (tree_entities[i].vy - direct_entities[i].vy).powi(2)
            + (tree_entities[i].vz - direct_entities[i].vz).powi(2))
            .sqrt();

        println!("Entity {}: pos_diff = {}, vel_diff = {}", i, pos_diff, vel_diff);

        // With small theta, differences should be very small
        assert!(pos_diff < 1.0, "Position difference too large: {}", pos_diff);
        assert!(vel_diff < 0.1, "Velocity difference too large: {}", vel_diff);
    }
}

#[test]
fn test_three_body_against_direct() {
    // Three-body problem with different masses
    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 100.0, 1.0),
        VerifiableEntity::new(50.0, 0.0, 0.0, 0.0, -0.5, 0.0, 100.0, 1.0),
        VerifiableEntity::new(25.0, 50.0, 0.0, 0.5, 0.0, 0.0, 50.0, 0.5),
    ];

    let time_step = 0.05;
    let num_steps = 5;

    // Run direct simulation
    let mut direct_sim = DirectNBodySimulator::new(entities.clone(), time_step);
    for _ in 0..num_steps {
        direct_sim.time_step();
    }

    // Run Barnes-Hut simulation
    let mut tree = GravTree::new(&entities, time_step, 3, 0.05, CalculateCollisions::No);
    for _ in 0..num_steps {
        tree = tree.time_step();
    }

    let direct_entities = direct_sim.get_entities();
    let tree_entities = tree.as_vec();

    // Compare positions
    for i in 0..3 {
        let pos_diff = ((tree_entities[i].x - direct_entities[i].x).powi(2)
            + (tree_entities[i].y - direct_entities[i].y).powi(2)
            + (tree_entities[i].z - direct_entities[i].z).powi(2))
            .sqrt();

        println!("Entity {}: pos_diff = {}", i, pos_diff);
        assert!(pos_diff < 2.0, "Position difference too large: {}", pos_diff);
    }
}

#[test]
fn test_momentum_conservation() {
    // Momentum should be conserved in an isolated system
    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 100.0, 1.0),
        VerifiableEntity::new(100.0, 0.0, 0.0, -0.5, 0.3, 0.0, 100.0, 1.0),
        VerifiableEntity::new(50.0, 100.0, 0.0, -0.5, -0.8, 0.0, 100.0, 1.0),
    ];

    let initial_momentum = total_momentum(&entities);
    let initial_p_mag = (initial_momentum.0.powi(2) + initial_momentum.1.powi(2) + initial_momentum.2.powi(2)).sqrt();
    println!("Initial momentum: {:?}, magnitude: {}", initial_momentum, initial_p_mag);

    let time_step = 0.1;
    let mut tree = GravTree::new(&entities, time_step, 3, 0.2, CalculateCollisions::No);

    // Run simulation for several steps
    for step in 0..20 {
        tree = tree.time_step();
        let entities = tree.as_vec();
        let current_momentum = total_momentum(&entities);

        let momentum_error = (
            (current_momentum.0 - initial_momentum.0).abs(),
            (current_momentum.1 - initial_momentum.1).abs(),
            (current_momentum.2 - initial_momentum.2).abs(),
        );

        let relative_error = if initial_p_mag > EPSILON {
            (momentum_error.0.powi(2) + momentum_error.1.powi(2) + momentum_error.2.powi(2)).sqrt() / initial_p_mag
        } else {
            (momentum_error.0.powi(2) + momentum_error.1.powi(2) + momentum_error.2.powi(2)).sqrt()
        };

        println!("Step {}: momentum = {:?}, error = {:?}, relative = {}", step, current_momentum, momentum_error, relative_error);

        // Momentum should be conserved (allowing for numerical errors)
        assert!(relative_error < 0.1, "Momentum not conserved: relative error = {}", relative_error);
    }
}

#[test]
fn test_energy_conservation() {
    // Total energy should be approximately conserved
    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0, 1.0),
        VerifiableEntity::new(100.0, 0.0, 0.0, 0.0, -1.0, 0.0, 100.0, 1.0),
    ];

    let initial_energy = total_energy(&entities);
    println!("Initial energy: {}", initial_energy);

    let time_step = 0.1;
    let mut tree = GravTree::new(&entities, time_step, 3, 0.1, CalculateCollisions::No);

    // Run simulation for several steps
    for step in 0..20 {
        tree = tree.time_step();
        let entities = tree.as_vec();
        let current_energy = total_energy(&entities);
        let energy_error = if initial_energy.abs() > EPSILON {
            ((current_energy - initial_energy) / initial_energy).abs()
        } else {
            (current_energy - initial_energy).abs()
        };

        println!("Step {}: energy = {}, relative error = {}", step, current_energy, energy_error);

        // Energy should be approximately conserved
        // Note: Euler integration is not energy-conserving, so we allow larger error
        assert!(energy_error < 1.0, "Energy conservation violated: relative error = {}", energy_error);
    }
}

#[test]
fn test_angular_momentum_conservation() {
    // Angular momentum should be conserved in an isolated system
    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0, 1.0),
        VerifiableEntity::new(100.0, 0.0, 0.0, 0.0, -1.0, 0.0, 100.0, 1.0),
        VerifiableEntity::new(0.0, 100.0, 0.0, 1.0, 0.0, 0.0, 100.0, 1.0),
    ];

    let initial_l = total_angular_momentum(&entities);
    println!("Initial angular momentum: {:?}", initial_l);

    let time_step = 0.1;
    let mut tree = GravTree::new(&entities, time_step, 3, 0.2, CalculateCollisions::No);

    // Calculate magnitude for relative error
    let initial_l_mag = (initial_l.0.powi(2) + initial_l.1.powi(2) + initial_l.2.powi(2)).sqrt();

    // Run simulation for several steps
    for step in 0..20 {
        tree = tree.time_step();
        let entities = tree.as_vec();
        let current_l = total_angular_momentum(&entities);

        let l_error = (
            (current_l.0 - initial_l.0).abs(),
            (current_l.1 - initial_l.1).abs(),
            (current_l.2 - initial_l.2).abs(),
        );

        let relative_error = if initial_l_mag > EPSILON {
            (l_error.0.powi(2) + l_error.1.powi(2) + l_error.2.powi(2)).sqrt() / initial_l_mag
        } else {
            (l_error.0.powi(2) + l_error.1.powi(2) + l_error.2.powi(2)).sqrt()
        };

        println!("Step {}: L = {:?}, error = {:?}, relative = {}", step, current_l, l_error, relative_error);

        // Angular momentum should be conserved (allowing for numerical errors)
        assert!(relative_error < 0.1, "Angular momentum not conserved: relative error = {}", relative_error);
    }
}

#[test]
fn test_stationary_particles() {
    // Two stationary particles should accelerate towards each other symmetrically
    let entities = vec![
        VerifiableEntity::new(-50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0),
        VerifiableEntity::new(50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0),
    ];

    let time_step = 0.1;
    let tree = GravTree::new(&entities, time_step, 3, 0.2, CalculateCollisions::No);
    let result = tree.time_step();
    let entities = result.as_vec();

    // Both should move towards the center (x=0)
    assert!(entities[0].x > -50.0, "Left particle should move right");
    assert!(entities[1].x < 50.0, "Right particle should move left");

    // By symmetry, they should have equal and opposite velocities
    let vel_sum = entities[0].vx + entities[1].vx;
    println!("Velocity sum: {} (should be ~0)", vel_sum);
    assert!(vel_sum.abs() < 0.01, "Velocities should be symmetric");

    // Y and Z components should remain zero
    assert!(entities[0].vy.abs() < EPSILON && entities[0].vz.abs() < EPSILON);
    assert!(entities[1].vy.abs() < EPSILON && entities[1].vz.abs() < EPSILON);
}

#[test]
fn test_circular_orbit_stability() {
    // Test a two-body system in approximate circular orbit
    // v = sqrt(G * M / r) for circular orbit
    let r = 100.0;
    let m1 = 1000.0;
    let m2 = 1.0;
    let v_orbit = (G * m1 / r).sqrt();

    let entities = vec![
        VerifiableEntity::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m1, 1.0),
        VerifiableEntity::new(r, 0.0, 0.0, 0.0, v_orbit, 0.0, m2, 1.0),
    ];

    let time_step = 1.0;
    let mut tree = GravTree::new(&entities, time_step, 3, 0.1, CalculateCollisions::No);

    let initial_distance = r;

    // Orbit should remain relatively stable
    for step in 0..100 {
        tree = tree.time_step();
        let entities = tree.as_vec();

        let current_distance = entities[1].distance_to(&entities[0]);
        let distance_change = (current_distance - initial_distance).abs() / initial_distance;

        if step % 10 == 0 {
            println!("Step {}: distance = {}, relative change = {}",
                     step, current_distance, distance_change);
        }

        // Distance shouldn't change drastically (allowing for numerical drift)
        assert!(distance_change < 0.3,
                "Orbit became unstable at step {}: distance changed by {}%",
                step, distance_change * 100.0);
    }
}

#[test]
fn test_theta_accuracy_tradeoff() {
    // Test that smaller theta values give more accurate results
    let entities: Vec<VerifiableEntity> = (0..10)
        .map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 10.0;
            let r = 100.0;
            VerifiableEntity::new(
                r * angle.cos(),
                r * angle.sin(),
                0.0,
                -angle.sin() * 0.5,
                angle.cos() * 0.5,
                0.0,
                10.0,
                1.0,
            )
        })
        .collect();

    let time_step = 0.1;

    // Run with direct simulation (reference)
    let mut direct_sim = DirectNBodySimulator::new(entities.clone(), time_step);
    direct_sim.time_step();
    let direct_entities = direct_sim.get_entities();

    // Test different theta values
    let theta_values = vec![0.01, 0.1, 0.5, 1.0];
    let mut errors = Vec::new();

    for &theta in &theta_values {
        let mut tree = GravTree::new(&entities, time_step, 3, theta, CalculateCollisions::No);
        tree = tree.time_step();
        let tree_entities = tree.as_vec();

        // Calculate average position error
        let avg_error: f64 = (0..entities.len())
            .map(|i| {
                let dx = tree_entities[i].x - direct_entities[i].x;
                let dy = tree_entities[i].y - direct_entities[i].y;
                let dz = tree_entities[i].z - direct_entities[i].z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum::<f64>() / entities.len() as f64;

        errors.push(avg_error);
        println!("Theta = {}: average error = {}", theta, avg_error);
    }

    // Verify that smaller theta gives smaller error
    for i in 0..errors.len() - 1 {
        assert!(
            errors[i] <= errors[i + 1] * 1.5, // Allow some numerical noise
            "Smaller theta should give more accurate results: theta[{}]={} has error {}, theta[{}]={} has error {}",
            i, theta_values[i], errors[i],
            i+1, theta_values[i+1], errors[i+1]
        );
    }
}

#[test]
fn test_many_body_against_direct() {
    // Test with more particles (but still small enough for direct simulation)
    let entities: Vec<VerifiableEntity> = (0..20)
        .map(|i| {
            VerifiableEntity::new(
                (i as f64) * 10.0,
                ((i * 7) % 20) as f64 * 10.0,
                ((i * 13) % 20) as f64 * 5.0,
                0.1 * (i as f64).sin(),
                0.1 * (i as f64).cos(),
                0.0,
                10.0,
                1.0,
            )
        })
        .collect();

    let time_step = 0.05;

    // Run direct simulation
    let mut direct_sim = DirectNBodySimulator::new(entities.clone(), time_step);
    for _ in 0..3 {
        direct_sim.time_step();
    }

    // Run Barnes-Hut simulation with small theta
    let mut tree = GravTree::new(&entities, time_step, 3, 0.1, CalculateCollisions::No);
    for _ in 0..3 {
        tree = tree.time_step();
    }

    let direct_entities = direct_sim.get_entities();
    let tree_entities = tree.as_vec();

    // Calculate average error
    let avg_error: f64 = (0..entities.len())
        .map(|i| {
            let dx = tree_entities[i].x - direct_entities[i].x;
            let dy = tree_entities[i].y - direct_entities[i].y;
            let dz = tree_entities[i].z - direct_entities[i].z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum::<f64>() / entities.len() as f64;

    println!("Average position error: {}", avg_error);

    // With 20 bodies and theta=0.1, error should be reasonable
    assert!(avg_error < 5.0, "Average error too large: {}", avg_error);
}

