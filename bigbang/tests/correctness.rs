// the situation seems to be as such:
// the simulation is totally broken on 2 entities
// collisions are not detected accurately -- perhaps having something to do with the tree
// structure?
// acceleration is zeroed out
// The issue probably arises when the total number of entities is less than max_pts?

extern crate bigbang;
use bigbang::{collisions::soft_body, AsEntity, CalculateCollisions, GravTree, Responsive, SimulationResult};

#[derive(Clone, PartialEq, AsEntity)]
struct MyEntity {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    radius: f64,
    mass: f64,
    collided_with: Vec<MyEntity>,
}

impl MyEntity {
    fn new(x: f64, y: f64, z: f64, radius: f64, mass: f64) -> MyEntity {
        MyEntity {
            x,
            y,
            z,
            vx: 0.,
            vy: 0.,
            vz: 0.,
            radius,
            mass,
            collided_with: Vec::new(),
        }
    }
}

impl Responsive for MyEntity {
    fn respond(&self, simulation_result: SimulationResult<Self>, time_step: f64) -> Self {
        let mut vx = self.vx;
        let mut vy = self.vy;
        let mut vz = self.vz;
        let mut collided_with = Vec::new();
        let (mut ax, mut ay, mut az) = simulation_result.gravitational_acceleration;
        for other in simulation_result.collisions.iter() {
            collided_with.push((*other).clone());
            let (collision_ax, collision_ay, collision_az) = soft_body(self, *other, 50f64);
            ax += collision_ax;
            ay += collision_ay;
            az += collision_az;
        }
        vx += ax * time_step;
        vy += ay * time_step;
        vz += az * time_step;

        MyEntity {
            vx,
            vy,
            vz,
            x: self.x + (vx * time_step),
            y: self.y + (vy * time_step),
            z: self.z + (vz * time_step),
            radius: self.radius,
            mass: self.mass,
            collided_with,
        }
    }
}

/// Test that, given two entities that are overlapping, the tree detects their collision.
#[test]
fn two_entities_collision() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 0., 0., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let after_time_step = test_tree.time_step().as_vec();

    // Each entity should have collided with exactly one other entity
    assert_eq!(after_time_step[0].collided_with.len(), 1);
    assert_eq!(after_time_step[1].collided_with.len(), 1);
}

/// Test that, given two entities that are not overlapping, the tree correctly does not report their collision.
#[test]
fn two_entities_no_collision() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 1000., 0., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let after_time_step = test_tree.time_step().as_vec();

    assert_eq!(after_time_step[0].collided_with.len(), 0);
    assert_eq!(after_time_step[1].collided_with.len(), 0);
}

/// Test that the gravitational acceleration of two distant particles is calculated correctly
#[test]
fn two_entities_accel() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 100., 0., 10., 50.),
        MyEntity::new(50., 0., 1., 10., 500.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.3, 3, 0.2, CalculateCollisions::Yes);
    let _after_time_step = test_tree.time_step().time_step().as_vec();

    // 1.0 isn't right but it should at least not be 0, what the current test is suggesting
    // Uncomment the following line when you're ready to fix this
    // assert_eq!(after_time_step[0].vx, 1.);
}

/// Test that, given entities that are at the _exact same position_, the tree detects their collision.
/// * NOTE:
/// This is how things _should_ be, but isn't working quite yet. This is related to the fundamental
/// structure of the tree and how it compares every particle to itself as part of the iteration.
/// This is not ideal.
/*
#[test]
fn exact_overlap_collision() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 0., 0., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
        MyEntity::new(0., 0., 1., 10., 5.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2);
    let after_time_step = test_tree.time_step().as_vec();

    // Each entity should have collided with exactly all four other entities
    assert_eq!(after_time_step[0].collided_with.len(), 4);
    assert_eq!(after_time_step[1].collided_with.len(), 4);
    assert_eq!(after_time_step[2].collided_with.len(), 4);
    assert_eq!(after_time_step[3].collided_with.len(), 4);
    assert_eq!(after_time_step[4].collided_with.len(), 4);
}
*/

/// Test that, given five entities that are overlapping, the tree detects their collision.
#[test]
fn five_entities_collision() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 0., 0., 10., 5.),
        MyEntity::new(0., 1., 0., 10., 5.),
        MyEntity::new(1., 0., 0., 10., 5.),
        MyEntity::new(1., 1., 1., 10., 5.),
        MyEntity::new(0., 1., 1., 10., 5.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.2, 3, 0.2, CalculateCollisions::Yes);
    let after_time_step = test_tree.time_step().as_vec();

    // Each entity should have collided with exactly all four other entities
    assert_eq!(after_time_step[0].collided_with.len(), 4);
    assert_eq!(after_time_step[1].collided_with.len(), 4);
    assert_eq!(after_time_step[2].collided_with.len(), 4);
    assert_eq!(after_time_step[3].collided_with.len(), 4);
    assert_eq!(after_time_step[4].collided_with.len(), 4);
}
/// Test that the gravitational acceleration of five particles is calculated correctly
/// by verifying their velocity afterwards
#[test]
fn five_entities_accel() {
    let vec_that_wants_to_be_a_kdtree: Vec<MyEntity> = vec![
        MyEntity::new(0., 100., 0., 10., 50.),
        MyEntity::new(50., 0., 1., 10., 500.),
        MyEntity::new(50., 20., 1., 10., 500.),
        MyEntity::new(10., 20., 1., 10., 500.),
        MyEntity::new(50., 100., 1., 10., 500.),
    ];

    let test_tree = GravTree::with_default_parallel_threshold(&vec_that_wants_to_be_a_kdtree, 0.3, 3, 0.2, CalculateCollisions::Yes);
    let after_time_step = test_tree.time_step().time_step().as_vec();

    const EPSILON: f64 = 1e-14;

    assert!((after_time_step[0].vx - 0.15431299859147837).abs() < EPSILON);
    assert!((after_time_step[0].vy - (-0.09585586271461218)).abs() < EPSILON);
    assert!((after_time_step[0].vz - 0.0035439741313927063).abs() < EPSILON);

    assert!((after_time_step[1].vx - (-0.13582446094615622)).abs() < EPSILON);
    assert!((after_time_step[1].vy - 0.8512257874024887).abs() < EPSILON);
    assert!((after_time_step[1].vz - (-0.000021554920561577058)).abs() < EPSILON);

    assert!((after_time_step[2].vx - (-0.18949313172952317)).abs() < EPSILON);
    assert!((after_time_step[2].vy - (-0.7019795056879561)).abs() < EPSILON);
    assert!((after_time_step[2].vz - (-0.000035662286878477935)).abs() < EPSILON);

    assert!((after_time_step[3].vx - 0.3386640864203134).abs() < EPSILON);
    assert!((after_time_step[3].vy - (-0.02923599470394115)).abs() < EPSILON);
    assert!((after_time_step[3].vz - (-0.00005721416764760718)).abs() < EPSILON);

    assert!((after_time_step[4].vx - (-0.028777793603781895)).abs() < EPSILON);
    assert!((after_time_step[4].vy - (-0.11042470073913033)).abs() < EPSILON);
    assert!((after_time_step[4].vz - (-0.00023996603805160843)).abs() < EPSILON);
}

/// Test that an entity in a predefined circular orbit maintains its orbit after 100000 iterations
#[test]
fn circular_orbit_stability() {
    use bigbang::{Entity, GravTree, CalculateCollisions};

    // Central body (e.g., Sun or Earth)
    let central_mass = 1000.0;
    let central_radius = 10.0;

    // Orbiting body
    let orbit_radius = 500.0; // Distance from central body
    let orbiting_mass = 1.0; // Small mass to not perturb the system
    let orbiting_radius = 1.0;

    // For a circular orbit: v = sqrt(G * M / r)
    // Assuming G = 1.0 in normalized units
    let g = 1.0;
    let orbital_velocity = ((g * central_mass / orbit_radius) as f64).sqrt();

    // Central body at origin (stationary)
    let central = Entity::new(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        central_radius,
        central_mass,
    );

    // Orbiting body starts at (orbit_radius, 0, 0) with velocity in +y direction
    let orbiting = Entity::new(
        0.0,
        orbital_velocity,
        0.0,
        orbit_radius,
        0.0,
        0.0,
        orbiting_radius,
        orbiting_mass,
    );

    let entities = vec![central, orbiting];

    // Create simulation with appropriate parameters
    // theta = 0.5 for Barnes-Hut approximation (realistic usage)
    // Smaller time step for accuracy with Velocity Verlet integration
    let time_step = 0.1;
    let theta = 0.5;
    let mut tree = GravTree::with_default_parallel_threshold(&entities, time_step, 3, theta, CalculateCollisions::No);

    // Track orbital parameters over time
    let mut orbital_radii = Vec::new();
    let mut velocities = Vec::new();

    // Run simulation for 100000 iterations using Velocity Verlet for better long-term stability
    for _ in 0..100000 {
        tree.time_step_mut_verlet();
        let entities = tree.as_vec();
        let central = &entities[0];
        let orbiter = &entities[1];

        // Calculate distance from central body (relative distance, not from origin)
        let dx = orbiter.x - central.x;
        let dy = orbiter.y - central.y;
        let dz = orbiter.z - central.z;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        orbital_radii.push(r);

        // Calculate relative velocity magnitude
        let dvx = orbiter.vx - central.vx;
        let dvy = orbiter.vy - central.vy;
        let dvz = orbiter.vz - central.vz;
        let v = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
        velocities.push(v);
    }

    // Calculate statistics for orbital radius
    let mean_radius: f64 = orbital_radii.iter().sum::<f64>() / orbital_radii.len() as f64;
    let max_radius = orbital_radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_radius = orbital_radii.iter().cloned().fold(f64::INFINITY, f64::min);

    // Calculate statistics for velocity
    let mean_velocity: f64 = velocities.iter().sum::<f64>() / velocities.len() as f64;
    let max_velocity = velocities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_velocity = velocities.iter().cloned().fold(f64::INFINITY, f64::min);

    // Assert orbital stability
    // The orbit should remain within 1% of the initial radius
    // (tolerance increased to account for Barnes-Hut approximation errors)
    let radius_tolerance = orbit_radius * 0.01;
    assert!(
        (mean_radius - orbit_radius).abs() < radius_tolerance,
        "Mean orbital radius {} deviates too much from initial radius {}",
        mean_radius,
        orbit_radius
    );
    assert!(
        max_radius < orbit_radius + radius_tolerance,
        "Maximum orbital radius {} exceeds tolerance",
        max_radius
    );
    assert!(
        min_radius > orbit_radius - radius_tolerance,
        "Minimum orbital radius {} below tolerance",
        min_radius
    );

    // Velocity should also remain stable within 1%
    let velocity_tolerance = orbital_velocity * 0.01;
    assert!(
        (mean_velocity - orbital_velocity).abs() < velocity_tolerance,
        "Mean velocity {} deviates too much from expected orbital velocity {}",
        mean_velocity,
        orbital_velocity
    );
    assert!(
        max_velocity < orbital_velocity + velocity_tolerance,
        "Maximum velocity {} exceeds tolerance",
        max_velocity
    );
    assert!(
        min_velocity > orbital_velocity - velocity_tolerance,
        "Minimum velocity {} below tolerance",
        min_velocity
    );
}

