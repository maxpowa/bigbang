// Hot path benchmarks for critical performance paths in BigBang
// These benchmarks focus on the most frequently called functions during simulation

use std::hint::black_box;
use bigbang::{collisions::soft_body, AsEntity, CalculateCollisions, Entity, GravTree, Responsive, SimulationResult};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::time;

// Helper function to create a simple entity
fn create_entity(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64, radius: f64) -> Entity {
    Entity {
        x,
        y,
        z,
        vx,
        vy,
        vz,
        mass,
        radius,
    }
}

// Helper function to create test data for various scenarios
fn create_test_entities(count: usize, spacing: f64) -> Vec<Entity> {
    let mut entities = Vec::with_capacity(count);
    let grid_size = (count as f64).cbrt().ceil() as usize;

    for i in 0..count {
        let x = ((i % grid_size) as f64) * spacing;
        let y = (((i / grid_size) % grid_size) as f64) * spacing;
        let z = ((i / (grid_size * grid_size)) as f64) * spacing;

        entities.push(create_entity(
            x, y, z,
            0.1, 0.1, 0.1,
            100.0,
            1.0,
        ));
    }

    entities
}

// Benchmark 1: Distance calculation - Called extensively in collision detection
fn bench_distance_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/distance_calculation");

    let e1 = create_entity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0);
    let e2 = create_entity(10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 100.0, 1.0);

    group.bench_function("close_entities", |b| {
        b.iter(|| {
            let dx = e2.x - e1.x;
            let dy = e2.y - e1.y;
            let dz = e2.z - e1.z;
            black_box(dx * dx + dy * dy + dz * dz)
        })
    });

    let e3 = create_entity(1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 100.0, 1.0);
    group.bench_function("far_entities", |b| {
        b.iter(|| {
            let dx = e3.x - e1.x;
            let dy = e3.y - e1.y;
            let dz = e3.z - e1.z;
            black_box(dx * dx + dy * dy + dz * dz)
        })
    });

    group.finish();
}

// Benchmark 2: Collision detection with did_collide_into()
fn bench_collision_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/collision_detection");

    // Test case 1: Colliding entities
    let e1 = create_entity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 5.0);
    let e2 = create_entity(8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 5.0);

    group.bench_function("colliding", |b| {
        b.iter(|| {
            let dx = e2.x - e1.x;
            let dy = e2.y - e1.y;
            let dz = e2.z - e1.z;
            let dist_squared = dx * dx + dy * dy + dz * dz;
            let radii_sum = e1.radius + e2.radius;
            black_box(dist_squared <= radii_sum * radii_sum)
        })
    });

    // Test case 2: Non-colliding entities
    let e3 = create_entity(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 5.0);
    group.bench_function("non_colliding", |b| {
        b.iter(|| {
            let dx = e3.x - e1.x;
            let dy = e3.y - e1.y;
            let dz = e3.z - e1.z;
            let dist_squared = dx * dx + dy * dy + dz * dz;
            let radii_sum = e1.radius + e3.radius;
            black_box(dist_squared <= radii_sum * radii_sum)
        })
    });

    // Test case 3: Batch collision detection
    let entities = create_test_entities(100, 10.0);
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            let e1 = &entities[0];
            let mut collision_count = 0;
            for e2 in &entities[1..] {
                let dx = e2.x - e1.x;
                let dy = e2.y - e1.y;
                let dz = e2.z - e1.z;
                let dist_squared = dx * dx + dy * dy + dz * dz;
                let radii_sum = e1.radius + e2.radius;
                if dist_squared <= radii_sum * radii_sum {
                    collision_count += 1;
                }
            }
            black_box(collision_count)
        })
    });

    group.finish();
}

// Benchmark 3: soft_body collision response
fn bench_soft_body_collision(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/soft_body");

    // Colliding entities
    let e1 = create_entity(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 100.0, 5.0);
    let e2 = create_entity(8.0, 0.0, 0.0, -1.0, 0.0, 0.0, 100.0, 5.0);

    group.bench_function("collision_response", |b| {
        b.iter(|| {
            black_box(soft_body(black_box(&e1), black_box(&e2), 20.0))
        })
    });

    // Non-colliding entities (early exit path)
    let e3 = create_entity(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 5.0);
    group.bench_function("no_collision_early_exit", |b| {
        b.iter(|| {
            black_box(soft_body(black_box(&e1), black_box(&e3), 20.0))
        })
    });

    group.finish();
}

// Benchmark 4: Entity response (velocity Verlet integration + collision response)
fn bench_entity_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/entity_response");

    let entity = create_entity(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 100.0, 5.0);

    // No collisions
    group.bench_function("no_collisions", |b| {
        b.iter(|| {
            let sim_result = SimulationResult {
                gravitational_acceleration: (0.1, 0.2, 0.3),
                collisions: vec![],
            };
            black_box(entity.respond(sim_result, 0.1))
        })
    });

    // Single collision
    let collision_entity = create_entity(8.0, 0.0, 0.0, -5.0, 0.0, 0.0, 100.0, 5.0);
    group.bench_function("single_collision", |b| {
        b.iter(|| {
            let sim_result = SimulationResult {
                gravitational_acceleration: (0.1, 0.2, 0.3),
                collisions: vec![&collision_entity],
            };
            black_box(entity.respond(sim_result, 0.1))
        })
    });

    // Multiple collisions
    let collision_entities: Vec<Entity> = (0..5)
        .map(|i| create_entity(
            (i as f64) * 8.0,
            0.0,
            0.0,
            -(i as f64),
            0.0,
            0.0,
            100.0,
            5.0
        ))
        .collect();

    group.bench_function("multiple_collisions", |b| {
        b.iter(|| {
            let collision_refs: Vec<&Entity> = collision_entities.iter().collect();
            let sim_result = SimulationResult {
                gravitational_acceleration: (0.1, 0.2, 0.3),
                collisions: collision_refs,
            };
            black_box(entity.respond(sim_result, 0.1))
        })
    });

    group.finish();
}

// Benchmark 5: Gravitational acceleration calculation (entity-to-entity)
fn bench_gravitational_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/gravitational_acceleration");

    let e1 = create_entity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0);
    let e2 = create_entity(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0);

    group.bench_function("close_pair", |b| {
        b.iter(|| {
            // Inline the gravitational acceleration calculation using public fields
            let dx = e2.x - e1.x;
            let dy = e2.y - e1.y;
            let dz = e2.z - e1.z;
            let dist_squared = dx * dx + dy * dy + dz * dz;
            let d_magnitude = dist_squared.sqrt();
            let d_mag_cubed = d_magnitude * d_magnitude * d_magnitude;
            let accel = (
                dx / d_mag_cubed * e2.mass,
                dy / d_mag_cubed * e2.mass,
                dz / d_mag_cubed * e2.mass,
            );
            black_box(accel)
        })
    });

    let e3 = create_entity(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000.0, 100.0);
    group.bench_function("far_massive_pair", |b| {
        b.iter(|| {
            let dx = e3.x - e1.x;
            let dy = e3.y - e1.y;
            let dz = e3.z - e1.z;
            let dist_squared = dx * dx + dy * dy + dz * dz;
            let d_magnitude = dist_squared.sqrt();
            let d_mag_cubed = d_magnitude * d_magnitude * d_magnitude;
            let accel = (
                dx / d_mag_cubed * e3.mass,
                dy / d_mag_cubed * e3.mass,
                dz / d_mag_cubed * e3.mass,
            );
            black_box(accel)
        })
    });

    // Batch acceleration calculation
    let entities = create_test_entities(50, 20.0);
    group.bench_function("batch_50_entities", |b| {
        b.iter(|| {
            let e1 = &entities[0];
            let mut total_accel = (0.0, 0.0, 0.0);
            for e2 in &entities[1..] {
                let dx = e2.x - e1.x;
                let dy = e2.y - e1.y;
                let dz = e2.z - e1.z;
                let dist_squared = dx * dx + dy * dy + dz * dz;
                let d_magnitude = dist_squared.sqrt();
                if d_magnitude == 0.0 {
                    continue;
                }
                let d_mag_cubed = d_magnitude * d_magnitude * d_magnitude;
                total_accel.0 += dx / d_mag_cubed * e2.mass;
                total_accel.1 += dy / d_mag_cubed * e2.mass;
                total_accel.2 += dz / d_mag_cubed * e2.mass;
            }
            black_box(total_accel)
        })
    });

    group.finish();
}

// Benchmark 6: Full simulation step with varying particle counts
fn bench_full_time_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/full_time_step");

    // Small simulation
    group.bench_function("n=50_with_collisions", |b| {
        b.iter_batched(
            || {
                let mut data = create_test_entities(50, 15.0);
                GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::Yes)
            },
            |tree| black_box(tree.time_step()),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("n=50_no_collisions", |b| {
        b.iter_batched(
            || {
                let mut data = create_test_entities(50, 15.0);
                GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::No)
            },
            |tree| black_box(tree.time_step()),
            BatchSize::SmallInput,
        )
    });

    // Medium simulation
    group.measurement_time(time::Duration::new(15, 0));
    group.bench_function("n=500_with_collisions", |b| {
        b.iter_batched(
            || {
                let mut data = create_test_entities(500, 15.0);
                GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::Yes)
            },
            |tree| black_box(tree.time_step()),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("n=500_no_collisions", |b| {
        b.iter_batched(
            || {
                let mut data = create_test_entities(500, 15.0);
                GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::No)
            },
            |tree| black_box(tree.time_step()),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// Benchmark 7: Tree construction (hot path for initialization)
fn bench_tree_construction_detailed(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/tree_construction");

    group.bench_function("n=50", |b| {
        b.iter_batched(
            || create_test_entities(50, 10.0),
            |mut data| GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::No),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("n=500", |b| {
        b.iter_batched(
            || create_test_entities(500, 10.0),
            |mut data| GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::No),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("n=1000", |b| {
        b.iter_batched(
            || create_test_entities(1000, 10.0),
            |mut data| GravTree::with_default_parallel_threshold(&mut data, 0.1, 3, 0.5, CalculateCollisions::No),
            BatchSize::SmallInput,
        )
    });

    // Test with different max_entities settings
    group.bench_function("n=500_max_entities=1", |b| {
        b.iter_batched(
            || create_test_entities(500, 10.0),
            |mut data| GravTree::with_default_parallel_threshold(&mut data, 0.1, 1, 0.5, CalculateCollisions::No),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("n=500_max_entities=10", |b| {
        b.iter_batched(
            || create_test_entities(500, 10.0),
            |mut data| GravTree::with_default_parallel_threshold(&mut data, 0.1, 10, 0.5, CalculateCollisions::No),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// Benchmark 8: as_entity() conversions (frequently called in loops)
fn bench_as_entity_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path/as_entity_conversion");

    let entity = create_entity(10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 100.0, 5.0);

    group.bench_function("single_conversion", |b| {
        b.iter(|| {
            black_box(entity.as_entity())
        })
    });

    let entities = create_test_entities(1000, 10.0);
    group.bench_function("batch_1000_conversions", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for entity in &entities {
                let e = entity.as_entity();
                sum += e.x + e.y + e.z;
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    hot_path_benches,
    bench_distance_calculation,
    bench_collision_detection,
    bench_soft_body_collision,
    bench_entity_response,
    bench_gravitational_acceleration,
    bench_full_time_step,
    bench_tree_construction_detailed,
    bench_as_entity_conversion,
);

criterion_main!(hot_path_benches);

