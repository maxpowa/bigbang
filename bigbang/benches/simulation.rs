// End-to-end simulation benchmark: measures complete simulation flow over many time steps
// This benchmark tests realistic usage patterns with 1000 simulation steps

use bigbang::{CalculateCollisions, Entity, GravTree};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Creates a set of entities in a grid pattern with some initial velocities
fn create_simulation_entities(count: usize) -> Vec<Entity> {
    let mut entities = Vec::with_capacity(count);
    let grid_size = (count as f64).cbrt().ceil() as usize;
    let spacing = 50.0;

    for i in 0..count {
        let gx = (i % grid_size) as f64;
        let gy = ((i / grid_size) % grid_size) as f64;
        let gz = (i / (grid_size * grid_size)) as f64;

        // Position on grid
        let x = gx * spacing;
        let y = gy * spacing;
        let z = gz * spacing;

        // Small random-ish velocities based on position
        let vx = (gx - grid_size as f64 / 2.0) * 0.1;
        let vy = (gy - grid_size as f64 / 2.0) * 0.1;
        let vz = (gz - grid_size as f64 / 2.0) * 0.1;

        entities.push(Entity::new(vx, vy, vz, x, y, z, 5.0, 100.0));
    }

    entities
}

/// Creates a GravTree for simulation
fn create_simulation_tree(entities: &[Entity], calculate_collisions: CalculateCollisions) -> GravTree<Entity> {
    let time_step = 0.1;
    let max_entities = 3;
    let theta = 0.5;
    GravTree::with_default_parallel_threshold(
        &mut entities.to_vec(),
        time_step,
        max_entities,
        theta,
        calculate_collisions,
    )
}

/// Runs the simulation for a specified number of steps
fn run_simulation_steps(mut tree: GravTree<Entity>, steps: usize) -> GravTree<Entity> {
    for _ in 0..steps {
        tree = tree.time_step();
    }
    tree
}

// Benchmark: End-to-end simulation with 1000 steps
fn bench_simulation_1000_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_simulation");

    // Increase measurement time for long-running benchmarks
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    // Small simulation: 64 entities, 1000 steps, no collisions
    group.bench_function("1000_steps/n=64/no_collisions", |b| {
        let entities = create_simulation_entities(64);
        b.iter(|| {
            let tree = create_simulation_tree(&entities, CalculateCollisions::No);
            run_simulation_steps(tree, 1000)
        })
    });

    // Small simulation: 64 entities, 1000 steps, with collisions
    group.bench_function("1000_steps/n=64/with_collisions", |b| {
        let entities = create_simulation_entities(64);
        b.iter(|| {
            let tree = create_simulation_tree(&entities, CalculateCollisions::Yes);
            run_simulation_steps(tree, 1000)
        })
    });

    // Medium simulation: 125 entities, 1000 steps, no collisions
    group.bench_function("1000_steps/n=125/no_collisions", |b| {
        let entities = create_simulation_entities(125);
        b.iter(|| {
            let tree = create_simulation_tree(&entities, CalculateCollisions::No);
            run_simulation_steps(tree, 1000)
        })
    });

    // Medium simulation: 125 entities, 1000 steps, with collisions
    group.bench_function("1000_steps/n=125/with_collisions", |b| {
        let entities = create_simulation_entities(125);
        b.iter(|| {
            let tree = create_simulation_tree(&entities, CalculateCollisions::Yes);
            run_simulation_steps(tree, 1000)
        })
    });

    // Larger simulation: 250 entities, 1000 steps, no collisions
    group.bench_function("1000_steps/n=250/no_collisions", |b| {
        let entities = create_simulation_entities(250);
        b.iter(|| {
            let tree = create_simulation_tree(&entities, CalculateCollisions::No);
            run_simulation_steps(tree, 1000)
        })
    });

    group.finish();
}

// Benchmark: Varying step counts to analyze performance scaling
fn bench_simulation_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_step_scaling");

    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    let entities = create_simulation_entities(100);

    for steps in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("n=100", steps), steps, |b, &steps| {
            b.iter(|| {
                let tree = create_simulation_tree(&entities, CalculateCollisions::No);
                run_simulation_steps(tree, steps)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_simulation_1000_steps, bench_simulation_step_scaling);
criterion_main!(benches);

