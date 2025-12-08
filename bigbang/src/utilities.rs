use super::Dimension;
#[allow(unused_imports)]
use crate::entity::Entity;
use std::cmp::Ordering;

/// Returns the absolute distance in every dimension (the range in every dimension)
/// using indices into an entity array (optimized to avoid cloning).
pub(crate) fn xyz_distances_indexed(entities: &[Entity], indices: &[usize]) -> (f64, f64, f64) {
    if indices.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let mut x_min = entities[indices[0]].x;
    let mut x_max = x_min;
    let mut y_min = entities[indices[0]].y;
    let mut y_max = y_min;
    let mut z_min = entities[indices[0]].z;
    let mut z_max = z_min;
    
    for &idx in &indices[1..] {
        let e = &entities[idx];
        x_min = x_min.min(e.x);
        x_max = x_max.max(e.x);
        y_min = y_min.min(e.y);
        y_max = y_max.max(e.y);
        z_min = z_min.min(e.z);
        z_max = z_max.max(e.z);
    }
    
    ((x_max - x_min).abs(), (y_max - y_min).abs(), (z_max - z_min).abs())
}

/// Returns the absolute distance in every dimension (the range in every dimension)
/// of an array slice of entities.
pub(crate) fn xyz_distances(entities: &[Entity]) -> (f64, f64, f64) {
    let (x_max, x_min, y_max, y_min, z_max, z_min) = max_min_xyz(entities);
    let x_distance = x_max - x_min;
    let y_distance = y_max - y_min;
    let z_distance = z_max - z_min;
    (x_distance.abs(), y_distance.abs(), z_distance.abs())
}

/// Returns max/min values using indices (optimized to avoid cloning).
#[inline]
pub(crate) fn max_min_xyz_indexed(entities: &[Entity], indices: &[usize]) -> (f64, f64, f64, f64, f64, f64) {
    if indices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    let first = &entities[indices[0]];
    let mut x_min = first.x;
    let mut x_max = first.x;
    let mut y_min = first.y;
    let mut y_max = first.y;
    let mut z_min = first.z;
    let mut z_max = first.z;
    
    for &idx in &indices[1..] {
        let e = &entities[idx];
        x_min = x_min.min(e.x);
        x_max = x_max.max(e.x);
        y_min = y_min.min(e.y);
        y_max = y_max.max(e.y);
        z_min = z_min.min(e.z);
        z_max = z_max.max(e.z);
    }
    
    (x_max, x_min, y_max, y_min, z_max, z_min)
}

/// Given an array slice of entities, returns the maximum and minimum x, y, and z values as
/// a septuple.
pub(crate) fn max_min_xyz(entities: &[Entity]) -> (&f64, &f64, &f64, &f64, &f64, &f64) {
    let (x_max, x_min) = max_min(Dimension::X, entities);
    let (y_max, y_min) = max_min(Dimension::Y, entities);
    let (z_max, z_min) = max_min(Dimension::Z, entities);
    (x_max, x_min, y_max, y_min, z_max, z_min)
}

/// Returns the maximum and minimum values in a slice of entities, given a dimension.
pub(crate) fn max_min(dim: Dimension, entities: &[Entity]) -> (&f64, &f64) {
    (
        entities
            .iter()
            .max_by(|a, b| {
                a.get_dim(&dim)
                    .partial_cmp(b.get_dim(&dim))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or_else(|| panic!("no max {} found", dim.as_string()))
            .get_dim(&dim),
        entities
            .iter()
            .min_by(|a, b| {
                a.get_dim(&dim)
                    .partial_cmp(b.get_dim(&dim))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or_else(|| panic!("no min {} found", dim.as_string()))
            .get_dim(&dim),
    )
}

/// Finds the median value for a given dimension in a slice of entities.
/// Making one that clones/uses immutability could be an interesting performance benchmark.
pub(crate) fn find_median(dim: Dimension, pts: &mut [Entity]) -> (&f64, usize) {
    find_median_helper(dim, pts, 0, pts.len(), pts.len() / 2usize)
}

/// Partitions indices based on median of dimension without mutating entities.
/// Returns (median_value, split_index) where indices[..split_index] < median.
/// The indices slice is reordered to match the partitioning.
pub(crate) fn partition_indices_by_median(
    dim: Dimension,
    entities: &[Entity],
    indices: &mut [usize],
) -> (f64, usize) {
    if indices.is_empty() {
        return (0.0, 0);
    }

    let mid = indices.len() / 2;
    partition_indices_helper(dim, entities, indices, 0, indices.len(), mid)
}

fn partition_indices_helper(
    dim: Dimension,
    entities: &[Entity],
    indices: &mut [usize],
    start: usize,
    end: usize,
    mid: usize,
) -> (f64, usize) {
    // Base case: empty range
    if start >= end {
        let idx = if start < indices.len() { start } else { indices.len() - 1 };
        return (*entities[indices[idx]].get_dim(&dim), idx);
    }

    let mut low = start + 1;
    let mut high = if end > 0 { end - 1 } else { 0 }; // exclusive end, so end-1 is the last valid index

    // Partition: elements < pivot go left, >= pivot go right
    while low <= high {
        // Critical: check bounds before dereferencing
        if low >= end {
            break;
        }

        if entities[indices[low]].get_dim(&dim) < entities[indices[start]].get_dim(&dim) {
            low += 1;
        } else {
            indices.swap(low, high);
            if high == 0 {
                break;
            }
            high -= 1;
        }
    }

    // Place pivot in correct position
    if high < indices.len() {
        indices.swap(start, high);
    }

    if high == mid {
        (*entities[indices[high]].get_dim(&dim), high)
    } else if high < mid {
        partition_indices_helper(dim, entities, indices, high + 1, end, mid)
    } else {
        partition_indices_helper(dim, entities, indices, start, high, mid)
    }
}

fn find_median_helper(
    dim: Dimension,
    pts: &mut [Entity],
    start: usize,
    end: usize,
    mid: usize,
) -> (&f64, usize) {
    let mut low = start + 1;
    let mut high = end - 1; //exclusive end
    while low <= high {
        if pts[low].get_dim(&dim) < pts[start].get_dim(&dim) {
            low += 1;
        } else {
            pts.swap(low, high);
            high -= 1;
        }
    }
    pts.swap(start, high);
    if start == mid {
        (pts[start].get_dim(&dim), start)
    } else if high < mid {
        find_median_helper(dim, pts, high + 1, end, mid)
    } else {
        find_median_helper(dim, pts, start, high, mid)
    }
}
