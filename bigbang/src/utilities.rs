use super::Dimension;
use crate::entity::Entity;

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