use either::{Either, Left, Right};

use super::{Dimension, Responsive};
use crate::as_entity::AsEntity;
use crate::simulation_result::SimulationResult;
use crate::Node;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum CalculateCollisions {
    Yes,
    No,
}

/// An Entity is an object (generalized to be spherical, having only a radius dimension) which has
/// velocity, position, radius, and mass. This gravitational tree contains many entities and it moves
/// them around according to the gravity they exert on each other.
#[cfg_attr(feature = "bevy_ecs", derive(::bevy_ecs::prelude::Component))]
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Entity {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub radius: f64,
    pub mass: f64,
    _private: ()
}

impl AsEntity for Entity {
    fn as_entity(&self) -> Entity {
        // Entity is a plain-old-data struct with Copy semantics
        // This is not a deep clone - just copying 8 f64 values (64 bytes)
        *self
    }
}

impl PartialEq for Entity {
    /// This is a workaround to prevent every particle from reporting that it is colliding with
    /// itself. If two particles truly become identical, they won't be reported as colliding. This
    /// is a spot for future improvement. The cost of adding some sort of unique ID is too much for
    /// the time being.
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.z == other.z
            && self.radius == other.radius
            && self.mass == other.mass
    }
}

impl Responsive for Entity {
    fn respond(&self, simulation_result: SimulationResult<Entity>, time_step: f64) -> Self {
        let mut new_entity = self.clone();
        new_entity.respond_mut(simulation_result, time_step);
        new_entity
    }

    fn respond_mut(&mut self, simulation_result: SimulationResult<Entity>, time_step: f64) {
        let (ax, ay, az) = simulation_result.gravitational_acceleration;
        let self_mass = self.mass;

        // Handle collisions with proper 3D elastic collision physics
        for other in simulation_result.collisions.iter() {
            let other_mass = other.mass;

            // Calculate collision normal vector (from self to other)
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

            // Project velocities onto collision normal
            let v1n = self.vx * nx + self.vy * ny + self.vz * nz;
            let v2n = other.vx * nx + other.vy * ny + other.vz * nz;

            // Apply 1D elastic collision formula to normal component
            let mass_sum = self_mass + other_mass;
            let v1n_new = ((self_mass - other_mass) * v1n + 2.0 * other_mass * v2n) / mass_sum;

            let delta_vn = v1n_new - v1n;

            // Update velocity in place
            self.vx += delta_vn * nx;
            self.vy += delta_vn * ny;
            self.vz += delta_vn * nz;
        }

        // Single-step Symplectic Euler integration
        // Note: This is less accurate than Velocity Verlet. Use respond_mut_verlet for better accuracy.
        self.vx += ax * time_step;
        self.vy += ay * time_step;
        self.vz += az * time_step;

        self.x += self.vx * time_step;
        self.y += self.vy * time_step;
        self.z += self.vz * time_step;
    }

    fn respond_mut_verlet(
        &mut self,
        simulation_result_current: SimulationResult<Entity>,
        simulation_result_next: SimulationResult<Entity>,
        time_step: f64,
    ) {
        let (ax_current, ay_current, az_current) = simulation_result_current.gravitational_acceleration;
        let (ax_next, ay_next, az_next) = simulation_result_next.gravitational_acceleration;
        let self_mass = self.mass;

        // Handle collisions with proper 3D elastic collision physics
        // Apply collisions from current timestep before integration
        for other in simulation_result_current.collisions.iter() {
            let other_mass = other.mass;

            // Calculate collision normal vector (from self to other)
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

            // Project velocities onto collision normal
            let v1n = self.vx * nx + self.vy * ny + self.vz * nz;
            let v2n = other.vx * nx + other.vy * ny + other.vz * nz;

            // Apply 1D elastic collision formula to normal component
            let mass_sum = self_mass + other_mass;
            let v1n_new = ((self_mass - other_mass) * v1n + 2.0 * other_mass * v2n) / mass_sum;

            let delta_vn = v1n_new - v1n;

            // Update velocity in place
            self.vx += delta_vn * nx;
            self.vy += delta_vn * ny;
            self.vz += delta_vn * nz;
        }

        // Proper Velocity Verlet integration:
        // v(t + dt/2) = v(t) + a(t) * dt/2
        // x(t + dt) = x(t) + v(t + dt/2) * dt
        // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        //
        // Equivalent to:
        // x(t + dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
        // v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt

        let dt = time_step;

        // Update position using current velocity and acceleration
        self.x += self.vx * dt + 0.5 * ax_current * dt * dt;
        self.y += self.vy * dt + 0.5 * ay_current * dt * dt;
        self.z += self.vz * dt + 0.5 * az_current * dt * dt;

        // Update velocity using average of current and next acceleration
        self.vx += 0.5 * (ax_current + ax_next) * dt;
        self.vy += 0.5 * (ay_current + ay_next) * dt;
        self.vz += 0.5 * (az_current + az_next) * dt;
    }
}

impl Entity {

    pub fn new(
        vx: f64,
        vy: f64,
        vz: f64,
        x: f64,
        y: f64,
        z: f64,
        radius: f64,
        mass: f64,
    ) -> Self {
        Entity {
            vx,
            vy,
            vz,
            x,
            y,
            z,
            radius,
            mass,
            _private: (),
        }
    }

    pub fn from_pos_radius_mass(
        x: f64,
        y: f64,
        z: f64,
        radius: f64,
        mass: f64,
    ) -> Self {
        Entity {
            vx: 0.,
            vy: 0.,
            vz: 0.,
            x,
            y,
            z,
            radius,
            mass,
            _private: (),
        }
    }

    /// Creates a new entity from position and radius, with zero velocity.
    /// Mass is calculated as radius * 1000.
    pub fn from_pos_radius(x: f64, y: f64, z: f64, radius: f64) -> Self {
        Entity {
            vx: 0.,
            vy: 0.,
            vz: 0.,
            x,
            y,
            z,
            radius,
            mass: radius * 1000.,
            _private: (),
        }
    }

    /// Needs to be reworked to use min/max position values, but it naively checks
    /// if two things collide right now.
    fn did_collide_into(&self, other: &Entity) -> bool {
        // OPTIMIZATION: Use distance_squared to avoid expensive sqrt
        if self == other {
            return false;
        }
        let radii_sum = self.radius + other.radius;
        self.distance_squared(other) <= radii_sum * radii_sum
    }

    /// Returns the entity as a string with space separated values.
    pub fn as_string(&self) -> String {
        format!(
            "{} {} {} {} {} {} {} {}",
            self.x, self.y, self.z, self.vx, self.vy, self.vz, self.mass, self.radius
        )
    }

    /// The returns the distance squared between two particles.
    /// Take the sqrt of this to get the distance.
    #[inline]
    pub(crate) fn distance_squared(&self, other: &Entity) -> f64 {
        // (x2 - x1) + (y2 - y1) + (z2 - z1)
        // all dist variables  are squared
        // This is being called from somewhere where `other` has NaN values
        let (x_dist, y_dist, z_dist) = self.distance_vector(other);
        x_dist * x_dist + y_dist * y_dist + z_dist * z_dist
    }

    /// Returns the distance between the two entities
    pub(crate) fn distance(&self, other: &Entity) -> f64 {
        // sqrt((x2 - x1) + (y2 - y1) + (z2 - z1))
        f64::sqrt(self.distance_squared(other))
    }

    /// Returns the distance between two entities as an (x:f64,y:f64,z:f64) tuple.
    pub(crate) fn distance_vector(&self, other: &Entity) -> (f64, f64, f64) {
        let x_dist = other.x - self.x;
        let y_dist = other.y - self.y;
        let z_dist = other.z - self.z;
        (x_dist, y_dist, z_dist)
    }

    pub(crate) fn get_dim(&self, dim: &Dimension) -> &f64 {
        match *dim {
            Dimension::X => &self.x,
            Dimension::Y => &self.y,
            Dimension::Z => &self.z,
        }
    }

    /// Returns a boolean representing whether or node the node is within the theta range
    /// of the entity.
    #[inline]
    fn theta_exceeded(&self, node: &Node, theta: f64) -> bool {
        // OPTIMIZATION: Calculate distance squared directly without creating temporary Entity
        let dx = node.center_of_mass.0 - self.x;
        let dy = node.center_of_mass.1 - self.y;
        let dz = node.center_of_mass.2 - self.z;
        let dist_squared = dx * dx + dy * dy + dz * dz;
        let max_dist = node.max_distance();
        dist_squared * (theta * theta) > (max_dist * max_dist)
    }

    /// Given two entities, self and other, returns the acceleration that other is exerting on
    /// self. Other can be either an entity or a node.
    #[inline]
    fn get_gravitational_acceleration<T: AsEntity + Clone>(
        &self,
        oth: Either<&Entity, &Node>,
    ) -> (f64, f64, f64) {
        // OPTIMIZATION: Use Copy instead of Clone for Entity (8 f64s = 64 bytes on stack)
        let other = match oth {
            Left(entity) => *entity,
            Right(node) => node.as_entity(),
        };
        let d_magnitude = self.distance(&other);
        if d_magnitude == 0. {
            // sort of other use of THETA here
            return (0., 0., 0.);
        }
        let d_vector = self.distance_vector(&other);
        let d_mag_cubed = d_magnitude * d_magnitude * d_magnitude;
        let d_over_d_cubed = (
            d_vector.0 / d_mag_cubed,
            d_vector.1 / d_mag_cubed,
            d_vector.2 / d_mag_cubed,
        );
        (
            d_over_d_cubed.0 * other.mass,
            d_over_d_cubed.1 * other.mass,
            d_over_d_cubed.2 * other.mass,
        )
    }

    /// Returns the acceleration of an entity after it has had gravity from the specified node applied to it.
    /// In this function, we approximate some entities if they exceed a certain critera specified in
    /// "exceeds_theta()". If we reach a node and it is a leaf, then we automatically get the
    /// acceleration from every entity in that node, but if we reach a node that is not a leaf and
    /// exceeds_theta() is true, then we treat the node as one giant entity and get the
    /// acceleration from it.
    pub(crate) fn get_acceleration_and_collisions<'a, T: AsEntity + Clone>(
        &'a self,
        node: &'a Node,
        arena: &'a [T],
        theta: f64,
    ) -> SimulationResult<'a, T> {
        // OPTIMIZATION: Pre-allocate with small capacity to avoid reallocations in common case
        let mut collisions = Vec::with_capacity(4);
        let mut acceleration = (0., 0., 0.);
        if let Some(node) = &node.left {
            if node.point_indices.is_some() {
                // if this node has some point_indices, calculate their gravitational acceleration
                let indices = node.point_indices.as_ref().expect("unexpected null node 2");
                for &idx in indices {
                    let other = &arena[idx];
                    // OPTIMIZATION: Cache as_entity() to avoid repeated conversions
                    let other_entity = other.as_entity();

                    // Check collision first
                    if self.did_collide_into(&other_entity) {
                        collisions.push(other);
                    }

                    // Calculate acceleration using cached entity
                    let tmp_accel = self.get_gravitational_acceleration::<Entity>(Left(&other_entity));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration::<Entity>(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let mut res = self.get_acceleration_and_collisions(node, arena, theta);
                let tmp_accel = res.gravitational_acceleration;
                collisions.append(&mut res.collisions);
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        if let Some(node) = &node.right {
            if node.point_indices.is_some() {
                // same logic as above
                let indices = node.point_indices.as_ref().expect("unexpected null node 2");
                for &idx in indices {
                    let other = &arena[idx];
                    // OPTIMIZATION: Cache as_entity() to avoid repeated conversions
                    let other_entity = other.as_entity();
                    if self.did_collide_into(&other_entity) {
                        collisions.push(other);
                    }
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&other_entity));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration::<Entity>(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let mut res = self.get_acceleration_and_collisions(node, arena, theta);
                let tmp_accel = res.gravitational_acceleration;
                collisions.append(&mut res.collisions);
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        SimulationResult {
            collisions,
            gravitational_acceleration: acceleration,
        }
    }
    pub(crate) fn get_acceleration_without_collisions<'a, T: AsEntity + Clone>(
        &'a self,
        node: &'a Node,
        arena: &'a [T],
        theta: f64,
    ) -> SimulationResult<'a, T> {
        let mut acceleration = (0., 0., 0.);
        if let Some(node) = &node.left {
            if node.point_indices.is_some() {
                // if this node has some point_indices, calculate their gravitational acceleration
                let indices = node.point_indices.as_ref().expect("unexpected null node 2");
                for &idx in indices {
                    let other = &arena[idx];
                    // OPTIMIZATION: Cache as_entity() to avoid repeated conversions
                    let other_entity = other.as_entity();
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&other_entity));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration::<Entity>(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let res = self.get_acceleration_without_collisions(node, arena, theta);
                let tmp_accel = res.gravitational_acceleration;
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        if let Some(node) = &node.right {
            if node.point_indices.is_some() {
                // same logic as above
                let indices = node.point_indices.as_ref().expect("unexpected null node 2");
                for &idx in indices {
                    let other = &arena[idx];
                    // OPTIMIZATION: Cache as_entity() to avoid repeated conversions
                    let other_entity = other.as_entity();
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&other_entity));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration::<Entity>(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let res = self.get_acceleration_without_collisions(node, arena, theta);
                let tmp_accel = res.gravitational_acceleration;
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        SimulationResult {
            collisions: vec![],
            gravitational_acceleration: acceleration,
        }
    }
}
