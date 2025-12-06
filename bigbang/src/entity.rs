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
#[derive(Clone, Default)]
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
}

impl AsEntity for Entity {
    fn as_entity(&self) -> Entity {
        self.clone()
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

        // Velocity Verlet integration - mutate in place
        let vx_half = self.vx + 0.5 * ax * time_step;
        let vy_half = self.vy + 0.5 * ay * time_step;
        let vz_half = self.vz + 0.5 * az * time_step;

        self.x += vx_half * time_step;
        self.y += vy_half * time_step;
        self.z += vz_half * time_step;

        self.vx = vx_half + 0.5 * ax * time_step;
        self.vy = vy_half + 0.5 * ay * time_step;
        self.vz = vz_half + 0.5 * az * time_step;
    }
}

impl Entity {

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
        }
    }

    /// Needs to be reworked to use min/max position values, but it naively checks
    /// if two things collide right now.
    fn did_collide_into(&self, other: &Entity) -> bool {
        self != other && self.distance(other) <= (self.radius + other.radius)
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
    fn distance_squared(&self, other: &Entity) -> f64 {
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
    fn theta_exceeded<T: AsEntity + Clone>(&self, node: &Node<T>, theta: f64) -> bool {
        // 1) distance from entity to COM of that node
        // 2) if 1) * theta > size (max diff) then
        // This frequently makes a node with NaN positions
        let node_as_entity = node.as_entity();
        let dist = self.distance_squared(&node_as_entity);
        let max_dist = node.max_distance();
        (dist) * (theta * theta) > (max_dist * max_dist)
    }

    /// Given two entities, self and other, returns the acceleration that other is exerting on
    /// self. Other can be either an entity or a node.
    fn get_gravitational_acceleration<T: AsEntity + Clone>(
        &self,
        oth: Either<&Entity, &Node<T>>,
    ) -> (f64, f64, f64) {
        // TODO get rid of this clone
        let other = match oth {
            Left(entity) => entity.clone(),
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
        node: &'a Node<T>,
        theta: f64,
    ) -> SimulationResult<T> {
        let mut collisions = Vec::new();
        let mut acceleration = (0., 0., 0.);
        if let Some(node) = &node.left {
            if node.points.is_some() {
                // if this node has some points, calculate their gravitational acceleration
                for i in node.points.as_ref().expect("unexpected null node 2") {
                    if self.did_collide_into(&i.as_entity()) {
                        collisions.push(i);
                    }
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&(i.as_entity())));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let mut res = self.get_acceleration_and_collisions(node, theta);
                let tmp_accel = res.gravitational_acceleration;
                collisions.append(&mut res.collisions);
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        if let Some(node) = &node.right {
            if node.points.is_some() {
                // same logic as above
                for i in node.points.as_ref().expect("unexpected null node 2") {
                    if self.did_collide_into(&i.as_entity()) {
                        collisions.push(i);
                    }
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&(i.as_entity())));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let mut res = self.get_acceleration_and_collisions(node, theta);
                let tmp_accel = res.gravitational_acceleration;
                collisions.append(&mut res.collisions);
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        SimulationResult {
            collisions,
            gravitational_acceleration: (
                acceleration.0,
                acceleration.1,
                acceleration.2,
            ),
        }
    }
    pub(crate) fn get_acceleration_without_collisions<'a, T: AsEntity + Clone>(
        &'a self,
        node: &'a Node<T>,
        theta: f64,
    ) -> SimulationResult<T> {
        let mut acceleration = (0., 0., 0.);
        if let Some(node) = &node.left {
            if node.points.is_some() {
                // if this node has some points, calculate their gravitational acceleration
                for i in node.points.as_ref().expect("unexpected null node 2") {
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&(i.as_entity())));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let res = self.get_acceleration_without_collisions(node, theta);
                let tmp_accel = res.gravitational_acceleration;
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        if let Some(node) = &node.right {
            if node.points.is_some() {
                // same logic as above
                for i in node.points.as_ref().expect("unexpected null node 2") {
                    let tmp_accel =
                        self.get_gravitational_acceleration::<Entity>(Left(&(i.as_entity())));
                    acceleration.0 += tmp_accel.0;
                    acceleration.1 += tmp_accel.1;
                    acceleration.2 += tmp_accel.2;
                }
            } else if self.theta_exceeded(node, theta) {
                // otherwise, if theta is exceeded, calculate the entire node as a big boi particle
                let tmp_accel = self.get_gravitational_acceleration(Right(node));
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            } else {
                // otherwise, theta has not been exceeded and this is not a leaf. recurse
                let res = self.get_acceleration_without_collisions(node, theta);
                let tmp_accel = res.gravitational_acceleration;
                acceleration.0 += tmp_accel.0;
                acceleration.1 += tmp_accel.1;
                acceleration.2 += tmp_accel.2;
            }
        };
        SimulationResult {
            collisions: vec![],
            gravitational_acceleration: (
                acceleration.0,
                acceleration.1,
                acceleration.2,
            ),
        }
    }
}
