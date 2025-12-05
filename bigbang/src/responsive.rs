use crate::SimulationResult;

/// Define how to respond to the results of the simulation at every time step.
pub trait Responsive {
    /// Respond to the forces that bigbang has calculated are acting upon the entity.
    /// It is recommended to at least set the position to where the simulation says
    /// it should be and add the velocity to the position. See the examples directory for examples.
    /// Basic collision functions are available in [collisions](crate::collisions].
    fn respond(&self, simulation_result: SimulationResult<Self>, time_step: f64) -> Self
    where
        Self: std::marker::Sized;

    /// Respond to the forces in-place, mutating the entity directly.
    /// This is more efficient than `respond()` as it avoids allocating a new entity.
    /// 
    /// The default implementation calls `respond()` and assigns the result,
    /// but you can override this for better performance.
    fn respond_mut(&mut self, simulation_result: SimulationResult<Self>, time_step: f64)
    where
        Self: std::marker::Sized + Clone,
    {
        *self = self.respond(simulation_result, time_step);
    }
}
