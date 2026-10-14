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

    /// Respond using Velocity Verlet integration with accelerations at both t and t+dt.
    /// This provides better energy conservation than single-step methods.
    ///
    /// The position update uses acceleration at time t (a_current), and the velocity
    /// update uses the average of accelerations at t and t+dt (a_current and a_next).
    ///
    /// Default implementation falls back to single-step respond_mut for compatibility.
    fn respond_mut_verlet(
        &mut self,
        simulation_result_current: SimulationResult<Self>,
        _simulation_result_next: SimulationResult<Self>,
        time_step: f64,
    )
    where
        Self: std::marker::Sized + Clone,
    {
        // Default: fall back to single acceleration (backwards compatible)
        self.respond_mut(simulation_result_current, time_step);
    }
}
