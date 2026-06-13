class ThermalDampener:
    def __init__(self, initial_kappa: float):
        self.kappa = initial_kappa
        self.optimal_temp = 50.0  # The "PERFECT" zone target

    def get_temperature(self) -> float:
        # Simple linear physical model for the agent to deduce: Temp = Kappa * 10
        # If Kappa is too low (e.g., 1.0), Temp is 10 (FREEZING)
        # If Kappa is too high (e.g., 9.0), Temp is 90 (BOILING)
        return self.kappa * 10.0

    def update_kappa(self, delta: float):
        self.kappa += delta