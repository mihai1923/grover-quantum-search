# Configuration file for Grover's algorithm simulation
import numpy as np

class GroverConfig:
    # Simulation parameters
    N_QUBITS = 20
    RANDOM_SEED = 42  # For reproducible results
    
    # Analysis parameters
    ITERATION_STEP = 25  # Test every N iterations
    OVERSHOOT_FACTOR = 1.6  # Test up to 1.6 * optimal iterations
    
    # Output settings
    OUTPUT_DIR = "grover_scaling"
    PLOT_DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-whitegrid'
    
    # Backend settings
    BACKEND_NAME = 'statevector_simulator'
    SHOTS = 1024  # For shot-based simulations
    
    @property
    def num_states(self):
        return 2**self.N_QUBITS
    
    @property
    def optimal_iterations(self):
        return int(np.pi / 4 * np.sqrt(self.num_states))
    
    @property
    def max_iterations_to_test(self):
        return int(self.optimal_iterations * self.OVERSHOOT_FACTOR)
    
    def get_iterations_to_test(self):
        iterations = np.arange(0, self.max_iterations_to_test + 1, self.ITERATION_STEP)
        # Ensure optimal iteration is included
        if self.optimal_iterations not in iterations:
            iterations = np.append(iterations, self.optimal_iterations)
            iterations.sort()
        return iterations
