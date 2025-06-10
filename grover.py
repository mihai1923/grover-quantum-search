import numpy as np
import matplotlib.pyplot as plt
import os
import time
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import Aer
from config import GroverConfig
from analysis import analyze_grover_results

# --- (The core Grover components remain the same) ---
def create_oracle(marked_item_binary):
    n = len(marked_item_binary)
    oracle_qc = QuantumCircuit(n, name=f"Oracle")
    for i, bit in enumerate(reversed(marked_item_binary)):
        if bit == '0':
            oracle_qc.x(i)
    oracle_qc.mcp(np.pi, list(range(n-1)), n-1)
    for i, bit in enumerate(reversed(marked_item_binary)):
        if bit == '0':
            oracle_qc.x(i)
    return oracle_qc

def create_diffuser(n_qubits):
    diffuser_qc = QuantumCircuit(n_qubits, name="Diffuser")
    diffuser_qc.h(range(n_qubits))
    zero_oracle = create_oracle('0' * n_qubits)
    zero_oracle.name = "Oracle(|0..0>)"
    diffuser_qc.append(zero_oracle, range(n_qubits))
    diffuser_qc.h(range(n_qubits))
    return diffuser_qc

# --- Main Simulation and Analysis for 20 Qubits ---
if __name__ == "__main__":
    
    # --- 1. Initialize Configuration ---
    config = GroverConfig()
    np.random.seed(config.RANDOM_SEED)  # For reproducible results
    
    # --- 2. Define the Large-Scale Problem ---
    N_QUBITS = config.N_QUBITS
    # Let's pick a random item to find in our huge database
    # For 20 qubits, this is an integer between 0 and 2**20 - 1
    marked_item_int = np.random.randint(0, 2**N_QUBITS)
    # Convert it to the binary string format our oracle needs
    MARKED_ITEM_BINARY = format(marked_item_int, f'0{N_QUBITS}b')
    
    num_states = config.num_states
    optimal_iterations = config.optimal_iterations
    
    print("="*60)
    print("      Grover's Search: 20-Qubit Simulation")
    print("="*60)
    print(f"Database size (N): {num_states:,} items")
    print(f"Marked item to find (integer): {marked_item_int}")
    print(f"Marked item (binary): |{MARKED_ITEM_BINARY}⟩")
    print(f"Optimal number of Grover iterations: {optimal_iterations}\n")
    print("Preparing simulation... This will take a few minutes.")
    print("We will test a range of iterations to see the probability peak.")    # We will collect the probability of the marked item at each step
    iterations_to_test = config.get_iterations_to_test()
        
    probabilities_of_marked_item = []

    # --- 3. Build the Core Components ---
    backend = Aer.get_backend(config.BACKEND_NAME)
    oracle = create_oracle(MARKED_ITEM_BINARY)
    diffuser = create_diffuser(N_QUBITS)
    
    # Pre-build the initial state circuit
    initial_qc = QuantumCircuit(N_QUBITS)
    initial_qc.h(range(N_QUBITS))
    initial_qc.barrier()
    
    # Pre-build one Grover iteration
    grover_iteration_qc = QuantumCircuit(N_QUBITS)
    grover_iteration_qc.append(oracle, range(N_QUBITS))
    grover_iteration_qc.append(diffuser, range(N_QUBITS))
    grover_iteration_qc.barrier()    # --- 4. Run the Simulation Loop ---
    start_time = time.time()
    
    # The initial state (0 iterations)
    qc = initial_qc.copy()
    statevector = backend.run(transpile(qc, backend)).result().get_statevector()
    
    # Loop through the number of iterations
    for i in range(1, max(iterations_to_test) + 1):
        # Add one more Grover iteration to the circuit
        qc.append(grover_iteration_qc, range(N_QUBITS))
        
        # If this is an iteration we want to measure, run the simulation
        if i in iterations_to_test:
            print(f"Simulating for {i} iterations...")
            # Run the full circuit and get the statevector
            statevector = backend.run(transpile(qc, backend)).result().get_statevector()
            # Calculate probabilities
            probabilities = np.abs(statevector)**2
            # Store the probability of the specific marked item
            prob_marked = probabilities[marked_item_int]
            probabilities_of_marked_item.append(prob_marked)
    
    # Manually add the probability for 0 iterations
    prob_at_zero_iter = (1 / num_states)
    probabilities_of_marked_item.insert(0, prob_at_zero_iter)

    end_time = time.time()
    print(f"\nSimulation complete. Total time: {end_time - start_time:.2f} seconds.\n")

    # --- 5. Enhanced Analysis and Visualization ---
    analyzer = analyze_grover_results(
        iterations_to_test, 
        probabilities_of_marked_item, 
        N_QUBITS, 
        marked_item_int, 
        optimal_iterations,
        config.OUTPUT_DIR
    )