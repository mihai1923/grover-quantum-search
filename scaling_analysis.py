import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import Aer
import os

def run_grover_analysis(n_qubits, max_time_minutes=5):
    from grover import create_oracle, create_diffuser
    
    print(f"\n{'='*50}")
    print(f"Running Grover's Algorithm for {n_qubits} qubits")
    print(f"{'='*50}")
    
    # Calculate parameters
    num_states = 2**n_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(num_states))
    
    # For larger problems, test fewer points
    if n_qubits <= 10:
        test_iterations = range(0, min(optimal_iterations * 2, 100), max(1, optimal_iterations // 10))
    else:
        test_iterations = range(0, min(optimal_iterations * 2, 200), max(1, optimal_iterations // 5))
    
    print(f"Database size: {num_states:,} items")
    print(f"Optimal iterations: {optimal_iterations}")
    print(f"Testing {len(test_iterations)} iteration points")
    
    # Random marked item
    np.random.seed(42)
    marked_item = np.random.randint(0, num_states)
    marked_binary = format(marked_item, f'0{n_qubits}b')
    
    # Build circuits
    backend = Aer.get_backend('statevector_simulator')
    oracle = create_oracle(marked_binary)
    diffuser = create_diffuser(n_qubits)
    
    # Initialize
    initial_qc = QuantumCircuit(n_qubits)
    initial_qc.h(range(n_qubits))
    
    grover_iteration = QuantumCircuit(n_qubits)
    grover_iteration.append(oracle, range(n_qubits))
    grover_iteration.append(diffuser, range(n_qubits))
    
    probabilities = []
    start_time = time.time()
    
    for iterations in test_iterations:
        if time.time() - start_time > max_time_minutes * 60:
            print(f"Time limit reached ({max_time_minutes} minutes)")
            break
            
        qc = initial_qc.copy()
        for _ in range(iterations):
            qc.append(grover_iteration, range(n_qubits))
        
        try:
            result = backend.run(transpile(qc, backend)).result()
            statevector = result.get_statevector()
            prob = np.abs(statevector[marked_item])**2
            probabilities.append(prob)
            
            if iterations % max(1, len(test_iterations) // 5) == 0:
                print(f"  Iteration {iterations}: P = {prob:.4f}")
                
        except Exception as e:
            print(f"Error at iteration {iterations}: {e}")
            break
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    
    # Find peak
    if probabilities:
        max_prob = max(probabilities)
        max_idx = probabilities.index(max_prob)
        peak_iteration = list(test_iterations)[max_idx]
        
        print(f"Peak probability: {max_prob:.4f} at iteration {peak_iteration}")
        print(f"Theoretical optimum: iteration {optimal_iterations}")
        
        return {
            'n_qubits': n_qubits,
            'iterations': list(test_iterations)[:len(probabilities)],
            'probabilities': probabilities,
            'optimal_iterations': optimal_iterations,
            'peak_iteration': peak_iteration,
            'peak_probability': max_prob,
            'execution_time': elapsed
        }
    
    return None

def create_scaling_comparison(results_list, output_dir="grover_scaling"):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # Plot 1: Success probability curves
    for i, result in enumerate(results_list):
        if result:
            ax1.plot(result['iterations'], result['probabilities'], 
                    'o-', color=colors[i], label=f"{result['n_qubits']} qubits", 
                    markersize=4, alpha=0.8)
            ax1.axvline(x=result['optimal_iterations'], color=colors[i], 
                       linestyle='--', alpha=0.5)
    
    ax1.set_xlabel("Grover Iterations")
    ax1.set_ylabel("Success Probability")
    ax1.set_title("Grover's Algorithm: Multi-Scale Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Scaling analysis
    qubits = [r['n_qubits'] for r in results_list if r]
    optimal_iters = [r['optimal_iterations'] for r in results_list if r]
    peak_probs = [r['peak_probability'] for r in results_list if r]
    exec_times = [r['execution_time'] for r in results_list if r]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(qubits, optimal_iters, 'o-', color='blue', label='Optimal Iterations')
    line2 = ax2_twin.plot(qubits, exec_times, 's-', color='red', label='Execution Time (s)')
    
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("Optimal Iterations", color='blue')
    ax2_twin.set_ylabel("Execution Time (seconds)", color='red')
    ax2.set_title("Grover's Algorithm Scaling")
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "grover_scaling_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nScaling comparison saved to: {filepath}")
    
    return filepath

if __name__ == "__main__":
    print("Grover's Algorithm Multi-Scale Analysis")
    print("This will test different qubit counts to show algorithm scaling")
    
    # Test different qubit counts (adjust based on your system capabilities)
    qubit_counts = [4, 6, 8, 10, 12]  # Start with smaller sizes
    
    results = []
    total_start = time.time()
    
    for n_qubits in qubit_counts:
        result = run_grover_analysis(n_qubits, max_time_minutes=3)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    if results:
        # Create comparison visualization
        create_scaling_comparison(results)
        
        print(f"\n{'='*60}")
        print("SCALING ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total analysis time: {total_time:.2f} seconds")
        print("\nResults by qubit count:")
        for r in results:
            speedup = (2**r['n_qubits']) / r['optimal_iterations']
            print(f"  {r['n_qubits']} qubits: {2**r['n_qubits']:,} items, "
                  f"{r['optimal_iterations']} iterations, {speedup:.1f}x speedup")
        print(f"{'='*60}")
    else:
        print("No results obtained. Try reducing qubit counts or increasing time limits.")
