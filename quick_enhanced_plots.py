import numpy as np
import matplotlib.pyplot as plt
import os

def create_speedup_evolution_plot():
    print("Creating speedup evolution plot...")
    
    qubits = np.arange(1, 21)
    database_sizes = 2**qubits
    
    # Classical search (average case)
    classical_queries = database_sizes / 2
    
    # Grover's algorithm
    grover_queries = np.pi / 4 * np.sqrt(database_sizes)
    
    # Speedup factor
    speedup = classical_queries / grover_queries
    
    plt.figure(figsize=(14, 8))
    
    # Main speedup plot
    plt.subplot(2, 2, 1)
    plt.loglog(database_sizes, speedup, 'bo-', linewidth=2, markersize=6, label='Grover Speedup')
    plt.loglog(database_sizes, np.sqrt(database_sizes), 'r--', linewidth=2, alpha=0.7, label='Theoretical √N')
    plt.xlabel('Database Size')
    plt.ylabel('Speedup Factor')
    plt.title('Quantum Speedup vs Database Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Queries comparison
    plt.subplot(2, 2, 2)
    plt.loglog(database_sizes, classical_queries, 'r-', linewidth=2, label='Classical Linear')
    plt.loglog(database_sizes, grover_queries, 'b-', linewidth=2, label="Grover's Quantum")
    plt.fill_between(database_sizes, classical_queries, grover_queries, alpha=0.3, color='green', label='Quantum Advantage')
    plt.xlabel('Database Size')
    plt.ylabel('Queries Required')
    plt.title('Query Requirements Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speedup by qubit count
    plt.subplot(2, 2, 3)
    plt.plot(qubits, speedup, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup by Qubit Count')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Relative improvement
    plt.subplot(2, 2, 4)
    relative_improvement = (classical_queries - grover_queries) / classical_queries * 100
    plt.plot(qubits, relative_improvement, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Improvement (%)')
    plt.title('Percentage Improvement over Classical')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("grover_scaling", exist_ok=True)
    filepath = "grover_scaling/speedup_evolution_analysis.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_probability_oscillation_plot():
    print("Creating probability oscillation plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    qubit_examples = [6, 8, 10, 12]
    
    for i, n_qubits in enumerate(qubit_examples):
        ax = axes[i//2, i%2]
        
        database_size = 2**n_qubits
        optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
        
        # Test iterations up to 2x optimal
        iterations = np.arange(0, 2 * optimal_iterations + 1)
        probabilities = []
        
        for it in iterations:
            if it == 0:
                prob = 1.0 / database_size
            else:
                theta = 2 * np.arcsin(1 / np.sqrt(database_size))
                prob = np.sin((2 * it + 1) * theta / 2)**2
            probabilities.append(prob)
        
        ax.plot(iterations, probabilities, 'b-', linewidth=2)
        ax.axvline(optimal_iterations, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal: {optimal_iterations}')
        ax.axhline(0.5, color='green', linestyle=':', alpha=0.7, label='50% threshold')
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Success Probability')
        ax.set_title(f'{n_qubits} Qubits ({database_size:,} items)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.suptitle('Grover Algorithm Probability Oscillations', fontsize=16)
    plt.tight_layout()
    
    filepath = "grover_scaling/probability_oscillations.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_complexity_comparison_chart():
    print("Creating complexity comparison chart...")
    
    # Data for different algorithm complexities
    n_values = np.logspace(1, 6, 100)  # From 10 to 1,000,000
    
    # Different algorithm complexities
    constant = np.ones_like(n_values)  # O(1)
    logarithmic = np.log2(n_values)  # O(log N)
    linear = n_values  # O(N)
    sqrt_n = np.sqrt(n_values)  # O(√N) - Grover's
    n_log_n = n_values * np.log2(n_values)  # O(N log N)
    quadratic = n_values**2  # O(N²)
    
    plt.figure(figsize=(14, 10))
    
    # Main complexity plot
    plt.subplot(2, 2, 1)
    plt.loglog(n_values, constant, label='O(1) - Constant', linewidth=2)
    plt.loglog(n_values, logarithmic, label='O(log N) - Logarithmic', linewidth=2)
    plt.loglog(n_values, sqrt_n, label='O(√N) - Grover\'s Algorithm', linewidth=3, color='blue')
    plt.loglog(n_values, linear, label='O(N) - Classical Search', linewidth=3, color='red')
    plt.loglog(n_values, n_log_n, label='O(N log N) - Comparison Sort', linewidth=2)
    plt.loglog(n_values, quadratic, label='O(N²) - Quadratic', linewidth=2)
    
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Operations Required')
    plt.title('Algorithm Complexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speedup comparison (Classical vs Grover)
    plt.subplot(2, 2, 2)
    speedup = linear / sqrt_n
    plt.loglog(n_values, speedup, 'g-', linewidth=3, label='Grover Speedup (N/√N)')
    plt.loglog(n_values, np.sqrt(n_values), 'g--', alpha=0.7, label='√N Reference')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Speedup Factor')
    plt.title('Quantum Search Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time savings percentage
    plt.subplot(2, 2, 3)
    time_saved = (linear - sqrt_n) / linear * 100
    plt.semilogx(n_values, time_saved, 'purple', linewidth=3)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Time Saved (%)')
    plt.title('Percentage Time Savings')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Practical examples
    plt.subplot(2, 2, 4)
    # Realistic problem sizes and their speedups
    problem_sizes = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
    qubits = [4, 6, 8, 10, 12, 14, 16, 18]
    speedups = [size / np.sqrt(size) for size in problem_sizes]
    
    bars = plt.bar(range(len(qubits)), speedups, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Speedup Factor')
    plt.title('Practical Quantum Speedups')
    plt.xticks(range(len(qubits)), [f'{q}\\n({2**q:,})' for q in qubits], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + speedup*0.02,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    filepath = "grover_scaling/complexity_comparison_chart.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    print("🎨 Creating Enhanced Grover Algorithm Visualizations")
    print("=" * 60)
    
    plots_created = []
    
    try:
        # Speedup evolution
        plot1 = create_speedup_evolution_plot()
        plots_created.append(("Speedup Evolution Analysis", plot1))
        
        # Probability oscillations
        plot2 = create_probability_oscillation_plot()
        plots_created.append(("Probability Oscillations", plot2))
        
        # Complexity comparison
        plot3 = create_complexity_comparison_chart()
        plots_created.append(("Complexity Comparison Chart", plot3))
        
        print(f"\n✅ Successfully created {len(plots_created)} enhanced plots:")
        for name, path in plots_created:
            print(f"  📊 {name}: {path}")
            
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
