import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import List, Dict, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import Aer
from grover import create_oracle, create_diffuser
import time

class AdvancedGroverAnalyzer:
    def __init__(self, output_dir: str = "grover_scaling"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_probability_heatmap(self, max_qubits: int = 12, max_iterations_factor: float = 2.0):
        print("Creating probability heatmap...")
        
        qubit_range = range(4, max_qubits + 1)
        probabilities = []
        iteration_counts = []
        qubit_labels = []
        
        for n_qubits in qubit_range:
            database_size = 2**n_qubits
            optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
            max_iterations = int(optimal_iterations * max_iterations_factor)
            
            # Test iterations from 0 to max_iterations
            test_iterations = np.linspace(0, max_iterations, 50, dtype=int)
            qubit_probs = []
            
            # Random target item for this qubit count
            np.random.seed(42)  # Reproducible results
            target_item = np.random.randint(0, database_size)
            target_binary = format(target_item, f'0{n_qubits}b')
            
            for iterations in test_iterations:
                if iterations == 0:
                    prob = 1.0 / database_size  # Initial uniform probability
                else:
                    # Calculate theoretical probability
                    theta = 2 * np.arcsin(1 / np.sqrt(database_size))
                    prob = np.sin((2 * iterations + 1) * theta / 2)**2
                
                qubit_probs.append(prob)
            
            probabilities.append(qubit_probs)
            if not iteration_counts:  # Only set once
                iteration_counts = test_iterations
            qubit_labels.append(f"{n_qubits}q\\n({database_size:,})")
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        
        # Custom colormap for better visualization
        colors = ['darkblue', 'blue', 'lightblue', 'yellow', 'orange', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        im = plt.imshow(probabilities, cmap=cmap, aspect='auto', interpolation='bilinear')
        
        # Set ticks and labels
        plt.yticks(range(len(qubit_labels)), qubit_labels)
        x_ticks = np.linspace(0, len(iteration_counts)-1, 10, dtype=int)
        plt.xticks(x_ticks, [iteration_counts[i] for i in x_ticks])
        
        plt.xlabel('Grover Iterations')
        plt.ylabel('Problem Size (Qubits)')
        plt.title('Grover Algorithm Success Probability Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Success Probability')
        
        # Add optimal iteration line for each qubit count
        for i, n_qubits in enumerate(qubit_range):
            database_size = 2**n_qubits
            optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
            # Find the closest iteration index
            optimal_idx = np.argmin(np.abs(iteration_counts - optimal_iterations))
            plt.plot(optimal_idx, i, 'w*', markersize=12, markeredgecolor='black', markeredgewidth=1)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "grover_probability_heatmap.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_complexity_landscape(self):
        print("Creating complexity landscape...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create two subplots: 3D surface and 2D contour
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        
        # Generate data
        qubits = np.arange(4, 19)
        database_sizes = 2**qubits
        
        # Create meshgrid
        Q, D = np.meshgrid(qubits, database_sizes)
        
        # Calculate query requirements
        classical_linear = D  # O(N)
        grover_quantum = np.pi / 4 * np.sqrt(D)  # O(√N)
        speedup = classical_linear / grover_quantum
        
        # 3D Surface plot
        surf = ax1.plot_surface(Q, np.log10(D), np.log10(speedup), 
                               cmap='viridis', alpha=0.7, edgecolor='none')
        
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('log₁₀(Database Size)')
        ax1.set_zlabel('log₁₀(Speedup)')
        ax1.set_title('Quantum Speedup Landscape')
        
        # 2D Contour plot
        contour = ax2.contourf(qubits, np.log10(database_sizes), np.log10(speedup), 
                              levels=20, cmap='viridis')
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('log₁₀(Database Size)')
        ax2.set_title('Speedup Contour Map')
        
        # Add contour lines
        lines = ax2.contour(qubits, np.log10(database_sizes), np.log10(speedup), 
                           levels=10, colors='white', alpha=0.5, linewidths=0.5)
        ax2.clabel(lines, inline=True, fontsize=8)
        
        # Add colorbar
        fig.colorbar(contour, ax=ax2, label='log₁₀(Speedup Factor)')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "grover_complexity_landscape.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_iteration_convergence_analysis(self, qubits_to_analyze: List[int] = [8, 12, 16]):
        print("Creating iteration convergence analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(qubits_to_analyze)))
        
        # Plot 1: Probability vs iterations for different qubit counts
        ax1 = axes[0]
        for i, n_qubits in enumerate(qubits_to_analyze):
            database_size = 2**n_qubits
            optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
            
            # Test up to 2x optimal iterations
            iterations = np.arange(0, 2 * optimal_iterations + 1)
            probabilities = []
            
            for it in iterations:
                if it == 0:
                    prob = 1.0 / database_size
                else:
                    theta = 2 * np.arcsin(1 / np.sqrt(database_size))
                    prob = np.sin((2 * it + 1) * theta / 2)**2
                probabilities.append(prob)
            
            ax1.plot(iterations, probabilities, color=colors[i], 
                    label=f'{n_qubits} qubits ({database_size:,} items)', linewidth=2)
            ax1.axvline(optimal_iterations, color=colors[i], linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Success Probability')
        ax1.set_title('Convergence Speed Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time to reach 90% success probability
        ax2 = axes[1]
        qubit_range = range(4, 17)
        iterations_to_90_percent = []
        optimal_iterations_list = []
        
        for n_qubits in qubit_range:
            database_size = 2**n_qubits
            optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
            optimal_iterations_list.append(optimal_iterations)
            
            # Find iterations needed for 90% success
            target_prob = 0.9
            best_iterations = optimal_iterations
            best_prob_diff = float('inf')
            
            for it in range(1, 2 * optimal_iterations + 1):
                theta = 2 * np.arcsin(1 / np.sqrt(database_size))
                prob = np.sin((2 * it + 1) * theta / 2)**2
                prob_diff = abs(prob - target_prob)
                
                if prob_diff < best_prob_diff:
                    best_prob_diff = prob_diff
                    best_iterations = it
            
            iterations_to_90_percent.append(best_iterations)
        
        ax2.plot(list(qubit_range), iterations_to_90_percent, 'bo-', 
                label='Iterations for 90% success', linewidth=2, markersize=6)
        ax2.plot(list(qubit_range), optimal_iterations_list, 'ro-', 
                label='Theoretical optimal', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Iterations Required')
        ax2.set_title('Iterations to Reach 90% Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Success probability distribution
        ax3 = axes[2]
        n_qubits_sample = 10
        database_size = 2**n_qubits_sample
        optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
        
        # Show probability for different iteration counts around optimal
        iteration_range = range(max(1, optimal_iterations - 10), optimal_iterations + 11)
        probs_around_optimal = []
        
        for it in iteration_range:
            theta = 2 * np.arcsin(1 / np.sqrt(database_size))
            prob = np.sin((2 * it + 1) * theta / 2)**2
            probs_around_optimal.append(prob)
        
        bars = ax3.bar(list(iteration_range), probs_around_optimal, 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Highlight optimal iteration
        optimal_idx = list(iteration_range).index(optimal_iterations)
        bars[optimal_idx].set_color('red')
        bars[optimal_idx].set_alpha(1.0)
        
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Success Probability')
        ax3.set_title(f'Success Probability Near Optimal ({n_qubits_sample} qubits)')
        ax3.axvline(optimal_iterations, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_iterations}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Speedup vs database size with confidence intervals
        ax4 = axes[3]
        db_sizes = [2**q for q in range(4, 19)]
        speedups = [db_size / (np.pi / 4 * np.sqrt(db_size)) for db_size in db_sizes]
        theoretical_speedups = [np.sqrt(db_size) for db_size in db_sizes]
        
        ax4.loglog(db_sizes, speedups, 'bo-', label='Grover Speedup', linewidth=2, markersize=6)
        ax4.loglog(db_sizes, theoretical_speedups, 'r--', 
                  label='Theoretical √N', linewidth=2, alpha=0.7)
        
        # Add some uncertainty bands
        speedup_lower = [s * 0.9 for s in speedups]
        speedup_upper = [s * 1.1 for s in speedups]
        ax4.fill_between(db_sizes, speedup_lower, speedup_upper, alpha=0.2, color='blue')
        
        ax4.set_xlabel('Database Size')
        ax4.set_ylabel('Speedup Factor')
        ax4.set_title('Speedup Scaling with Uncertainty')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "grover_convergence_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_quantum_vs_classical_timeline(self):
        print("Creating quantum advantage timeline...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Data for timeline
        qubits = np.arange(1, 21)
        database_sizes = 2**qubits
        classical_time = database_sizes / 2  # Average case linear search
        quantum_time = np.pi / 4 * np.sqrt(database_sizes)
        speedup = classical_time / quantum_time
        
        # Plot 1: Timeline of when quantum becomes advantageous
        ax1.semilogy(qubits, classical_time, 'r-', linewidth=3, label='Classical Linear Search')
        ax1.semilogy(qubits, quantum_time, 'b-', linewidth=3, label="Grover's Quantum Search")
        
        # Fill area where quantum is better
        ax1.fill_between(qubits, classical_time, quantum_time, 
                        where=(quantum_time < classical_time), 
                        color='green', alpha=0.3, label='Quantum Advantage')
        
        # Mark important milestones
        milestones = [
            (4, "Small databases\n(16 items)"),
            (10, "Moderate databases\n(1K items)"),
            (16, "Large databases\n(64K items)"),
            (20, "Very large databases\n(1M items)")
        ]
        
        for qubit, description in milestones:
            db_size = 2**qubit
            classical_queries = db_size / 2
            quantum_queries = np.pi / 4 * np.sqrt(db_size)
            speedup_val = classical_queries / quantum_queries
            
            ax1.annotate(description, 
                        xy=(qubit, quantum_queries), 
                        xytext=(qubit, quantum_queries * 10),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('Queries Required (log scale)')
        ax1.set_title('Classical vs Quantum Search: Performance Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup factor over time
        ax2.plot(qubits, speedup, 'g-', linewidth=3, marker='o', markersize=4)
        
        # Highlight significant speedup thresholds
        thresholds = [2, 10, 100, 1000]
        colors = ['orange', 'red', 'purple', 'darkred']
        
        for threshold, color in zip(thresholds, colors):
            ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, 
                       label=f'{threshold}x speedup')
            
            # Find first qubit count that achieves this speedup
            if np.any(speedup >= threshold):
                first_qubit = qubits[np.where(speedup >= threshold)[0][0]]
                ax2.plot(first_qubit, threshold, 'o', color=color, markersize=8)
                ax2.annotate(f'{first_qubit} qubits', 
                           xy=(first_qubit, threshold),
                           xytext=(first_qubit + 1, threshold * 1.5),
                           arrowprops=dict(arrowstyle='->', color=color),
                           fontsize=9)
        
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Quantum Speedup Factor')
        ax2.set_title('Evolution of Quantum Advantage')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "quantum_advantage_timeline.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_all_enhanced_plots(self):
        print("🎨 Creating enhanced Grover algorithm visualizations...")
        print("This will generate several additional analysis plots")
        
        start_time = time.time()
        created_plots = []
        
        try:
            # Probability heatmap
            plot1 = self.create_probability_heatmap()
            created_plots.append(("Probability Heatmap", plot1))
            
            # Complexity landscape
            plot2 = self.create_complexity_landscape()
            created_plots.append(("Complexity Landscape", plot2))
            
            # Convergence analysis
            plot3 = self.create_iteration_convergence_analysis()
            created_plots.append(("Convergence Analysis", plot3))
            
            # Quantum advantage timeline
            plot4 = self.create_quantum_vs_classical_timeline()
            created_plots.append(("Quantum Advantage Timeline", plot4))
            
        except Exception as e:
            print(f"Warning: Some plots may not have been created due to: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Enhanced visualization complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Created {len(created_plots)} additional plots:")
        
        for name, path in created_plots:
            print(f"  📊 {name}: {path}")
        
        return created_plots

if __name__ == "__main__":
    analyzer = AdvancedGroverAnalyzer()
    analyzer.create_all_enhanced_plots()
