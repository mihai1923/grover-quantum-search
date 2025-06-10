import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from typing import List, Dict, Tuple

class ClassicalSearchAnalyzer:
    def __init__(self, database_size: int):
        self.database_size = database_size
    
    def linear_search_average(self) -> float:
        return (self.database_size + 1) / 2
    
    def random_search_average(self) -> float:
        # For random search with replacement, expected queries = N * (1 - (1-1/N)^k)
        # For finding 1 item, this approaches N for large N
        # More accurately: N * (H_N) where H_N is harmonic number ≈ ln(N) + γ
        if self.database_size <= 1:
            return 1
        return self.database_size * (np.log(self.database_size) + 0.5772)  # Euler-Mascheroni constant
    
    def binary_search_average(self) -> float:
        if self.database_size <= 1:
            return 1
        return np.log2(self.database_size)

class GroverSearchAnalyzer:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.database_size = 2**n_qubits
        self.optimal_iterations = self.calculate_optimal_iterations()
    
    def calculate_optimal_iterations(self) -> int:
        return int(np.pi / 4 * np.sqrt(self.database_size))
    
    def theoretical_success_probability(self, iterations: int = None) -> float:
        if iterations is None:
            iterations = self.optimal_iterations
        
        if iterations == 0:
            return 1.0 / self.database_size
        
        # Grover's algorithm probability formula
        theta = 2 * np.arcsin(1 / np.sqrt(self.database_size))
        return np.sin((2 * iterations + 1) * theta / 2)**2
    
    def queries_needed(self) -> int:
        return self.optimal_iterations

def analyze_search_performance(n_qubits: int) -> Dict:
    database_size = 2**n_qubits
    
    # Classical algorithms
    classical = ClassicalSearchAnalyzer(database_size)
    linear_avg = classical.linear_search_average()
    random_avg = classical.random_search_average()
    binary_avg = classical.binary_search_average()
    
    # Grover's quantum algorithm
    grover = GroverSearchAnalyzer(n_qubits)
    grover_queries = grover.queries_needed()
    grover_success_prob = grover.theoretical_success_probability()
    
    # Calculate speedups
    speedup_vs_linear = linear_avg / grover_queries
    speedup_vs_random = random_avg / grover_queries
    speedup_vs_binary = binary_avg / grover_queries
    
    return {
        'n_qubits': n_qubits,
        'database_size': database_size,
        'linear_avg': linear_avg,
        'random_avg': random_avg,
        'binary_avg': binary_avg,
        'grover_queries': grover_queries,
        'grover_success_prob': grover_success_prob,
        'speedup_vs_linear': speedup_vs_linear,
        'speedup_vs_random': speedup_vs_random,
        'speedup_vs_binary': speedup_vs_binary
    }

def run_extended_comparison_analysis():
    print("Extended Classical vs Quantum Search Comparison")
    print("Analyzing database sizes from 16 items (4 qubits) to 262,144 items (18 qubits)")
    print("="*80)
    
    # Extended range: 4 to 18 qubits
    qubit_range = list(range(4, 19))  # 4, 5, 6, ..., 18
    results = []
    
    start_time = time.time()
    
    for n_qubits in qubit_range:
        print(f"\nAnalyzing {n_qubits} qubits ({2**n_qubits:,} items)...")
        
        result = analyze_search_performance(n_qubits)
        results.append(result)
        
        print(f"  Linear search avg: {result['linear_avg']:,.1f} queries")
        print(f"  Random search avg: {result['random_avg']:,.1f} queries")
        print(f"  Binary search avg: {result['binary_avg']:,.1f} queries")
        print(f"  Grover optimal: {result['grover_queries']:,} queries")
        print(f"  Grover success prob: {result['grover_success_prob']:.4f}")
        print(f"  Speedup vs linear: {result['speedup_vs_linear']:.1f}x")
        print(f"  Speedup vs random: {result['speedup_vs_random']:.1f}x")
        print(f"  Speedup vs binary: {result['speedup_vs_binary']:.1f}x")
    
    total_time = time.time() - start_time
    
    # Create comprehensive visualizations
    plot_path = create_extended_comparison_plots(results)
    
    # Print summary
    print_extended_summary(results, total_time)
    
    return results, plot_path

def create_extended_comparison_plots(results: List[Dict], output_dir: str = "grover_scaling"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    
    # Extract data
    qubits = [r['n_qubits'] for r in results]
    db_sizes = [r['database_size'] for r in results]
    linear_queries = [r['linear_avg'] for r in results]
    random_queries = [r['random_avg'] for r in results]
    binary_queries = [r['binary_avg'] for r in results]
    grover_queries = [r['grover_queries'] for r in results]
    speedup_linear = [r['speedup_vs_linear'] for r in results]
    speedup_random = [r['speedup_vs_random'] for r in results]
    speedup_binary = [r['speedup_vs_binary'] for r in results]
    
    # Plot 1: Query count comparison (log-log)
    ax1 = plt.subplot(2, 3, 1)
    ax1.loglog(db_sizes, linear_queries, 'r-o', linewidth=2, markersize=5, label='Linear Search')
    ax1.loglog(db_sizes, random_queries, 'orange', marker='s', linewidth=2, markersize=5, label='Random Search')
    ax1.loglog(db_sizes, binary_queries, 'purple', marker='^', linewidth=2, markersize=5, label='Binary Search')
    ax1.loglog(db_sizes, grover_queries, 'b-o', linewidth=3, markersize=6, label="Grover's Algorithm")
    
    # Add theoretical lines
    N = np.array(db_sizes)
    ax1.loglog(N, N, 'r--', alpha=0.5, label='O(N) theoretical')
    ax1.loglog(N, np.log2(N), 'purple', linestyle='--', alpha=0.5, label='O(log N) theoretical')
    ax1.loglog(N, np.sqrt(N), 'b--', alpha=0.5, label='O(√N) theoretical')
    
    ax1.set_xlabel('Database Size')
    ax1.set_ylabel('Queries Required')
    ax1.set_title('Algorithm Query Requirements (Log-Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(qubits, speedup_linear, 'r-o', linewidth=2, markersize=6, label='vs Linear Search')
    ax2.plot(qubits, speedup_random, 'orange', marker='s', linewidth=2, markersize=6, label='vs Random Search')
    ax2.plot(qubits, speedup_binary, 'purple', marker='^', linewidth=2, markersize=6, label='vs Binary Search')
    
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title("Grover's Quantum Speedup")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Database size vs speedup
    ax3 = plt.subplot(2, 3, 3)
    ax3.loglog(db_sizes, speedup_linear, 'r-o', linewidth=2, markersize=5, label='vs Linear')
    ax3.loglog(db_sizes, speedup_random, 'orange', marker='s', linewidth=2, markersize=5, label='vs Random')
    ax3.loglog(db_sizes, speedup_binary, 'purple', marker='^', linewidth=2, markersize=5, label='vs Binary')
    
    # Theoretical speedup lines
    ax3.loglog(N, np.sqrt(N), 'g--', linewidth=2, alpha=0.7, label='√N (vs Linear)')
    
    ax3.set_xlabel('Database Size')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Speedup vs Database Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar chart comparison for select sizes
    ax4 = plt.subplot(2, 3, 4)
    
    # Select representative qubit counts
    selected_indices = [0, 4, 8, 12, 14]  # 4, 8, 12, 16, 18 qubits
    selected_qubits = [qubits[i] for i in selected_indices]
    selected_linear = [linear_queries[i] for i in selected_indices]
    selected_random = [random_queries[i] for i in selected_indices]
    selected_binary = [binary_queries[i] for i in selected_indices]
    selected_grover = [grover_queries[i] for i in selected_indices]
    
    x_pos = np.arange(len(selected_qubits))
    width = 0.2
    
    ax4.bar(x_pos - 1.5*width, selected_linear, width, label='Linear', color='red', alpha=0.7)
    ax4.bar(x_pos - 0.5*width, selected_random, width, label='Random', color='orange', alpha=0.7)
    ax4.bar(x_pos + 0.5*width, selected_binary, width, label='Binary', color='purple', alpha=0.7)
    ax4.bar(x_pos + 1.5*width, selected_grover, width, label='Grover', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Problem Size')
    ax4.set_ylabel('Queries Required')
    ax4.set_title('Query Requirements Comparison')
    ax4.set_yscale('log')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{q}q\n({2**q:,})' for q in selected_qubits])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Success probability
    ax5 = plt.subplot(2, 3, 5)
    success_probs = [r['grover_success_prob'] for r in results]
    ax5.plot(qubits, success_probs, 'b-o', linewidth=2, markersize=6)
    ax5.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% threshold')
    ax5.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax5.set_xlabel('Number of Qubits')
    ax5.set_ylabel('Success Probability')
    ax5.set_title("Grover's Algorithm Success Rate")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.8, 1.02)
    
    # Plot 6: Quantum advantage visualization
    ax6 = plt.subplot(2, 3, 6)
    
    # Show the "quantum advantage factor" - how much better quantum is
    quantum_advantage = np.array(speedup_linear)
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantum_advantage)))
    
    bars = ax6.bar(qubits, quantum_advantage, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Number of Qubits')
    ax6.set_ylabel('Quantum Advantage Factor')
    ax6.set_title('Quantum Advantage Growth')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars for key points
    for i, (qubit, advantage) in enumerate(zip(qubits, quantum_advantage)):
        if i % 3 == 0 or i == len(qubits) - 1:  # Label every 3rd bar and the last one
            ax6.text(qubit, advantage * 1.1, f'{advantage:.0f}x', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "extended_classical_vs_quantum_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def print_extended_summary(results: List[Dict], execution_time: float):
    print("\n" + "="*100)
    print("                    EXTENDED CLASSICAL vs QUANTUM SEARCH COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Qubits':<6} {'Database':<12} {'Linear':<12} {'Random':<12} {'Binary':<12} {'Grover':<12} {'Speedup (Linear)':<15}")
    print(f"{'(N)':<6} {'Size':<12} {'Queries':<12} {'Queries':<12} {'Queries':<12} {'Queries':<12} {'Factor':<15}")
    print("-" * 100)
    
    # Data rows
    for result in results:
        print(f"{result['n_qubits']:<6} "
              f"{result['database_size']:<12,} "
              f"{result['linear_avg']:<12,.0f} "
              f"{result['random_avg']:<12,.0f} "
              f"{result['binary_avg']:<12,.1f} "
              f"{result['grover_queries']:<12,} "
              f"{result['speedup_vs_linear']:<15.1f}x")
    
    # Summary statistics
    max_speedup_linear = max(r['speedup_vs_linear'] for r in results)
    max_speedup_random = max(r['speedup_vs_random'] for r in results)
    max_speedup_binary = max(r['speedup_vs_binary'] for r in results)
    avg_success_prob = np.mean([r['grover_success_prob'] for r in results])
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS:")
    print(f"  Maximum speedup vs Linear Search: {max_speedup_linear:,.1f}x")
    print(f"  Maximum speedup vs Random Search: {max_speedup_random:,.1f}x")
    print(f"  Maximum speedup vs Binary Search: {max_speedup_binary:,.1f}x")
    print(f"  Average Grover success probability: {avg_success_prob:.3%}")
    print(f"  Largest database analyzed: {results[-1]['database_size']:,} items ({results[-1]['n_qubits']} qubits)")
    print(f"  Analysis execution time: {execution_time:.2f} seconds")
    print(f"  Theoretical complexity advantage: O(√N) vs O(N)")
    print("="*100)
    
    # Highlight key milestones
    print("\nKEY QUANTUM ADVANTAGE MILESTONES:")
    milestones = [
        (10, "1,000x speedup threshold"),
        (16, "10,000x speedup threshold"),
        (18, "Maximum analyzed size")
    ]
    
    for target_qubits, description in milestones:
        if target_qubits <= results[-1]['n_qubits']:
            result = next((r for r in results if r['n_qubits'] == target_qubits), None)
            if result:
                print(f"  {target_qubits} qubits ({result['database_size']:,} items): "
                      f"{result['speedup_vs_linear']:.0f}x speedup - {description}")

if __name__ == "__main__":
    print("🚀 Extended Classical vs Quantum Search Comparison")
    print("   Supporting up to 18 qubits (262,144 database items)")
    print()
    
    try:
        results, plot_path = run_extended_comparison_analysis()
        print(f"\n📊 Comparison plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
