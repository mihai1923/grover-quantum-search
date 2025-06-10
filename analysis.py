
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
import os
from datetime import datetime

class GroverAnalyzer:
    def __init__(self, iterations: List[int], probabilities: List[float], 
                 n_qubits: int, marked_item: int, optimal_iterations: int):
        self.iterations = np.array(iterations)
        self.probabilities = np.array(probabilities)
        self.n_qubits = n_qubits
        self.marked_item = marked_item
        self.optimal_iterations = optimal_iterations
        self.num_states = 2**n_qubits
    
    def find_peak_probability(self) -> Tuple[int, float]:
        max_idx = np.argmax(self.probabilities)
        return self.iterations[max_idx], self.probabilities[max_idx]
    
    def calculate_theoretical_probability(self, iteration: int) -> float:
        theta = 2 * np.arcsin(1 / np.sqrt(self.num_states))
        return np.sin((2 * iteration + 1) * theta / 2)**2
    
    def analyze_performance(self) -> dict:
        peak_iter, peak_prob = self.find_peak_probability()
        
        # Find probability at optimal iteration
        opt_idx = np.where(self.iterations == self.optimal_iterations)[0]
        prob_at_optimal = self.probabilities[opt_idx[0]] if len(opt_idx) > 0 else None
        
        # Calculate theoretical maximum
        theoretical_max = self.calculate_theoretical_probability(self.optimal_iterations)
        
        analysis = {
            'n_qubits': self.n_qubits,
            'database_size': self.num_states,
            'marked_item': self.marked_item,
            'optimal_iterations_theoretical': self.optimal_iterations,
            'peak_iteration_observed': int(peak_iter),
            'peak_probability_observed': float(peak_prob),
            'probability_at_optimal': float(prob_at_optimal) if prob_at_optimal is not None else None,
            'theoretical_max_probability': float(theoretical_max),
            'efficiency': float(peak_prob / theoretical_max) if theoretical_max > 0 else 0,
            'iteration_accuracy': abs(peak_iter - self.optimal_iterations) / self.optimal_iterations,
            'speedup_factor': self.num_states / self.optimal_iterations,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def create_enhanced_plot(self, output_dir: str = "grover_scaling") -> str:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main plot
        ax1.plot(self.iterations, self.probabilities, 'o-', 
                label='Simulated Results', color='#1f77b4', markersize=4, linewidth=2)
        
        # Theoretical curve
        theoretical_iters = np.linspace(0, max(self.iterations), 1000)
        theoretical_probs = [self.calculate_theoretical_probability(i) for i in theoretical_iters]
        ax1.plot(theoretical_iters, theoretical_probs, '--', 
                label='Theoretical Curve', color='#ff7f0e', alpha=0.7, linewidth=2)
        
        # Highlight optimal point
        ax1.axvline(x=self.optimal_iterations, color='#d62728', linestyle='--', 
                   label=f'Optimal Iterations ({self.optimal_iterations})', linewidth=2)
        
        # Mark peak
        peak_iter, peak_prob = self.find_peak_probability()
        ax1.plot(peak_iter, peak_prob, 'r*', markersize=15, 
                label=f'Peak: {peak_iter} iter, P={peak_prob:.3f}')
        
        ax1.set_xlabel("Number of Grover Iterations", fontsize=12)
        ax1.set_ylabel("Success Probability", fontsize=12)
        ax1.set_title(f"Grover's Algorithm: {self.n_qubits}-Qubit Search ({self.num_states:,} items)", 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Zoomed view around optimal
        zoom_range = max(50, self.optimal_iterations // 10)
        zoom_start = max(0, self.optimal_iterations - zoom_range)
        zoom_end = self.optimal_iterations + zoom_range
        
        zoom_mask = (self.iterations >= zoom_start) & (self.iterations <= zoom_end)
        if np.any(zoom_mask):
            ax2.plot(self.iterations[zoom_mask], self.probabilities[zoom_mask], 
                    'o-', color='#1f77b4', markersize=6, linewidth=2)
            
            zoom_theoretical_iters = np.linspace(zoom_start, zoom_end, 200)
            zoom_theoretical_probs = [self.calculate_theoretical_probability(i) for i in zoom_theoretical_iters]
            ax2.plot(zoom_theoretical_iters, zoom_theoretical_probs, '--', 
                    color='#ff7f0e', alpha=0.7, linewidth=2)
            
            ax2.axvline(x=self.optimal_iterations, color='#d62728', linestyle='--', linewidth=2)
            ax2.set_xlabel("Number of Grover Iterations", fontsize=12)
            ax2.set_ylabel("Success Probability", fontsize=12)
            ax2.set_title(f"Zoomed View Around Optimal Point", fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"grover_enhanced_analysis_{self.n_qubits}q.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def save_analysis(self, output_dir: str = "grover_scaling") -> str:
        analysis = self.analyze_performance()
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"grover_analysis_{self.n_qubits}q.json")
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return filepath
    
    def print_summary(self):
        analysis = self.analyze_performance()
        
        print("\n" + "="*60)
        print("           GROVER'S ALGORITHM ANALYSIS SUMMARY")
        print("="*60)
        print(f"Database size: {analysis['database_size']:,} items ({analysis['n_qubits']} qubits)")
        print(f"Marked item: {analysis['marked_item']}")
        print(f"Theoretical optimal iterations: {analysis['optimal_iterations_theoretical']}")
        print(f"Observed peak at iteration: {analysis['peak_iteration_observed']}")
        print(f"Peak success probability: {analysis['peak_probability_observed']:.4f}")
        print(f"Theoretical maximum: {analysis['theoretical_max_probability']:.4f}")
        print(f"Algorithm efficiency: {analysis['efficiency']:.2%}")
        print(f"Quantum speedup factor: {analysis['speedup_factor']:.1f}x")
        print(f"Iteration accuracy: {analysis['iteration_accuracy']:.2%} deviation from optimal")
        print("="*60)

def analyze_grover_results(iterations: List[int], probabilities: List[float], 
                          n_qubits: int, marked_item: int, optimal_iterations: int,
                          output_dir: str = "grover_scaling") -> GroverAnalyzer:
    analyzer = GroverAnalyzer(iterations, probabilities, n_qubits, marked_item, optimal_iterations)
    
    # Generate enhanced plot
    plot_path = analyzer.create_enhanced_plot(output_dir)
    print(f"Enhanced analysis plot saved to: {plot_path}")
    
    # Save analysis data
    analysis_path = analyzer.save_analysis(output_dir)
    print(f"Analysis data saved to: {analysis_path}")
    
    # Print summary
    analyzer.print_summary()
    
    return analyzer
