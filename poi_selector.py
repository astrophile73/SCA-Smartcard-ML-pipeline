"""
Points of Interest (POI) Selection

Reduces trace dimensionality by selecting the most informative samples.
This is critical for memory-constrained environments.
"""

import numpy as np
from typing import Tuple


class POISelector:
    """Select Points of Interest from power traces to reduce dimensionality."""
    
    def __init__(self, num_poi: int = 5000):
        """
        Initialize POI selector.
        
        Args:
            num_poi: Number of points to select (default 5000, down from 131124)
        """
        self.num_poi = num_poi
        self.poi_indices = None
    
    def select_poi_variance(self, traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Select POI based on variance across traces.
        
        High variance points are more likely to contain information leakage.
        
        Args:
            traces: Array of shape (N, trace_length)
            labels: Labels for supervision (not used in variance method)
            
        Returns:
            Indices of selected POI
        """
        print(f"\nSelecting {self.num_poi} Points of Interest (POI)...")
        print(f"  Original trace length: {traces.shape[1]}")
        
        # Calculate variance across all traces for each time point
        variances = np.var(traces, axis=0)
        
        # Select top N points with highest variance
        self.poi_indices = np.argsort(variances)[-self.num_poi:]
        self.poi_indices = np.sort(self.poi_indices)  # Keep temporal order
        
        print(f"✓ Selected {len(self.poi_indices)} POI")
        print(f"  Variance range: {variances[self.poi_indices].min():.6f} to {variances[self.poi_indices].max():.6f}")
        print(f"  Dimensionality reduction: {traces.shape[1]} → {self.num_poi} ({self.num_poi/traces.shape[1]*100:.1f}%)")
        
        return self.poi_indices
    
    def apply_poi(self, traces: np.ndarray) -> np.ndarray:
        """
        Apply POI selection to traces.
        
        Args:
            traces: Array of shape (N, trace_length)
            
        Returns:
            Reduced traces of shape (N, num_poi)
        """
        if self.poi_indices is None:
            raise ValueError("POI not selected yet. Call select_poi_variance() first.")
        
        return traces[:, self.poi_indices]
    
    def fit_incremental(self, trace_gen):
        """
        Calculate variance incrementally (Welford's algorithm) using a trace generator.
        This is extremely memory-efficient.
        """
        print(f"\nDiscovering POIs incrementally...")
        n = 0
        mean = None
        M2 = None
        
        for trace in trace_gen:
            n += 1
            if mean is None:
                mean = np.zeros_like(trace)
                M2 = np.zeros_like(trace)
            
            delta = trace - mean
            mean += delta / n
            delta2 = trace - mean
            M2 += delta * delta2
            
            if n % 1000 == 0:
                print(f"  Processed {n} traces...")
        
        variances = M2 / n
        self.poi_indices = np.argsort(variances)[-self.num_poi:]
        self.poi_indices = np.sort(self.poi_indices)
        print(f"✓ POI discovery complete ({n} traces processed)")
        return self.poi_indices

    def fit_transform(self, traces: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        """Original batch fit (kept for small datasets)."""
        variances = np.var(traces, axis=0)
        self.poi_indices = np.argsort(variances)[-self.num_poi:]
        self.poi_indices = np.sort(self.poi_indices)
        return traces[:, self.poi_indices]
    
    def transform(self, traces: np.ndarray) -> np.ndarray:
        """
        Apply previously selected POI to new traces.
        
        Args:
            traces: Array of shape (N, trace_length)
            
        Returns:
            Reduced traces of shape (N, num_poi)
        """
        return self.apply_poi(traces)
    
    def save_poi(self, filepath: str):
        """Save POI indices to file."""
        if self.poi_indices is None:
            raise ValueError("No POI selected yet")
        
        np.save(filepath, self.poi_indices)
        print(f"✓ POI indices saved: {filepath}")
    
    def load_poi(self, filepath: str):
        """Load POI indices from file."""
        self.poi_indices = np.load(filepath)
        self.num_poi = len(self.poi_indices)
        print(f"✓ POI indices loaded: {filepath}")
        print(f"  Number of POI: {self.num_poi}")


if __name__ == "__main__":
    # Test POI selector
    print("=" * 60)
    print("Testing POI Selector")
    print("=" * 60)
    
    # Create dummy data
    traces = np.random.randn(100, 131124).astype(np.float32)
    labels = np.random.randint(0, 16, size=(100, 8))
    
    print(f"\nOriginal traces shape: {traces.shape}")
    
    # Select POI
    selector = POISelector(num_poi=5000)
    reduced_traces = selector.fit_transform(traces, labels)
    
    print(f"Reduced traces shape: {reduced_traces.shape}")
    print(f"Memory savings: {(traces.nbytes - reduced_traces.nbytes) / (1024**2):.2f} MB")
    
    # Test transform on new data
    new_traces = np.random.randn(10, 131124).astype(np.float32)
    new_reduced = selector.transform(new_traces)
    print(f"\nNew traces transformed: {new_reduced.shape}")
    
    print("\n" + "=" * 60)
    print("POI Selection Complete!")
    print("=" * 60)
