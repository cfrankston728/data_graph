"""
Data Graph Package - Tools for creating, visualizing, and analyzing graphs with semimetric edge weights.
"""

# Import main classes for easy access
from .data_graph import DataGraph
from .data_graph_generator import DataGraphGenerator
from .data_graph_visualizer import DataGraphVisualizer
from .data_graph_analyzer import DataGraphAnalyzer

# Import core utilities that might be directly useful
from .core_utilities import (
    TimingStats, 
    BatchStats,
    make_parallel_batcher,
    find_knee_point,
    find_2hop_neighbors_efficient
)

# Define what gets imported with `from manifold_graph import *`
__all__ = [
    # Main classes
    'DataGraph',
    'DataGraphGenerator',
    'DataGraphVisualizer',
    'DataGraphAnalyzer',
    
    # Utility classes
    'TimingStats',
    'BatchStats',
    
    # Core functions
    'make_parallel_batcher',
    'find_knee_point',
    'find_2hop_neighbors_efficient',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Connor Frankston'