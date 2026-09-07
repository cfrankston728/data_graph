"""
Data Graph Package - Tools for creating, visualizing, and analyzing graphs with semimetric edge weights.
"""

# Import main classes for easy access
from .data_graph import DataGraph
from .construction_coarsening_generator import ConstructionCoarseningDataGraphGenerator as DataGraphGenerator
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


# R189 consolidated public community API
from .community_api import (
    CommunityResult,
    SampledCommunityResult,
    community_backend_status,
    run_communities,
    run_sampled_communities,
)

try:
    __all__
except NameError:
    __all__ = []

for _community_api_name in (
    "CommunityResult",
    "SampledCommunityResult",
    "community_backend_status",
    "run_communities",
    "run_sampled_communities",
):
    if _community_api_name not in __all__:
        __all__.append(_community_api_name)

del _community_api_name
