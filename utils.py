from typing import List, Dict, Tuple, Hashable

import numpy as np

# def normal_2d(mean: np.ndarray, cov: np.ndarray)

def surface_height(u, v, grid_x, grid_y, surface_values):
       """Map u, v to the nearest indices in the precomputed surface."""
       # Map u, v to grid indices
       idx_x = np.abs(grid_x[0, :] - u).argmin()  # Closest x-index
       idx_y = np.abs(grid_y[:, 0] - v).argmin()  # Closest y-index
    
       # Fetch the height value
       return surface_values[idx_y, idx_x]  # Note: grid_y is vertical (row), grid_x is horizontal (col)
