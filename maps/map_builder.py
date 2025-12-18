import yaml
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import os

class Nav2Map:
    """A class to represent a Nav2 map. 
       This assumes the map is bottom left origin.
       Adjust as necessary for different conventions.
    """
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self._load_map()

    def _load_map(self):
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        pgm_path = os.path.join(os.path.dirname(self.yaml_path), data["image"])
        self.resolution = data["resolution"]
        self.origin     = np.array(data["origin"])
        occ_thresh      = data["occupied_thresh"] #occupancy
        free_thresh     = data["free_thresh"]

        # Load Map PGM
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 0)

        occ_grid = np.zeros_like(img, dtype=np.uint8)
        occ_grid[img < occ_thresh * 255] = 1 # occupied
        occ_grid[img > free_thresh * 255] = 0 # free
        occ_grid[(img > occ_thresh * 255) & (img < free_thresh * 255)] = 2 # unknown
        self.occ_grid = occ_grid
        self.height, self.width = occ_grid.shape
        
        robot_radius = 0.4  # meters
        inflation_radius = robot_radius + 0.1  # meters
        self._compute_inflation(radius=inflation_radius)  # meters

    def _compute_inflation(self, radius):
        radius_px = int(radius / self.resolution)
        obstacle_mask = (self.occ_grid == 1)

        # Compute distance transform Euclidean
        dist = distance_transform_edt(~obstacle_mask)
        dist = np.clip(dist, 0, radius_px)

        # Create inflated costmap
        self.inflation_costmap = (255 * (1 - dist / radius_px)).astype(np.uint8)
        self.inflation_costmap[obstacle_mask] = 255 #walls

    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        grid_y = self.height - grid_y - 1  # Invert y-axis

        return grid_y, grid_x
    
    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.resolution + self.origin[0]
        y = (self.height - grid_y - 1) * self.resolution + self.origin[1]

        return x, y
    
    def is_occupied(self, grid_x, grid_y):
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.occ_grid[grid_y, grid_x] == 1
        else:
            raise IndexError("Grid coordinates out of bounds")