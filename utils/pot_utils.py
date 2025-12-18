import numpy as np

def generate_random_pots(nav2_map, n_pots=5):
    pots = []

    while len(pots) < n_pots:
        # random grid location
        gy = np.random.randint(0, nav2_map.height)
        gx = np.random.randint(0, nav2_map.width)

        # skip obstacles or inflated obstacles
        if nav2_map.inflation_costmap[gy, gx] == 255:
            continue

        # convert to world coordinates
        x, y = nav2_map.grid_to_world(gy, gx)
        pots.append((x, y))

    return pots

import numpy as np

def generate_grid_pots_auto(nav2_map, rows=3, cols=2, spacing=0.5, min_clearance=0.4):
    """
    Automatically places a rows×cols grid in a free space area.
    Ensures minimum distance from walls using inflation costmap.
    """

    free_mask = (nav2_map.inflation_costmap < 10)   # low cost = free
    H, W = free_mask.shape

    # Convert min_clearance (meters) to pixels
    clearance_px = int(min_clearance / nav2_map.resolution)

    # Compute required grid dimensions in pixels
    grid_w_px = int(spacing * (cols - 1) / nav2_map.resolution)
    grid_h_px = int(spacing * (rows - 1) / nav2_map.resolution)

    # Search for valid placement region
    best_candidate = None

    for gy in range(clearance_px, H - clearance_px - grid_h_px):
        for gx in range(clearance_px, W - clearance_px - grid_w_px):

            # Check 3×2 grid positions
            valid = True
            for r in range(rows):
                for c in range(cols):

                    py = gy + int(r * spacing / nav2_map.resolution)
                    px = gx + int(c * spacing / nav2_map.resolution)

                    if not free_mask[py, px]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                best_candidate = (gx, gy)
                break
        if best_candidate:
            break

    if best_candidate is None:
        raise RuntimeError("No free 3x2 grid space found!")

    gx0, gy0 = best_candidate

    # Convert grid coordinates → world coordinates
    pot_positions = []
    for r in range(rows):
        for c in range(cols):
            gy = gy0 + int(r * spacing / nav2_map.resolution)
            gx = gx0 + int(c * spacing / nav2_map.resolution)
            wx, wy = nav2_map.grid_to_world(gy, gx)
            pot_positions.append((wx, wy))

    return pot_positions