import numpy as np

class Grid:
    def __init__(self, width, height):
        self.width = width  # grid width in cells
        self.height = height  # grid height in cells
        self.data = np.zeros((width, height), dtype=np.int32)  # 2D array: 0=empty, >0=creature index+1
        self.barriers = set()  # permanent obstacles, set of (x, y) tuples
    
    def in_bounds(self, x, y):
        # Check if coordinates are within grid boundaries
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_empty(self, x, y):
        # Cell is empty if no creature (0) and not a barrier
        return self.data[x, y] == 0 and (x, y) not in self.barriers
    
    def is_barrier(self, x, y):
        # Check if position is a permanent barrier
        return (x, y) in self.barriers
    
    def is_occupied(self, x, y):
        # Check if a creature is at this position
        return self.data[x, y] != 0
    
    def set(self, x, y, index):
        """Place creature (by index+1) at position"""
        self.data[x, y] = index + 1  # Store as index+1 so 0 means empty
    
    def get(self, x, y):
        """Returns creature index at position, or None if empty"""
        val = self.data[x, y]  # Get stored value
        return val - 1 if val > 0 else None  # Convert back to 0-indexed, None if empty
    
    def move(self, old_x, old_y, new_x, new_y):
        """Move whatever is at old pos to new pos"""
        self.data[new_x, new_y] = self.data[old_x, old_y]  # Copy creature to new position
        self.data[old_x, old_y] = 0  # Clear old position
    
    def clear(self):
        """Clear all creatures but keep barriers"""
        self.data.fill(0)  # Set all cells to 0 (empty)
    
    def add_barrier(self, x, y):
        self.barriers.add((x, y))  # Add position to barrier set
    
    def random_empty_location(self):
        """Find a random empty cell — used for spawning"""
        while True:  # Keep trying until we find an empty spot
            x = np.random.randint(0, self.width)  # Random x coordinate
            y = np.random.randint(0, self.height)  # Random y coordinate
            if self.is_empty(x, y):  # Check if position is valid
                return x, y  # Return first empty spot found
    
    def count_neighbors(self, x, y, radius=1):
        """Count occupied cells within radius — used for POPULATION sensor"""
        count = 0  # Initialize neighbor counter
        for dx in range(-radius, radius + 1):  # Loop through x offsets
            for dy in range(-radius, radius + 1):  # Loop through y offsets
                if dx == 0 and dy == 0:  # Skip the center cell (self)
                    continue
                nx, ny = x + dx, y + dy  # Calculate neighbor position
                if self.in_bounds(nx, ny) and self.is_occupied(nx, ny):  # Valid and occupied?
                    count += 1  # Increment counter
        return count  # Return total neighbors found