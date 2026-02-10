from enum import IntEnum
from brain import Brain
import math
import random as rng

class Sensor(IntEnum):
    LOC_X = 0
    LOC_Y = 1
    BOUNDARY_DIST = 2
    AGE = 3
    LAST_MOVE_DIR_X = 4
    LAST_MOVE_DIR_Y = 5
    RANDOM = 6
    NEAREST_EDGE_X = 7      # -1 = nearest edge is left, +1 = right, rescaled to [0,1]
    NEAREST_EDGE_Y = 8      # -1 = nearest edge is bottom, +1 = top, rescaled to [0,1]
    POPULATION_DENSITY = 9  # how crowded the area around me is
    BLOCKED_FORWARD = 10    # 1.0 if the cell ahead is blocked, 0.0 if free
    OSCILLATOR = 11         # sine wave that cycles over the generation

class Action(IntEnum):
    MOVE_X = 0
    MOVE_Y = 1
    MOVE_FORWARD = 2
    MOVE_RANDOM = 3
    SET_RESPONSIVENESS = 4

NUM_ACTIONS = len(Action)
NUM_SENSORS = len(Sensor)
NUM_INTERNAL = 4  # Number of internal neurons

class Individual:
    def __init__(self, genome, x, y):
        self.genome = genome
        self.x = x
        self.y = y
        self.brain = Brain(genome, NUM_SENSORS, NUM_ACTIONS, NUM_INTERNAL)  
        self.alive = True
        self.last_dx = 0  # last movement: -1, 0, or +1
        self.last_dy = 0
        self.responsiveness = 0.5

    def compute_sensors(self, grid, step, steps_per_gen):

        # ! READ SENSORS ! 
        # create a dict of sensor values keyed by sensor ID
        sensors = {sensor: 0.0 for sensor in Sensor}

        # LOC_X: 0.0 at left edge, 1.0 at right edge
        sensors[Sensor.LOC_X] = self.x / (grid.width - 1)
        
        # LOC_Y: 0.0 at bottom, 1.0 at top
        sensors[Sensor.LOC_Y] = self.y / (grid.height - 1)

          
        # BOUNDARY_DIST: how far from the nearest edge, normalized
        # min distance to any of the 4 edges, divided by max possible
        dist_to_edge = min(self.x, self.y, grid.width - 1 - self.x, grid.height - 1 - self.y)
        max_possible = min(grid.width, grid.height) // 2
        sensors[Sensor.BOUNDARY_DIST] = dist_to_edge / max_possible

        # AGE: 0.0 at start of generation, 1.0 at end
        sensors[Sensor.AGE] = step / steps_per_gen
        
        # LAST_MOVE_DIR_X: -1/0/+1 rescaled to 0.0/0.5/1.0
        sensors[Sensor.LAST_MOVE_DIR_X] = (self.last_dx + 1) / 2
        
        # LAST_MOVE_DIR_Y: same
        sensors[Sensor.LAST_MOVE_DIR_Y] = (self.last_dy + 1) / 2

        # RANDOM: fresh random value each step
        sensors[Sensor.RANDOM] = rng.random()

        # NEAREST_EDGE_X: which edge is closer horizontally?
        # left half → -1 (rescaled to 0.0), right half → +1 (rescaled to 1.0)
        dist_left = self.x
        dist_right = grid.width - 1 - self.x
        if dist_left <= dist_right:
            sensors[Sensor.NEAREST_EDGE_X] = 0.0  # nearest edge is left
        else:
            sensors[Sensor.NEAREST_EDGE_X] = 1.0  # nearest edge is right

        # NEAREST_EDGE_Y: which edge is closer vertically?
        dist_bottom = self.y
        dist_top = grid.height - 1 - self.y
        if dist_bottom <= dist_top:
            sensors[Sensor.NEAREST_EDGE_Y] = 0.0  # nearest edge is bottom
        else:
            sensors[Sensor.NEAREST_EDGE_Y] = 1.0  # nearest edge is top

        # POPULATION_DENSITY: check 4 adjacent cells (fast)
        neighbors = 0
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = self.x + dx, self.y + dy
            if grid.in_bounds(nx, ny) and grid.is_occupied(nx, ny):
                neighbors += 1
        sensors[Sensor.POPULATION_DENSITY] = neighbors / 4.0

        # BLOCKED_FORWARD: is the cell in my last-moved direction blocked?
        fx = self.x + self.last_dx
        fy = self.y + self.last_dy
        if not grid.in_bounds(fx, fy) or not grid.is_empty(fx, fy):
            sensors[Sensor.BLOCKED_FORWARD] = 1.0
        else:
            sensors[Sensor.BLOCKED_FORWARD] = 0.0

        # OSCILLATOR: sine wave cycling over the generation
        sensors[Sensor.OSCILLATOR] = (math.sin(2 * math.pi * step / steps_per_gen) + 1) / 2

        return {sensor.value: value for sensor, value in sensors.items()}

    
    def execute_actions(self, action_outputs, grid):
         # action_outputs is a list of floats in [-1, 1] from tanh
        
        # Track desired movement
        move_dx = 0
        move_dy = 0

         # MOVE_X: positive = east, negative = west
        val = action_outputs[Action.MOVE_X]
        prob = abs(val) * self.responsiveness
        # chance for example 75% prob only if number is less than prob will movement run
        if rng.random() < prob:
            move_dx += 1 if val > 0 else -1

         # MOVE_Y: positive = north, negative = south
        val = action_outputs[Action.MOVE_Y]
        prob = abs(val) * self.responsiveness
        if rng.random() < prob:
            move_dy += 1 if val > 0 else -1
        
        # MOVE_FORWARD: move in last-moved direction
        val = action_outputs[Action.MOVE_FORWARD]
        prob = abs(val) * self.responsiveness
        if rng.random() < prob:
            move_dx += self.last_dx
            move_dy += self.last_dy

        # MOVE_RANDOM: move in a random direction
        val = action_outputs[Action.MOVE_RANDOM]
        prob = abs(val) * self.responsiveness
        if rng.random() < prob:
            move_dx += rng.choice([-1, 0, 1])
            move_dy += rng.choice([-1, 0, 1])

        # SET_RESPONSIVENESS: adjust how strongly actions fire
        val = action_outputs[Action.SET_RESPONSIVENESS]
        # rescale tanh output [-1,1] to [0,1] and blend with current
        self.responsiveness = (self.responsiveness + (val + 1) / 2) / 2

        # Clamp movement to -1, 0, or +1 in each axis clamping stop jumping multiple squares in grid
        move_dx = max(-1, min(1, move_dx))
        move_dy = max(-1, min(1, move_dy))
        
        # Try to move
        new_x = self.x + move_dx
        new_y = self.y + move_dy
        
        if (move_dx != 0 or move_dy != 0):
            if grid.in_bounds(new_x, new_y) and grid.is_empty(new_x, new_y):
                grid.move(self.x, self.y, new_x, new_y)
                self.x = new_x
                self.y = new_y
                self.last_dx = move_dx
                self.last_dy = move_dy

    def take_step(self, grid, step, steps_per_gen):
        if not self.alive:
            return
        
        # get sensor values from environment
        sensors = self.compute_sensors(grid, step, steps_per_gen)
        
        # get action outputs from brain
        action_outputs = self.brain.feed_forward(sensors)
        
        # execute actions
        self.execute_actions(action_outputs, grid)