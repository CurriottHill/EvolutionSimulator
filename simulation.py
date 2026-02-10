"""
simulation.py — The main evolution loop.

Ties together: Grid, Individual, Genome, Gene, Brain.

The loop is simple:
    1. Spawn creatures on the grid (random genomes for gen 1, children after)
    2. Run N steps — each creature senses, thinks, acts
    3. Apply selection — who survived the challenge?
    4. Reproduce — pair survivors, crossover + mutate, make child genomes
    5. Go to 1 with child genomes

This file is written to match YOUR existing interfaces in:
    gene.py, genome.py, brain.py, grid.py, individual.py
"""

import random
from enum import IntEnum
from grid import Grid
from individual import Individual, NUM_SENSORS, NUM_ACTIONS
from genome import Genome


# ── Selection criteria ─────────────────────────────────────────
# These define WHAT the creatures must evolve to do.
# "RIGHT_HALF" means: at the end of the generation, creatures
# whose x position is in the right half of the grid survive.
# Everything else dies. Simple, but enough to prove evolution works.

class SelectionCriteria(IntEnum):
    RIGHT_HALF = 0
    LEFT_HALF = 1
    CENTER_CIRCLE = 2
    CORNERS = 3
    RIGHT_QUARTER = 4


class Simulation:

    def __init__(
        self,
        world_width=128,         # grid width in cells
        world_height=128,        # grid height in cells
        population_size=1000,    # how many creatures per generation
        genome_length=24,        # how many genes each creature starts with
        num_internal_neurons=3,  # hidden neurons in the brain
        steps_per_gen=300,       # sim steps before selection happens
        mutation_rate=0.01,      # chance each gene mutates per generation
        selection=SelectionCriteria.RIGHT_QUARTER,
    ):
        # ── Store all config ──
        # These are the knobs your GUI sliders will control later

        self.world_width = world_width
        self.world_height = world_height
        self.population_size = population_size
        self.genome_length = genome_length
        self.num_internal_neurons = num_internal_neurons
        self.steps_per_gen = steps_per_gen
        self.mutation_rate = mutation_rate
        self.selection = selection

        # ── Create the world ──
        # Grid is just a 2D numpy array: 0 = empty, >0 = creature
        self.grid = Grid(world_width, world_height)

        # ── Population ──
        # List of Individual objects. Index in this list = creature ID.
        self.individuals: list[Individual] = []

        # ── Tracking ──
        self.generation = 0       # which generation we're on
        self.current_step = 0     # which step within current generation
        self.kill_count = 0       # kills by KILL_FORWARD this generation

        # ── History ──
        # One dict per completed generation. Feeds your graphs.
        self.history: list[dict] = []

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: SPAWN
    # ══════════════════════════════════════════════════════════════

    def spawn_generation(self, genomes=None):
        """
        Place creatures on the grid.

        genomes=None  → generation 1, create random genomes
        genomes=list  → generation 2+, these came from reproduce()

        Every creature gets a RANDOM position. Doesn't matter where
        parents were. The brain must figure out where to go.
        """

        # Wipe all creatures off the grid (barriers stay)
        self.grid.clear()

        # Reset population
        self.individuals = []

        # Reset step and kill counters
        self.current_step = 0
        self.kill_count = 0

        # Generation 1: no genomes provided, make random ones
        if genomes is None:
            genomes = []
            for _ in range(self.population_size):
                genomes.append(Genome.random(self.genome_length))

        # Place each creature on the grid
        for i, genome in enumerate(genomes):

            # Find a random empty cell
            x, y = self._random_empty_location()

            # Create the creature
            # Your Individual.__init__ takes (genome, x, y)
            indiv = Individual(genome, x, y)

            # Store it
            self.individuals.append(indiv)

            # Register in grid. Your grid.set does index+1 internally,
            # so we pass the raw index i.
            self.grid.set(x, y, i)

    def _random_empty_location(self):
        """
        Pick random coordinates until we find an empty, non-barrier cell.

        Fast when grid is sparse. 1000 creatures on 128x128 = 6% full,
        so we'll almost always find a spot on the first try.
        """
        while True:
            x = random.randint(0, self.world_width - 1)
            y = random.randint(0, self.world_height - 1)

            # Need to check: in bounds, not occupied, not a barrier
            # NOTE: your grid.in_bounds() is actually checking is_empty,
            # not bounds. So we do the bounds check manually here and
            # use is_occupied + is_barrier for the rest.
            if (0 <= x < self.world_width and
                0 <= y < self.world_height and
                not self.grid.is_occupied(x, y) and
                not self.grid.is_barrier(x, y)):
                return x, y

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: SIMULATE
    # ══════════════════════════════════════════════════════════════

    def run_step(self):
        """
        One tick of the simulation. Every living creature:
          1. Reads its sensors (where am I? who's near me?)
          2. Feeds sensors through brain (neural net forward pass)
          3. Executes actions (move, kill, etc.)

        Your Individual.take_step() does all three of these.
        It takes (grid, step, steps_per_gen).
        """

        for indiv in self.individuals:
            if not indiv.alive:
                continue

            # Your take_step calls:
            #   compute_sensors → brain.feed_forward → execute_actions
            indiv.take_step(self.grid, self.current_step, self.steps_per_gen)

        self.current_step += 1

    def run_all_steps(self):
        """
        Run every step in the current generation.
        Use this for headless mode (no GUI).
        For GUI mode, call run_step() one at a time so you can render between.
        """
        for _ in range(self.steps_per_gen):
            self.run_step()

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: SELECTION
    # ══════════════════════════════════════════════════════════════

    def apply_selection(self):
        """
        Natural selection. Check each creature's final position.
        If it doesn't meet the criteria, it's dead.

        Returns list of survivors (Individual objects).
        """
        survivors = []

        for indiv in self.individuals:
            # Already dead — killed by another creature during the sim
            if not indiv.alive:
                continue

            # Does this creature pass the survival challenge?
            if self._survives(indiv):
                survivors.append(indiv)

        return survivors

    def _survives(self, indiv):
        """
        Check one creature against the current selection criteria.

        Start with RIGHT_HALF — survival rate should climb from ~50%
        (random chance, half are on the right) up to 80-90%+ as
        creatures evolve brains that move them rightward.
        """
        w = self.world_width
        h = self.world_height

        if self.selection == SelectionCriteria.RIGHT_HALF:
            return indiv.x >= w // 2

        elif self.selection == SelectionCriteria.LEFT_HALF:
            return indiv.x < w // 2

        elif self.selection == SelectionCriteria.CENTER_CIRCLE:
            # Survive if within a circle at the center
            cx, cy = w / 2, h / 2
            radius = min(w, h) / 4
            dist_sq = (indiv.x - cx) ** 2 + (indiv.y - cy) ** 2
            return dist_sq <= radius ** 2

        elif self.selection == SelectionCriteria.CORNERS:
            # Survive if in any of the four corners
            quarter_w = w // 4
            quarter_h = h // 4
            in_left = indiv.x < quarter_w
            in_right = indiv.x >= w - quarter_w
            in_bottom = indiv.y < quarter_h
            in_top = indiv.y >= h - quarter_h
            return (in_left or in_right) and (in_bottom or in_top)

        elif self.selection == SelectionCriteria.RIGHT_QUARTER:
            # Harder than right half — only rightmost 25% survives
            return indiv.x >= w * 3 // 4

        # Fallback: everyone survives (no selection = no evolution)
        return True

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: REPRODUCTION
    # ══════════════════════════════════════════════════════════════

    def reproduce(self, survivors):
        """
        Make the next generation's genomes from the survivors.

        Sexual reproduction:
          1. Pick two random parents from survivors
          2. Crossover their genomes (your Genome.crossover)
          3. Mutate the child (your Genome.mutate)
          4. Repeat until we have population_size children

        Note: your Genome.crossover already calls mutate(0.1) internally.
        We call mutate again here with your configured rate for extra control.
        You might want to remove the mutate call from crossover later so
        mutation only happens once at the rate you configure.
        """

        # Zero survivors = extinction. Restart with random genomes.
        if len(survivors) == 0:
            print(f"  Gen {self.generation}: EXTINCTION — restarting")
            return [Genome.random(self.genome_length) for _ in range(self.population_size)]

        # One survivor = can't crossover. Clone + mutate (asexual).
        if len(survivors) == 1:
            parent = survivors[0].genome
            children = []
            for _ in range(self.population_size):
                # Copy genes into new list so mutations don't corrupt parent
                child = Genome([g.copy() for g in parent.genes])
                child.mutate(self.mutation_rate)
                children.append(child)
            return children

        # Normal: 2+ survivors. Sexual reproduction.
        children = []
        for _ in range(self.population_size):
            # Pick two different parents
            parent_a, parent_b = random.sample(survivors, 2)

            # Crossover: your method takes first chunk of A + last chunk of B
            # NOTE: your crossover already calls mutate(0.1) inside it
            child = Genome.crossover(parent_a.genome, parent_b.genome)

            # Additional mutation at our configured rate
            child.mutate(self.mutation_rate)

            children.append(child)

        return children

    # ══════════════════════════════════════════════════════════════
    # FULL GENERATION CYCLE
    # ══════════════════════════════════════════════════════════════

    def run_one_generation(self):
        """
        One complete cycle: all steps → selection → reproduction → spawn next.

        Returns a stats dict for this generation.
        Call this from your GUI loop, or from run() for headless mode.
        """

        # Run all simulation steps
        self.run_all_steps()
        # Debug: are creatures actually moving?
        total_moved = sum(1 for ind in self.individuals if ind.alive and (ind.last_dx != 0 or ind.last_dy != 0))
        print(f"  Creatures that moved at least once: {total_moved}/{self.population_size}")

        # Who survived the challenge?
        survivors = self.apply_selection()
        num_survivors = len(survivors)
        survival_rate = num_survivors / self.population_size

        # Average genome length — interesting to track
        avg_genome_len = 0.0
        if survivors:
            avg_genome_len = sum(len(s.genome.genes) for s in survivors) / len(survivors)

        # Build stats
        stats = {
            "generation": self.generation,
            "survivors": num_survivors,
            "survival_rate": survival_rate,
            "kill_count": self.kill_count,
            "avg_genome_length": avg_genome_len,
        }

        # Print to terminal so you can watch evolution happen
        print(
            f"Gen {self.generation:4d} | "
            f"Survivors: {num_survivors:4d}/{self.population_size} "
            f"({survival_rate:5.1%}) | "
            f"Kills: {self.kill_count:3d} | "
            f"Avg genes: {avg_genome_len:.1f}"
        )

        # Save for graphing
        self.history.append(stats)

        # Reproduce: survivors → child genomes for next gen
        child_genomes = self.reproduce(survivors)

        # Advance
        self.generation += 1

        # Spawn next generation
        self.spawn_generation(child_genomes)

        return stats

    def run(self, num_generations=100):
        """
        Run the full simulation headless (no GUI).

        Usage:
            sim = Simulation()
            sim.spawn_generation()
            sim.run(200)
        """
        for _ in range(num_generations):
            self.run_one_generation()


# ══════════════════════════════════════════════════════════════════
# Test it
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Python Simulator ===")
    print()

    sim = Simulation(
        world_width=128,
        world_height=128,
        population_size=1000,
        genome_length=24,
        num_internal_neurons=3,
        steps_per_gen=300,
        mutation_rate=0.01,
        selection=SelectionCriteria.RIGHT_QUARTER
        ,
    )

    print(f"Grid:       {sim.world_width}x{sim.world_height}")
    print(f"Population: {sim.population_size}")
    print(f"Steps/gen:  {sim.steps_per_gen}")
    print(f"Genes:      {sim.genome_length}")
    print(f"Neurons:    {sim.num_internal_neurons}")
    print(f"Mutation:   {sim.mutation_rate}")
    print(f"Selection:  {sim.selection.name}")
    print("-" * 60)

    sim.spawn_generation()
    sim.run(num_generations=200)

    print("-" * 60)
    if sim.history:
        first = sim.history[0]["survival_rate"]
        last = sim.history[-1]["survival_rate"]
        best = max(h["survival_rate"] for h in sim.history)
        print(f"Gen 0 survival:    {first:.1%}")
        print(f"Gen {len(sim.history)-1} survival: {last:.1%}")
        print(f"Best survival:     {best:.1%}")