# Evolution Simulator

A neural network evolution simulator built with Python and Pygame. Watch populations of creatures evolve brains that learn to navigate a 2D grid and survive selection pressures over generations.

## How It Works

Each creature has a **genome** — a list of genes that encode neural connections. These connections form a small **brain** that reads sensor inputs and produces movement actions. Every generation:

1. **Sense** — Creatures read environmental data (position, nearest edges, obstacles, etc.)
2. **Think** — The brain processes sensor inputs through weighted connections and internal neurons
3. **Act** — Output signals drive movement (move X/Y, move forward, move random)
4. **Select** — Only creatures in the survival zone reproduce
5. **Reproduce** — Survivors' genomes are crossed over and mutated to create the next generation

Over time, natural selection favours brains that move creatures into the survival zone.

## Features

- **Real-time visualization** — 128x128 grid rendered with Pygame, creatures shown as colored circles
- **Genome-based coloring** — Similar genomes produce similar colors, so you can watch the population converge
- **Adjustable parameters** — Sliders for population size, genome length, internal neurons, steps per generation, mutation rate, and simulation speed
- **Multiple selection criteria** — Right Half, Left Half, Center Circle, Corners, Right Quarter
- **Brain Viewer** — Pause the sim and inspect 10 random creatures' neural networks with a visual map of sensors → neurons → actions, weighted arrows, and a plain-English explanation of their strategy
- **Survival graph** — Track survival rate over generations

## Sensors

| Sensor | Description |
|--------|-------------|
| LOC_X / LOC_Y | Normalized grid position |
| BOUNDARY_DIST | Distance to nearest wall |
| AGE | Progress through the generation (0→1) |
| LAST_MOVE_DIR_X/Y | Previous movement direction |
| RANDOM | Fresh random value each step |
| NEAREST_EDGE_X/Y | Which edge is closer (left/right, top/bottom) |
| POPULATION_DENSITY | How crowded the nearby area is |
| BLOCKED_FORWARD | Whether the cell ahead is blocked |
| OSCILLATOR | Sine wave cycling over the generation |

## Actions

| Action | Description |
|--------|-------------|
| MOVE_X / MOVE_Y | Move left/right or up/down |
| MOVE_FORWARD | Continue in last movement direction |
| MOVE_RANDOM | Random movement |
| SET_RESPONSIVENESS | How strongly the creature reacts to brain output |

## Requirements

- Python 3.10+
- pygame-ce
- NumPy

## Running

```bash
pip install pygame-ce numpy
python main.py
```

## Controls

- **Start** — Begin the simulation with current settings
- **Pause / Resume** — Pause or resume the simulation
- **Reset** — Stop and clear everything
- **View Brain** — Inspect 10 random creatures' neural networks (pauses sim automatically)
- **Selection dropdown** — Choose the survival zone
- **Sliders** — Adjust population, genome length, neurons, steps, mutation rate, speed

## Project Structure

```
├── main.py          # Pygame UI, rendering, and main loop
├── simulation.py    # Simulation orchestration, selection, reproduction
├── individual.py    # Creature class, sensors, and actions
├── brain.py         # Neural network built from genome
├── genome.py        # Genome class, crossover, mutation
├── gene.py          # Gene encoding/decoding, bit-level mutation
└── grid.py          # 2D world grid management
```
