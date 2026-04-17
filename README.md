
# Plants vs. Zombies RL

A reinforcement learning implementation of Plants vs. Zombies game logic in Python.

## Overview

This project provides a complete game simulation engine for Plants vs. Zombies, designed for reinforcement learning research. It includes plant placement, zombie spawning, combat mechanics, and level progression.

## Features

- **Plant System**: 10 plant types with unique abilities (damage, sun production, crowd control)
- **Zombie Types**: 8 zombie variants with different stats and special behaviors
- **Wave-based Spawning**: Configurable difficulty progression with flag waves
- **Combat Mechanics**: 
    - Plant attacks with targeting
    - Zombie special abilities (pole vault, newspaper shield, etc.)
    - Slow effects and shield systems
    - Lawn mower defense
- **Level Configuration**: Customizable difficulty, wave timing, and zombie distributions

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `game_logic.py` - Main game engine (`PvZGame` class)
- `pools.py` - Plant and zombie data structures
- `plants.csv` - Plant stats configuration
- `zombies.csv` - Zombie stats configuration

## Usage

```python
from game_logic import PvZGame, LevelConfig
import numpy as np

# Create level configuration
config = LevelConfig(
        n_flags=3,
        p_init=np.array([1.0, 0.5, 0.3]),  # Initial zombie distribution
        p_fin=np.array([0.2, 1.0, 1.0])    # Final zombie distribution
)

# Initialize game
game = PvZGame(config)

# Simulate gameplay
game.place_plant(plant_type=1, row=0, col=2)
info = game.update(dt=0.1)
```

## Dependencies

- numpy >= 2.4.4
- pandas >= 3.0.2
