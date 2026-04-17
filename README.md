
# Plants vs Zombies Reinforcement Learning

A Python implementation of Plants vs Zombies game logic with support for reinforcement learning agents.

## Overview

This project simulates the core game mechanics of Plants vs Zombies, including:
- **Plant placement and abilities** (damage, sun production, crowd control)
- **Zombie spawning and movement** (with special abilities like pole vaulting)
- **Resource management** (sun collection and seed cooldowns)
- **Wave-based progression** with configurable difficulty scaling

## Project Structure

- `game_logic.py` - Core game simulation and update loop
- `pools.py` - Plant and zombie state management
- `plants.csv` - Plant stats (health, cooldown, damage, cost, etc.)
- `zombies.csv` - Zombie stats (health, speed, shield, special abilities)

## Key Classes

### `PvZGame`
Main game controller handling:
- Plant placement and removal
- Zombie spawning and wave management
- Game state updates and win/lose conditions

### `LevelConfig`
Configurable level parameters:
- Wave composition and difficulty ramping
- Sun generation rates
- Lawn mower availability

### `PlantGrid` & `ZombiePool`
State management for entities using NumPy arrays for efficient vectorized operations.

## Usage

```python
config = LevelConfig(n_flags=5, p_init=np.array([...]), p_fin=np.array([...]))
game = PvZGame(config)
game.place_plant(plant_type=1, row=2, col=3)
step_info = game.update(dt=0.016)
```

## Features

- Vectorized NumPy operations for performance
- Configurable plant and zombie stats via CSV
- Lawn mower defensive mechanic
- Special zombie behaviors (pole vault, newspaper shield)
- Extensible for RL training environments
