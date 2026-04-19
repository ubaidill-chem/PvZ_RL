from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd

plants_df = pd.read_csv('plants.csv')
PLANTS = np.zeros(np.max(plants_df['type'] + 1), dtype=[
        ('name', 'S10'),
        ('type', 'i4'),
        ('health', 'f4'),
        ('cooldown', 'f4'),
        ('damage', 'f4'),
        ('cost', 'i4'),
        ('seed_recharge', 'f4'),
        ('sun_prod', 'i4'),
        ('atk_mode', 'i4'),  # 0 = Single-hit, 1 = Area-of-Effect, -1 = No attack
        ('aoe_rad', 'u4'),
        ('instant', 'b1'),
        ('single_use', 'b1'),
        ('slow_dur', 'f4'),
        ('atk_range', 'i4'),
        ]
    )

for p in plants_df.itertuples(index=False):
    PLANTS[p[1]] = p

zombies_df = pd.read_csv('zombies.csv')
ZOMBIES = np.zeros(np.max(zombies_df['type'] + 1), dtype=[
        ('name', 'S10'),
        ('type', 'i4'),
        ('health', 'f4'),
        ('shield_health', 'f4'),
        ('speed', 'f4'),
        ('damage', 'f4'),
        ('special_speed', 'f4'),
        ]
    )

for z in zombies_df.itertuples(index=False):
    ZOMBIES[z[1]] = z

class PlantGrid:
    def __init__(self, rows=5, cols=9) -> None:
        self.nrows = rows
        self.ncols = cols
        self.state= np.zeros((rows, cols),
            dtype=[
                ('type', 'i4'),
                ('health', 'f4'),
                ('cooldown', 'f4'),
                ('sun_prod', 'i4'),
                ('atk_mode', 'i4'),
                ('instant', 'b1'),
                ('single_use', 'b1'),
                ('timer', 'f4')
            ]
        )
    
    def place(self, row: int, col: int, ptype: int, init_cooldown_discount: float = 0.3):
        if ptype <= 0 or ptype > PLANTS.size:
            print(f"Plant type {ptype} does not exist")
            return False

        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            print("Out of bounds")
            return False

        if self.state['type'][row, col] > 0:
            print("Tile occupied")
            return False
        
        pstate = PLANTS[ptype]
        self.state[row, col] = (
            ptype,
            pstate['health'],
            pstate['cooldown'],
            pstate['sun_prod'],
            pstate['atk_mode'],
            pstate['instant'],
            pstate['single_use'],
            pstate['cooldown'] * init_cooldown_discount  # discounted cooldown first time
        )
        return True

    def get_damage(self, damage_array: npt.NDArray[np.float32]):
        if not damage_array.any():
            return
        self.state['health'] -= damage_array
        self.remove(self.state['health'] <= 0)

    def remove(self, mask: Union[npt.NDArray[np.bool_], tuple[int, int], tuple[npt.NDArray, npt.NDArray]]):
        if isinstance(mask, np.ndarray) and not mask.any():
            return
        self.state[mask] = 0

        
class ZombiePool:
    def __init__(self, rows=5, cols=9, max_zombies_per_row=20) -> None:
        self.nrows = rows
        self.ncols = cols
        self.state = np.zeros((rows, max_zombies_per_row),
            dtype=[
                ('x', 'f4'),
                ('type', 'i4'),
                ('health', 'f4'),
                ('shield_health', 'f4'),
                ('speed', 'f4'),
                ('default_speed', 'f4'),
                ('damage', 'f4'),
                ('is_moving', 'b1'),
                ('slow_timer', 'f4'),
                ('special_state', 'i4'),
                ('special_speed', 'f4')
            ]
        )

    def spawn(self, row: int, ztype: int, x_pos: float = 0.0):
        if ztype <= 0 or ztype > len(ZOMBIES):
            print(f"Zombie type {ztype} does not exist")
            return False

        if row < 0 or row >= self.nrows:
            print("Out of bounds")
            return False
    
        empty = np.where(self.state['type'][row] == 0)[0]
        if empty.size == 0:
            print(f"Max zombie in row ({self.state.shape[1]}) reached")
            return False

        zstate = ZOMBIES[ztype]
        self.state[row, empty[0]] = (
            self.ncols + 1 + x_pos,
            ztype,
            zstate['health'],
            zstate['shield_health'],
            zstate['speed'],
            zstate['speed'],
            zstate['damage'],
            True,
            0.0,
            0,
            zstate['special_speed'],
        )
        return True

    def get_damage(self, damage_array: npt.NDArray[np.float32]):
        if not damage_array.any():
            return
        self.state['health'] -= np.where(self.state['shield_health'] > 0, 0, damage_array)
        self.state['shield_health'] = np.maximum(self.state['shield_health'] - damage_array, 0)
        self.remove(self.state['health'] <= 0)

    def remove(self, mask: Union[npt.NDArray[np.bool_], tuple[int, ...], tuple[npt.NDArray, ...]]):
        if isinstance(mask, np.ndarray) and not mask.any():
            return
        self.state[mask] = 0
