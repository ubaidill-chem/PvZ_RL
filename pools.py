from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd

PLANTS = np.array(pd.read_csv('plants.csv').iloc[:, 1:].to_records(index=False), dtype=[
        ('type', 'u4'),
        ('health', 'f4'),
        ('cooldown', 'f4'),
        ('damage', 'f4'),
        ('sun_prod', 'u4'),
        ('cost', 'u4'),
        ('seed_recharge', 'f4'),
        ('slow_dur', 'f4')
        ]
    )

ZOMBIES = np.array(pd.read_csv('zombies.csv').iloc[:, 1:].to_records(index=False), dtype=[
        ('type', 'u4'),
        ('health', 'f4'),
        ('shield_health', 'f4'),
        ('speed', 'f4'),
        ('damage', 'f4'),
        ('speed_special', 'f4'),
        ]
    )

class PlantGrid:
    def __init__(self, rows=5, cols=9) -> None:
        self.nrows = rows
        self.ncols = cols
        self.state= np.zeros((rows, cols),
            dtype=[
                ('type', 'u4'),
                ('health', 'f4'),
                ('cooldown', 'f4'),
                ('damage', 'f4'),
                ('sun_prod', 'u4'),
                ('slow_dur', 'f4'),
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
        
        pstate = PLANTS[ptype - 1]
        self.state[row, col] = (
            ptype,
            pstate['health'],
            pstate['cooldown'],
            pstate['damage'],
            pstate['sun_prod'],
            pstate['slow_dur'],
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
                ('type', 'u4'),
                ('health', 'f4'),
                ('shield_health', 'f4'),
                ('speed', 'f4'),
                ('default_speed', 'f4'),
                ('damage', 'f4'),
                ('is_moving', 'b1'),
                ('slow_timer', 'f4'),
                ('special_state', 'u4'),
                ('speed_special', 'f4')
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

        zstate = ZOMBIES[ztype - 1]
        self.state[row, empty[0]] = (
            self.ncols + x_pos,
            ztype,
            zstate['health'],
            zstate['shield_health'],
            zstate['speed'],
            zstate['speed'],
            zstate['damage'],
            True,
            0.0,
            0,
            zstate['speed_special'],
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
