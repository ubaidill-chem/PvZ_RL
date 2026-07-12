from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, NamedTuple, Optional

import numpy as np

from pools import PLANTS, ZOMBIES, PlantGrid, ZombiePool


# Game Mechanics Constants
EATING_DISTANCE_THRESHOLD = 0.5  # must be closer than 0.5 tiles away to eat
BITE_RATE_MULTIPLIER = 10  # 10 bites takes the same time as moving 1 tile
SLOW_SPEED_MULTIPLIER = 0.5  # slowed zombies move at 50% speed
POLE_VAULT_JUMP_DISTANCE = 1.0  # distance pole vault zombies jump per action

# Zombie Movement Constants
MAX_SUN_STORAGE = 9900  # cap on sun that can be stored
SPAWN_OFFSET_RANGE = 0.2  # spawn position random offset range (+)


class Z(IntEnum):
    FLAG_ZOMBIE = 2
    POLE_VAULT = 5
    NEWSPAPER = 6


class P(IntEnum):
    PUFFSHROOM = 9
    SUNSHROOM = 10


class Wave(NamedTuple):
    rows: np.ndarray[tuple[int], np.dtype[np.int64]]
    zombies: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.float64]]


class StepInfo(NamedTuple):
    lvl_outcome: int
    zombie_dmg_arr: np.ndarray[tuple[int, int]]
    plant_dmg_arr: np.ndarray[tuple[int, int]]
    sun_gained: int
    zombies_killed: int
    plants_lost: int
    lawn_mower_used: int


@dataclass
class LevelConfig:
    plants: list[int]
    n_flags: int
    p_init: np.ndarray[tuple[int], np.dtype[np.float64]]
    p_fin: np.ndarray[tuple[int], np.dtype[np.float64]]
    flag_freq: int = 8
    wave_delay: float = 30.0
    wave_size_init: float = 0.6
    wave_size_ramp: float = 0.4
    flag_multi: float = 2.5
    n_rows: int = 5
    n_cols: int = 9
    lawn_mowers: int = 1
    init_sun: int = 50
    sun_cooldown: float | Literal['night'] = 10.0
    sun_value: int = 50

    def __post_init__(self):
        self.n_waves = self.n_flags * self.flag_freq
        self.p_init = np.pad(self.p_init, (0, ZOMBIES.size - self.p_init.size))
        self.p_fin = np.pad(self.p_fin, (0, ZOMBIES.size - self.p_fin.size))

        init_sum = float(self.p_init.sum())
        fin_sum = float(self.p_fin.sum())
        if init_sum == 0.0 or fin_sum == 0.0:
            raise ValueError("p_init and p_fin must have a non-zero sum")
        
        self.p_init = self.p_init / init_sum
        self.p_fin = self.p_fin / fin_sum


    def spawn_roster(self, seed: Optional[int] = None) -> list[Wave]:
        wave_nums = np.arange(self.n_waves)
        is_flag = (wave_nums + 1) % self.flag_freq == 0

        multiplier = np.where(is_flag, self.flag_multi, 1.0)
        raw_sizes = (self.wave_size_init + wave_nums * self.wave_size_ramp) * multiplier
        wave_sizes = np.maximum(raw_sizes.round(), 1)

        progress = np.linspace(0.0, 1.0, self.n_waves).reshape(-1, 1)
        probs = (1 - progress) * self.p_init + progress * self.p_fin

        rng = np.random.default_rng(seed)
        roster: list[Wave] = []
        for f, k, p in zip(is_flag, wave_sizes, probs):
            k = int(k)
            if f:
                zombies = np.concatenate(([Z.FLAG_ZOMBIE], rng.choice(ZOMBIES['type'], max(k-1, 1), p=p)))
            else:
                zombies = rng.choice(ZOMBIES['type'], k, p=p)
            wave = Wave(rng.choice(self.n_rows, k), zombies, rng.uniform(0, SPAWN_OFFSET_RANGE, size=k))
            roster.append(wave)
        return roster


def generate_seed_bank(p_types: list[int]):
    if len(p_types) > 9:
        raise ValueError("Seed bank can only hold up to 9 plants")
        
    seed_bank = np.array([(
        p, 
        PLANTS[p]['cost'], 
        PLANTS[p]['seed_recharge'], 
        PLANTS[p]['seed_recharge'] if PLANTS[p]['seed_recharge'] > 10 else 0
        ) for p in p_types],
        dtype=[
            ('type', 'f4'),
            ('cost', 'i4'),
            ('recharge', 'f4'),
            ('timer', 'f4')
        ]
    )
    seed_bank['timer'] = np.maximum(seed_bank['timer'], 0)
    return seed_bank
        

class PvZGame:
    def __init__(self, lvlconfig: LevelConfig):
        self.lvlconfig = lvlconfig
        self.n_rows = self.lvlconfig.n_rows
        self.n_cols = self.lvlconfig.n_cols

        self.sun: int = self.lvlconfig.init_sun
        self.seed_bank = generate_seed_bank(self.lvlconfig.plants)
        self.seed_timers_init = self.seed_bank['timer'].copy()

        self.plants = PlantGrid(self.n_rows, self.n_cols)
        self.zombies = ZombiePool(self.n_rows, self.n_cols)
        self.lawn_mowers = np.full(self.n_rows, self.lvlconfig.lawn_mowers)

        self.p = self.plants.state
        self.z = self.zombies.state
        self.row_vect = np.arange(self.n_rows).reshape(-1, 1)
        self.reset()

    def reset(self):
        self.p[:] = 0
        self.z[:] = 0

        self.seed_bank['timer'] = self.seed_timers_init
        if self.lvlconfig.sun_cooldown != 'night':
            self.sun_timer: float = self.lvlconfig.sun_cooldown
        self.selected_plant_idx = None

        self.spawn_roster = self.lvlconfig.spawn_roster()
        self.spawn_timer: float = self.lvlconfig.wave_delay
        self.upcoming_wave: int = 0

    def update(self, dt: float) -> StepInfo:
        sun_before = self.sun
        zombies_before = (self.z['type'] > 0).sum()
        plants_before  = (self.p['type'] > 0).sum()
        mowers_before  = self.lawn_mowers.sum()

        self.seed_bank['timer'] = np.maximum(self.seed_bank['timer'] - dt, 0)
        self.update_sky_sun(dt)
        z_dmg_arr = self.update_plants(dt)
        is_win = self.update_spawn(dt)
        is_lose, p_dmg_arr = self.update_zombies(dt)
        
        return StepInfo(
            lvl_outcome = 1 if is_win else (-1 if is_lose else 0),  # TODO: Replace with Enum
            zombie_dmg_arr = z_dmg_arr,
            plant_dmg_arr = p_dmg_arr,
            sun_gained = self.sun - sun_before,
            zombies_killed = int(zombies_before - (self.z['type'] > 0).sum()),
            plants_lost = int(plants_before  - (self.p['type'] > 0).sum()),
            lawn_mower_used = mowers_before - self.lawn_mowers.sum()
        )

    def update_sky_sun(self, dt: float):
        if self.lvlconfig.sun_cooldown == 'night':
            return
        
        self.sun_timer -= dt
        if self.sun_timer <= 0:
            self.sun += self.lvlconfig.sun_value
            self.sun_timer += self.lvlconfig.sun_cooldown
        self.sun = min(self.sun, MAX_SUN_STORAGE)

    def update_spawn(self, dt: float):
        self.prog = self.upcoming_wave / self.lvlconfig.n_waves
        if self.upcoming_wave == self.lvlconfig.n_waves:
            if not self.z['type'].any():
                print("Win!!!")
                return True
            return False

        self.spawn_timer -= dt
        if self.spawn_timer <= 0 or (not self.z['type'].any() and self.upcoming_wave > 0):
            for row, ztype, offset in zip(*self.spawn_roster[self.upcoming_wave]):
                self.zombies.spawn(int(row), int(ztype), float(offset))
            self.spawn_timer = self.lvlconfig.wave_delay
            self.upcoming_wave += 1

    def update_zombies(self, dt: float) -> tuple[bool, np.ndarray[tuple[int, int]]]:
        # Anger newspaper
        to_anger = (self.z['type'] == Z.NEWSPAPER) & (self.z['shield_health'] <= 0)
        self.z['special_state'] = np.where(to_anger, 1, self.z['special_state'])

        # Update special speeds
        self.z['default_speed'] = np.where(self.z['special_state'] == 0, self.z['default_speed'], self.z['special_speed'])

        # Update slow
        self.z['slow_timer'] = np.maximum(self.z['slow_timer'] - dt, 0)
        slowed = self.z['slow_timer'] > 0
        self.z['speed'] = np.where(slowed, self.z['default_speed'] * SLOW_SPEED_MULTIPLIER, self.z['default_speed'])

        # Move zombies
        deltax = np.where(self.z['is_moving'], self.z['speed'] * dt, 0.0)
        self.z['x'] -= deltax
        
        trespassed = (self.z['type'] > 0) & (self.z['x'] <= -0.5)
        if trespassed.any():
            rows = np.unique(np.where(trespassed)[0])
            for r in rows:
                if self.lawn_mowers[r] > 0:
                    self.lawn_mowers[r] -= 1
                    self.zombies.remove((np.array([r]),))
                else:
                    print("Game over")
                    return True, np.zeros(self.p.shape, dtype=np.float32)

        # Immobilize zombies
        int_xs = np.clip(np.floor(self.z['x']), 0, self.n_cols - 1).astype(int)
        is_facing_plant = (self.z['type'] > 0) & (self.p[self.row_vect, int_xs]['type'] > 0) & (self.z['x'] - int_xs < EATING_DISTANCE_THRESHOLD)        
        is_running_pole = (self.z['type'] == Z.POLE_VAULT) & (self.z['special_state'] == 0)
        is_eating = is_facing_plant & ~is_running_pole
        self.z['is_moving'] = ~is_eating

        # Pole vault jumps
        to_jump = is_facing_plant & is_running_pole
        self.z['x'] -= np.where(to_jump, POLE_VAULT_JUMP_DISTANCE, 0)
        self.z['special_state'] += np.where(to_jump, 1, 0)

        # Damage plants
        damage_from_zomb = self.z['speed'] * dt * self.z['damage'] * BITE_RATE_MULTIPLIER
        damage_to_plants = np.zeros(self.p.shape, dtype=np.float32)
        rows, _ = np.where(is_eating)
        np.add.at(damage_to_plants, (rows, int_xs[is_eating]), damage_from_zomb[is_eating])
        self.plants.get_damage(damage_to_plants)
        return False, damage_to_plants

    def update_plants(self, dt: float) -> np.ndarray:
        self.p['timer'] -= np.where(self.p['timer'] > 0, dt, 0)        
        acting = self.p['timer'] <= 0
        did_act = np.zeros(self.p.shape, dtype=np.bool_)
        dmg_arr = np.zeros(self.z.shape, dtype=np.float32)
        shield_dmg_arr = np.zeros(self.z.shape, dtype=np.float32)

        self.update_single_hitters(acting, dmg_arr, shield_dmg_arr, did_act)
        self.update_aoe_atk(acting, dmg_arr, shield_dmg_arr, did_act)

        # Sun production
        self.sun += np.sum(self.p['sun_prod'][acting])
        did_act |= (self.p['sun_prod'] > 0) & acting

        self.p['timer'] += np.where(did_act, self.p['cooldown'], 0)
        self.plants.remove(did_act & (self.p['instant'] | self.p['single_use']))  # Remove single-use plants
        self.zombies.get_damage(dmg_arr)  # Damage zombies
        self.zombies.get_shield_damage(shield_dmg_arr)  # Damage zombies

        # Second timer
        self.p['timer2'] -= np.where(self.p['timer2'] > 0, dt, 0)
        did_act2 = np.zeros(self.p.shape, dtype=np.bool_)

        # Puff-shroom
        dying_puff = (self.p['type'] == P.PUFFSHROOM) & (self.p['timer2'] <= 0)
        self.p['special_state'] += np.where(dying_puff, 1, 0)
        self.plants.remove(dying_puff & (self.p['special_state'] >= 3))
        did_act2 |= dying_puff & (self.p['special_state'] < 3)

        # Sun-shroom
        grow_sun = (self.p['type'] == P.SUNSHROOM) & (self.p['timer2'] <= 0) & (self.p['special_state'] < 2)
        self.p['special_state'] += np.where(grow_sun, 1, 0)
        self.p['sun_prod'] += np.where(grow_sun, 25, 0)
        self.p['cooldown'] = np.where(grow_sun & (self.p['special_state'] == 2), 34, self.p['cooldown'])
        self.p['cooldown2'] = np.where(grow_sun & (self.p['special_state'] == 1), 36, self.p['cooldown2'])
        did_act2 |= grow_sun

        self.p['timer2'] += np.where(did_act2, self.p['cooldown2'], 0)

        return dmg_arr + shield_dmg_arr
    
    def update_single_hitters(self, acting: np.ndarray, dmg_arr: np.ndarray, shield_dmg_arr: np.ndarray, did_act: np.ndarray):
        single_hitters = acting & (self.p['atk_mode'] == 0)
        for row, pcol in np.argwhere(single_hitters):  # TODO: Vectorize
            ptype = self.p[row, pcol]['type']
            atk_limit = PLANTS[ptype]['atk_range'] or np.inf
            dist = self.z[row]['x'] - pcol
            valid_target = (self.z[row]['type'] > 0) & (dist > 0) & (dist < atk_limit + 0.5)
            if valid_target.any():
                to_hit = np.argmin(np.where(valid_target, self.z[row]['x'], np.inf))
                if self.z[row, to_hit]['shield_health'] == 0:
                    self.z[row, to_hit]['slow_timer'] = max(self.z[row, to_hit]['slow_timer'], PLANTS[ptype]['slow_dur'])
                    dmg_arr[row, to_hit] += PLANTS[ptype]['damage']
                else:
                    shield_dmg_arr[row, to_hit] += PLANTS[ptype]['damage']
                did_act[row, pcol] = True

    def update_aoe_atk(self, acting: np.ndarray, dmg_arr: np.ndarray, shield_dmg_arr: np.ndarray, did_act: np.ndarray):
        aoe_attack = acting & (self.p['atk_mode'] == 1)
        for row, pcol in np.argwhere(aoe_attack):  # TODO: Vectorize
            ptype = self.p[row, pcol]['type']
            aoe_rad = int(PLANTS[ptype]['aoe_rad'])
            z_start = max(row - aoe_rad, 0)
            z_stop = min(row + aoe_rad + 1, self.n_rows)
            zrows = slice(z_start, z_stop)
            valid_target = (self.z[zrows]['type'] > 0) & (np.abs(self.z[zrows]['x'] - pcol) < (aoe_rad + 0.5))
            tr, tc = np.where(valid_target)
            
            if tr.size:
                np.add.at(dmg_arr, (tr + z_start, tc), PLANTS[ptype]['damage'])
                np.add.at(shield_dmg_arr, (tr + z_start, tc), PLANTS[ptype]['damage'])
            if tr.size or self.p[row, pcol]['instant']:
                did_act[row, pcol] = True
        
    def select_plant(self, idx: int) -> bool:
        self.deselect_plant()
        if self.sun < self.seed_bank[idx]['cost']:
            print("Not enough sun")
            return False
        
        if self.seed_bank[idx]['timer'] > 0:
            print("Plant not ready")
            return False
        
        self.selected_plant_idx = idx
        return True

    def place_plant(self, row: int, col: int) -> tuple[bool, int]:
        idx = self.selected_plant_idx
        if idx is None:
            return False, 0
        ptype = int(self.seed_bank[idx]['type'])
        if self.plants.place(row, col, ptype):
            sun_spent = self.seed_bank[idx]['cost']
            self.sun -= sun_spent
            self.seed_bank[idx]['timer'] = self.seed_bank[idx]['recharge']
            self.deselect_plant()
            return True, sun_spent
        return False, 0
    
    def select_and_place(self, idx: int, row: int, col: int) -> tuple[bool, int]:
        if self.select_plant(idx):
            return self.place_plant(row, col)
        return False, 0

    def deselect_plant(self):
        self.selected_plant_idx = None

    def shovel_plant(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            print("Out of bounds")
            return False
        
        if self.p[row, col]['type'] == 0:
            print("No plant in tile")
            return False

        self.plants.remove((row, col))
        return True
