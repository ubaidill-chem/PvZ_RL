from dataclasses import dataclass, field
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
SPAWN_OFFSET_MIN = 0.4  # spawn position random offset range (+)
SPAWN_OFFSET_MAX = 1  # spawn position random offset range (+)


class Z(IntEnum):
    FLAG_ZOMBIE = 2
    POLE_VAULT = 5
    NEWSPAPER = 6
    DISCO = 9
    BACKUP = 10


class P(IntEnum):
    PUFFSHROOM = 9
    SUNSHROOM = 10
    SCAREDY = 13


class Wave(NamedTuple):
    zombies: np.ndarray[tuple[int], np.dtype[np.uint32]]
    rows: np.ndarray[tuple[int], np.dtype[np.int64]]
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
    wave_delay: float = 25.0
    wave_size_init: float = 0.6
    wave_size_ramp: float = 0.4
    flag_multi: float = 2.5
    n_rows: int = 5
    n_cols: int = 9
    lawn_mowers: int = 1
    init_sun: int = 50
    sun_cooldown: float | Literal['night'] = 10.0
    sun_value: int = 50
    preplaced_plants: list[tuple[int, int, int]] = field(default_factory=list)

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

            wave = Wave(zombies, rng.choice(self.n_rows, k), rng.uniform(SPAWN_OFFSET_MIN, SPAWN_OFFSET_MAX, size=k))
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
    def __init__(self, lvlconfig: LevelConfig, seed: Optional[int] = None):
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
        self.reset(seed)

    def reset(self, seed):
        self.p[:] = 0
        self.z[:] = 0

        self.seed_bank['timer'] = self.seed_timers_init
        if self.lvlconfig.sun_cooldown != 'night':
            self.sun_timer: float = self.lvlconfig.sun_cooldown
        self.selected_plant_idx = None

        self.spawn_roster = self.lvlconfig.spawn_roster(seed)
        self.spawn_timer: float = self.lvlconfig.wave_delay
        self.upcoming_wave: int = 0

        for ptype, row, col in self.lvlconfig.preplaced_plants:
            self.plants.place(ptype, row, col)

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
            for ztype, row, offset in zip(*self.spawn_roster[self.upcoming_wave]):
                self.zombies.spawn(int(ztype), int(row), self.n_cols - 0.5 + float(offset))
            self.spawn_timer = self.lvlconfig.wave_delay
            self.upcoming_wave += 1

    def update_zombies(self, dt: float) -> tuple[bool, np.ndarray[tuple[int, int]]]:
        # Update timer
        self.z['special_timer'] -= dt
        updating = self.z['special_timer'] <= 0
        self.update_disco(updating)
        self.z['special_timer'] += np.where(updating, self.z['cooldown'], 0)

        # Anger newspaper
        to_anger = (self.z['type'] == Z.NEWSPAPER) & (self.z['shield_health'] <= 0)
        self.z['special_state'] = np.where(to_anger, 1, self.z['special_state'])

        # Update special speeds
        self.z['speed'] = np.where(self.z['special_state'] == 0, self.z['speed'], self.z['special_speed'])

        # Update slow and freeze
        self.z['slow_timer'] -= np.where(self.z['slow_timer'] > 0, dt, 0)
        slowed = self.z['slow_timer'] > 0
        self.z['speed_mult'] = np.where(slowed, SLOW_SPEED_MULTIPLIER, 1.0)

        # Update freeze
        self.z['freeze_timer'] -= np.where(self.z['freeze_timer'] > 0, dt, 0)
        freezed = self.z['freeze_timer'] > 0
        self.z['speed_mult'] = np.where(freezed, 0.0, self.z['speed_mult'])

        # Move zombies
        real_speed = self.z['speed'] * self.z['speed_mult']
        deltax = np.where(self.z['is_moving'], real_speed * dt, 0.0)
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
        damage_from_zomb = real_speed * dt * self.z['damage'] * BITE_RATE_MULTIPLIER
        damage_to_plants = np.zeros(self.p.shape, dtype=np.float32)
        rows, _ = np.where(is_eating)
        np.add.at(damage_to_plants, (rows, int_xs[is_eating]), damage_from_zomb[is_eating])
        self.plants.get_damage(damage_to_plants)
        return False, damage_to_plants

    def update_plants(self, dt: float) -> np.ndarray:
        # First timer
        self.p['timer'] -= np.where(self.p['timer'] > 0, dt, 0)        
        acting = (self.p['timer'] <= 0) & (self.p['cooldown'] > 0)
        
        did_act = np.zeros(self.p.shape, dtype=np.bool_)
        dmg_arr = np.zeros(self.z.shape, dtype=np.float32)
        shield_dmg_arr = np.zeros(self.z.shape, dtype=np.float32)

        self.update_single_hitters(acting, dmg_arr, shield_dmg_arr, did_act)
        self.update_aoe_atk(acting, dmg_arr, shield_dmg_arr, did_act)

        # Sun production
        self.sun += np.sum(self.p['sun_prod'][acting])
        did_act |= (self.p['sun_prod'] > 0) & acting

        self.p['timer'] += np.where(did_act, self.p['cooldown'], 0)
        self.plants.remove(did_act & self.p['instant'] > 0)  # Remove single-use plants
        self.zombies.get_damage(dmg_arr)  # Damage zombies
        self.zombies.get_shield_damage(shield_dmg_arr)  # Damage zombies

        # Second timer
        self.p['timer2'] -= np.where(self.p['timer2'] > 0, dt, 0)
        acting2 = self.p['timer2'] <= 0
        did_act2 = np.zeros(self.p.shape, dtype=np.bool_)

        self.update_puff_shroom(acting2, did_act2)
        self.update_sun_shroom(acting2, did_act2)

        self.p['timer2'] += np.where(did_act2, self.p['cooldown2'], 0)

        return dmg_arr + shield_dmg_arr
    
    def update_single_hitters(self, acting: np.ndarray, dmg_arr: np.ndarray, shield_dmg_arr: np.ndarray, did_act: np.ndarray):
        single_hitters = acting & (self.p['atk_mode'] == 0)
        for row, pcol in np.argwhere(single_hitters):  # TODO: Vectorize
            ptype = self.p[row, pcol]['type']
            if self.is_scared_scaredy(ptype, row, pcol):
                continue
            
            range_front = self.get_range(ptype, 'range_front')
            range_back = self.get_range(ptype, 'range_back')
            dist = self.z[row]['x'] - pcol
            valid_target = (self.z[row]['type'] > 0) & (dist < range_front - 0.5) & (-dist < range_back - 0.5)

            if valid_target.any():
                to_hit = np.argmin(np.where(valid_target, self.z[row]['x'], np.inf))
                if self.z[row, to_hit]['shield_health'] == 0:
                    dmg_arr[row, to_hit] += PLANTS[ptype]['damage']
                    self.z[row, to_hit]['slow_timer'] = max(self.z[row, to_hit]['slow_timer'], PLANTS[ptype]['slow_dur'])
                    self.z[row, to_hit]['freeze_timer'] = max(self.z[row, to_hit]['freeze_timer'], PLANTS[ptype]['freeze_dur'])
                else:
                    shield_dmg_arr[row, to_hit] += PLANTS[ptype]['damage']
                did_act[row, pcol] = True

    def update_aoe_atk(self, acting: np.ndarray, dmg_arr: np.ndarray, shield_dmg_arr: np.ndarray, did_act: np.ndarray):
        aoe_attack = acting & (self.p['atk_mode'] == 1)
        for row, pcol in np.argwhere(aoe_attack):  # TODO: Vectorize
            ptype = self.p[row, pcol]['type']
            range_front = self.get_range(ptype, 'range_front')
            range_back = self.get_range(ptype, 'range_back')
            range_side: int = self.get_range(ptype, 'range_side') # type: ignore

            zrows = self.slice_rows(row, range_side, range_side)
            valid_target = self.zombies_at(-1, zrows, (pcol - range_back + 0.5), (pcol + range_front - 0.5))
            tr, tc = np.where(valid_target)
            if tr.size:
                z_start = max(row - range_side, 0)
                np.add.at(dmg_arr, (tr + z_start, tc), PLANTS[ptype]['damage'])
                np.add.at(shield_dmg_arr, (tr + z_start, tc), PLANTS[ptype]['damage'])

            np.maximum(PLANTS[ptype]['slow_dur'], self.z[zrows]['slow_timer'], out=self.z[zrows]['slow_timer'], where=valid_target)
            np.maximum(PLANTS[ptype]['freeze_dur'], self.z[zrows]['freeze_timer'], out=self.z[zrows]['freeze_timer'], where=valid_target)

            if tr.size or self.p[row, pcol]['instant'] == 2:
                did_act[row, pcol] = True

    def update_puff_shroom(self, acting2: np.ndarray, did_act2: np.ndarray):
        dying_puff = (self.p['type'] == P.PUFFSHROOM) & acting2
        self.p['special_state'] += np.where(dying_puff, 1, 0)
        self.plants.remove(dying_puff & (self.p['special_state'] >= 3))
        did_act2 |= dying_puff & (self.p['special_state'] < 3)

    def update_sun_shroom(self, acting2: np.ndarray, did_act2: np.ndarray):
        grow_sun = (self.p['type'] == P.SUNSHROOM) & acting2 & (self.p['special_state'] < 2)
        self.p['special_state'] += np.where(grow_sun, 1, 0)
        self.p['sun_prod'] += np.where(grow_sun, 25, 0)
        self.p['cooldown'] = np.where(grow_sun & (self.p['special_state'] == 2), 34, self.p['cooldown'])
        self.p['cooldown2'] = np.where(grow_sun & (self.p['special_state'] == 1), 36, self.p['cooldown2'])
        did_act2 |= grow_sun

    def is_scared_scaredy(self, ptype, row: int, pcol: int):
        if ptype == P.SCAREDY:
            zrows = self.slice_rows(row, 1, 1)
            is_zombie_nearby = np.any((self.z[zrows]['type'] > 0) & (np.abs(self.z[zrows]['x'] - pcol) < 1.5))
            self.p[row, pcol]['special_state'] = int(is_zombie_nearby)
            return is_zombie_nearby

    def update_disco(self, updating: np.ndarray):
        disco_spawn = (self.z['type'] == Z.DISCO) & updating
        self.z['special_state'] += np.where(disco_spawn, 1, 0)
        for row, col in np.argwhere(disco_spawn):  # TODO: Vectorize
            x = float(self.z[row, col]['x'])
            
            is_front_backup = np.any(self.zombies_at(Z.BACKUP, row, x))
            if not is_front_backup:
                self.zombies.spawn(Z.BACKUP, row, x-1)

            is_back_backup = np.any(self.zombies_at(Z.BACKUP, row, x))
            if not is_back_backup:
                self.zombies.spawn(Z.BACKUP, row, x+1)

            is_top_backup = np.any(self.zombies_at(Z.BACKUP, row-1, x)) if row > 0 else True
            if not is_top_backup:
                self.zombies.spawn(Z.BACKUP, row-1, x)

            is_bottom_backup = np.any(self.zombies_at(Z.BACKUP, row+1, x)) if row < self.n_rows - 1 else True
            if not is_bottom_backup:
                self.zombies.spawn(Z.BACKUP, row+1, x)

    def slice_rows(self, row: int, up: int, down: int):
        return slice(max(row - up, 0), min(row + down + 1, self.n_rows))

    def get_range(self, ptype, range_name: str):
        rang = int(PLANTS[ptype][range_name])
        return (self.n_rows if range_name.endswith('side') else np.inf) if rang == -1 else rang

    def zombies_at(self, ztype: int, row: int | slice, col_start: float, col_end: Optional[float] = None):
        valid_type: np.ndarray = (self.z[row]['type'] == ztype) if ztype >= 0 else (self.z[row]['type'] > 0)
        if col_end is None:
            col_start -= 0.5
            col_end = col_start + 0.5
        return valid_type & (col_start <= self.z[row]['x']) & (self.z[row]['x'] <= col_end)
        
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
        self.deselect_plant()
        if self.plants.place(ptype, row, col):
            sun_spent = self.seed_bank[idx]['cost']
            self.sun -= sun_spent
            self.seed_bank[idx]['timer'] = self.seed_bank[idx]['recharge']
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
