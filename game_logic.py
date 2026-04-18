from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np

from pools import PLANTS, ZOMBIES, PlantGrid, ZombiePool


EATING_DISTANCE_THRESHOLD = 0.6  # must be closer than 0.6 tiles away to eat
BITE_RATE_MULTIPLIER = 10  # 10 bites takes the same time as moving 1 tile


class Z(IntEnum):
    FLAG_ZOMBIE = 2
    POLE_VAULT = 5
    NEWSPAPER = 6


class Wave(NamedTuple):
    rows: np.ndarray[tuple[int], np.dtype[np.uint32]]
    zombies: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.float32]]
    delay: float


class StepInfo(NamedTuple):
    lvl_outcome: int
    damage_dealt: float
    sun_gained: int
    zombies_killed: int
    plants_lost: int
    lawn_mower_used: int


@dataclass
class LevelConfig:
    n_flags: int
    p_init: np.ndarray[tuple[int], np.dtype[np.float64]]
    p_fin: np.ndarray[tuple[int], np.dtype[np.float64]]
    flag_freq: int = 10
    t_normal: float = 10.0
    t_flag: float = 30.0
    wave_size_init: float = 1.0
    wave_size_ramp: float = 0.5
    flag_multi: float = 2.5
    n_rows: int = 5
    n_cols: int = 9
    lawn_mowers: int = 1
    sun_cooldown: float = 10.0
    sun_value: int = 50

    def __post_init__(self):
        self.n_waves = self.n_flags * self.flag_freq
        self.p_init = np.pad(self.p_init, (0, ZOMBIES.size - self.p_init.size)) / self.p_init.sum()
        self.p_fin = np.pad(self.p_fin, (0, ZOMBIES.size - self.p_fin.size)) / self.p_fin.sum()

    def spawn_roster(self, seed: Optional[int] = None) -> list[Wave]:
        wave_nums = np.arange(self.n_waves)
        is_flag = (wave_nums + 1) % self.flag_freq == 0

        wave_delays = np.where(is_flag, self.t_flag, self.t_normal)
        multiplier = np.where(is_flag, self.flag_multi, 1.0)
        wave_sizes = ((self.wave_size_init + wave_nums * self.wave_size_ramp) * multiplier).astype(np.uint32)

        progress = np.linspace(0.0, 1.0, self.n_waves).reshape(-1, 1)
        probs = progress * self.p_init + (1 - progress) * self.p_fin

        rng = np.random.default_rng(seed)
        roster: list[Wave] = []
        for f, k, p, t in zip(is_flag, wave_sizes, probs, wave_delays):
            if f:
                zombies = np.concatenate(([Z.FLAG_ZOMBIE], rng.choice(ZOMBIES['type'], k-1, p=p)))
            else:
                zombies = rng.choice(ZOMBIES['type'], k, p=p)
            wave = Wave(rng.choice(self.n_rows, k), zombies, rng.uniform(-0.4, 0.4, size=k), t)
            roster.append(wave)
        return roster


class PvZGame:
    def __init__(self, lvlconfig: LevelConfig, init_sun: int = 50, seed_timer_init: float = 10.0):
        self.lvlconfig = lvlconfig
        self.n_rows = self.lvlconfig.n_rows
        self.n_cols = self.lvlconfig.n_cols

        self.sun: int = init_sun
        self.seed_timers_init = np.maximum(PLANTS['seed_recharge'] - seed_timer_init, 0)
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

        self.seed_timers = self.seed_timers_init
        self.sun_timer: float = self.lvlconfig.sun_cooldown

        self.spawn_roster = self.lvlconfig.spawn_roster()
        self.spawn_timer: float = self.lvlconfig.t_flag
        self.upcoming_wave: int = 0

    def update(self, dt: float) -> StepInfo:
        sun_before = self.sun
        zombies_before = (self.z['type'] > 0).sum()
        plants_before  = (self.p['type'] > 0).sum()
        mowers_before  = self.lawn_mowers.sum()

        self.seed_timers = np.maximum(self.seed_timers - dt, 0)
        self.update_sky_sun(dt)
        damage_dealt = self.update_plants(dt)
        is_win = self.update_spawn(dt)
        is_lose = self.update_zombies(dt)
        
        return StepInfo(
            lvl_outcome = 1 if is_win else (-1 if is_lose else 0),
            damage_dealt = damage_dealt,
            sun_gained = self.sun - sun_before,
            zombies_killed = int(zombies_before - (self.z['type'] > 0).sum()),
            plants_lost = int(plants_before  - (self.p['type'] > 0).sum()),
            lawn_mower_used = mowers_before - self.lawn_mowers.sum()
        )

    def update_sky_sun(self, dt: float):
        self.sun_timer -= dt
        if self.sun_timer <= 0:
            self.sun += self.lvlconfig.sun_value
            self.sun_timer += self.lvlconfig.sun_cooldown

    def update_spawn(self, dt: float):
        if self.upcoming_wave == self.lvlconfig.n_waves:
            if not self.z['type'].any():
                print("Win!!!")
                return True
            return False

        self.spawn_timer -= dt
        if self.spawn_timer <= 0 or not self.z['type'].any():
            rows, ztypes, offsets, t = self.spawn_roster[self.upcoming_wave]
            for row, ztype, offset in zip(rows, ztypes, offsets):
                self.zombies.spawn(int(row), int(ztype), float(offset))
            self.spawn_timer = t
            self.upcoming_wave += 1

    def update_zombies(self, dt: float):
        # Anger newspaper
        to_anger = (self.z['type'] == Z.NEWSPAPER) & (self.z['shield_health'] == 0)
        self.z['special_state'] = np.where(to_anger, 1, self.z['special_state'])

        # Update special speeds
        self.z['default_speed'] = np.where(self.z['special_state'] == 0, self.z['default_speed'], self.z['special_speed'])

        # Update slow
        self.z['slow_timer'] = np.maximum(self.z['slow_timer'] - dt, 0)
        slowed = self.z['slow_timer'] > 0
        self.z['speed'] = np.where(slowed, self.z['default_speed'] * 0.5, self.z['default_speed'])

        # Move zombies
        deltax = np.where(self.z['is_moving'], self.z['speed'] * dt, 0.0)
        self.z['x'] -= deltax
        
        trespassed = (self.z['type'] > 0) & (self.z['x'] <= 0)
        if trespassed.any():
            rows = np.unique(np.where(trespassed)[0])
            if (self.lawn_mowers[rows] > 0).all():
                self.lawn_mowers[rows] -= 1
                self.zombies.remove((rows,))
            else:
                print("Game over")
                return True

        # Immobilize zombies
        int_xs = np.clip(self.z['x'].astype(np.uint32), 0, self.n_cols - 1)
        is_facing_plant = (self.z['type'] > 0) & (self.p[self.row_vect, int_xs]['type'] > 0)        
        is_running_pole = (self.z['type'] == Z.POLE_VAULT) & (self.z['special_state'] == 0)
        is_eating = is_facing_plant & ~is_running_pole & (self.z['x'] - int_xs < EATING_DISTANCE_THRESHOLD)
        self.z['is_moving'] = ~is_eating

        # Pole vault jumps
        to_jump = is_facing_plant & is_running_pole
        self.z['x'] -= np.where(to_jump, 1, 0)
        self.z['special_state'] += np.where(to_jump, 1, 0)

        # Damage plants
        damage_from_zomb = self.z['speed'] * dt * self.z['damage'] * BITE_RATE_MULTIPLIER
        damage_to_plants = np.zeros(self.p.shape, dtype=np.float32)
        rows, _ = np.where(is_eating)
        np.add.at(damage_to_plants, (rows, int_xs[is_eating]), damage_from_zomb[is_eating])
        self.plants.get_damage(damage_to_plants)

    def update_plants(self, dt: float) -> float:
        self.p['timer'] -= dt
        acting = self.p['timer'] <= 0
        did_act = np.zeros(self.p.shape, dtype=np.bool_)
        damage_array = np.zeros(self.z.shape, dtype=np.float32)

        # Peashooter attack
        single_hits = acting & (self.p['atk_mode'] == 0)
        for row, pcol in np.argwhere(single_hits):  # TODO: Vectorize
            valid_target = (self.z[row]['type'] > 0) & (self.z[row]['x'] > pcol)
            if valid_target.any():
                ptype = self.p[row, pcol]['type']
                to_hit = np.argmin(np.where(valid_target, self.z[row]['x'], np.inf))
                damage_array[row, to_hit] += PLANTS[ptype - 1]['damage']
                if self.z[row, to_hit]['shield_health'] == 0:
                    self.z[row, to_hit]['slow_timer'] = np.maximum(self.z[row, to_hit]['slow_timer'], PLANTS[ptype - 1]['slow_dur'])
                did_act[row, pcol] = True

        # AoE attack
        aoe_attack = acting & (self.p['atk_mode'] == 1)
        for row, pcol in np.argwhere(aoe_attack):  # TODO: Vectorize
            ptype = self.p[row, pcol]['type']
            aoe_rad = PLANTS[ptype - 1]['aoe_rad']
            zrows = slice(max(row - aoe_rad + 1, 0), min(row + aoe_rad, self.n_rows))
            valid_target = (self.z[zrows]['type'] > 0) & (np.abs(self.z[zrows]['x'] - pcol) < (aoe_rad - 0.5))
            if self.p[row, pcol]['instant'] or valid_target.any():
                damage_array[zrows][valid_target] += PLANTS[ptype - 1]['damage']
                did_act[row, pcol] = True

        # Sun production
        self.sun += np.sum(self.p['sun_prod'][acting])
        did_act |= (self.p['sun_prod'] > 0) & acting

        self.p['timer'] += np.where(did_act, self.p['cooldown'], 0)  # Reset timers
        self.plants.remove(did_act & (self.p['instant'] | self.p['single_use']))  # Remove single-use plants
        self.zombies.get_damage(damage_array)  # Damage zombies
        return damage_array.sum()
        
    def place_plant(self, plant_type: int, row: int, col: int) -> tuple[bool, int]:
        if plant_type <= 0 or plant_type > PLANTS.size:
            print(f"Plant type {plant_type} does not exist")
            return False, 0
        
        if self.sun < PLANTS[plant_type - 1]['cost']:
            print("Not enough sun")
            return False, 0
        
        if self.seed_timers[plant_type - 1] > 0:
            print("Plant not ready")
            return False, 0

        if self.plants.place(row, col, plant_type):
            sun_spent = PLANTS[plant_type - 1]['cost']
            self.sun -= sun_spent
            self.seed_timers[plant_type - 1] = PLANTS[plant_type - 1]['seed_recharge']
            return True, sun_spent
        return False, 0

    def shovel_plant(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            print("Out of bounds")
            return False
        
        if self.p[row, col]['type'] == 0:
            print("No plant in tile")
            return False

        self.plants.remove((row, col))
        return True
