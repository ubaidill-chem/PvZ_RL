import numpy as np
import pygame
from numpy.typing import NDArray

import assets
from assets import (
    COST_FONT,
    COST_FONT_COLOR,
    GRID_LINE_COLOR,
    GRID_START_X,
    GRID_START_Y,
    HEALTH_TEXTURES,
    IMGS,
    INERT_TEXTURES,
    LAWN_MOWER_X,
    LIGHT_GREEN,
    PBAR_MARGIN,
    PLANT_NAMES,
    PLANT_X_OFFSET,
    PLANT_Y_OFFSET,
    PROG_BAR_H,
    PROG_BAR_W,
    PROG_BAR_X,
    PROG_BAR_Y,
    PROGRESS_BAR,
    SCREEN,
    SEED_DARK_OVERLAY,
    SEED_LIGHT_OVERLAY,
    SEED_START_X,
    SEED_START_Y,
    SHIELD_HEALTH_TEXTURES,
    SHOVEL_OVERLAY,
    SHOVEL_POS,
    STATE_TEXTURES,
    SUN_DISPLAY_POS,
    SUN_FONT,
    SUN_FONT_COLOR,
    SUN_TXT_BOX,
    ZOMB_NAMES,
    ZOMBIE_X_OFFSET,
    ZOMBIE_Y_OFFSET,
)

img_reported_missing = []

def render_misc(sun: int, lvl_prog: float, n_flags: int, lawn_mowers: NDArray[np.int32], is_shovel: bool):
    sun_txt = SUN_FONT.render(str(sun), True, SUN_FONT_COLOR)
    sun_txt_rect = sun_txt.get_rect(center=SUN_DISPLAY_POS)

    to_blit = [
        (IMGS['shovel'], SHOVEL_POS),
        (SUN_TXT_BOX, (31, 10)),
        (IMGS['sun_icon'], (11, -2)),
        (sun_txt, sun_txt_rect)
    ]

    if lvl_prog > 0:
        bar_len = int(lvl_prog * PROG_BAR_W)
        pygame.draw.rect(PROGRESS_BAR, LIGHT_GREEN, (((PBAR_MARGIN + PROG_BAR_W - bar_len), PBAR_MARGIN), (bar_len, PROG_BAR_H)))
        flag_blits = [(PROGRESS_BAR, (PROG_BAR_X, PROG_BAR_Y))]
        for i in range(n_flags):
            flag_x = PROG_BAR_X + int(PROG_BAR_W * i / n_flags)
            flag_y = 3 if (i / n_flags) >= (1 - lvl_prog) else 10
            flag_blits.append((IMGS['f'], (flag_x, flag_y)))
        flag_blits.append((IMGS['bar_marker'], (PROG_BAR_X + PROG_BAR_W - bar_len, PROG_BAR_Y)))
        to_blit.extend(flag_blits)

    if is_shovel:
        to_blit.append((SHOVEL_OVERLAY, SHOVEL_POS))

    x, y = pygame.mouse.get_pos()
    if (GRID_START_X <= x <= GRID_START_X + assets.TILE_W * assets.GRID_COLS) and (GRID_START_Y <= y <= GRID_START_Y + assets.TILE_H * assets.GRID_ROWS):
        lawn_col = (x - GRID_START_X) // assets.TILE_W
        lawn_row = (y - GRID_START_Y) // assets.TILE_H
        to_blit.append((assets.LAWN_OVERLAY_V, (GRID_START_X + assets.TILE_W * lawn_col, GRID_START_Y)))
        to_blit.append((assets.LAWN_OVERLAY_H, (GRID_START_X, GRID_START_Y + assets.TILE_H * lawn_row)))

    lawn_mower_img = IMGS['lawnmower']
    for i in np.where(lawn_mowers > 0)[0]:
        rect = lawn_mower_img.get_rect(bottomleft=(LAWN_MOWER_X, GRID_START_Y + assets.TILE_H * (i + 1)))
        to_blit.append((lawn_mower_img, rect))

    return to_blit


def render_seedbank(seedbank: NDArray[np.void], sun: int, selected_idx: int | None):
    seed_img = IMGS['seedpacket']
    seed_w, seed_h = seed_img.size
    seeds = []
    for i, seed in enumerate(seedbank):
        seed_y = SEED_START_Y + seed_h * i
        seed_pos = (SEED_START_X, seed_y)
        seeds.append((seed_img, seed_pos))

        name = PLANT_NAMES[int(seed['type'])]
        if name not in IMGS:
            if name not in img_reported_missing:
                print(f"Image '{name}' not found")
                img_reported_missing.append(name)
            name = 'unknownp'
        img = IMGS[f"seed_{name}"]

        rect = img.get_rect(center=(SEED_START_X + int(seed_w * 0.36), seed_y + seed_h // 2))
        seeds.append((img, rect))

        cost_txt = COST_FONT.render(str(seed['cost']), False, COST_FONT_COLOR)
        cost_rect = cost_txt.get_rect(bottomright=(SEED_START_X + int(seed_w * 0.92), seed_y + int(seed_h * 0.95)))
        seeds.append((cost_txt, cost_rect))

        recharge_prog = 1.0
        if seed['recharge'] > 0:
            recharge_prog = 1 - float(seed['timer'] / seed['recharge'])

        if recharge_prog < 1:
            overlay_rect = pygame.Rect(0, 0, seed_img.width, int(seed_img.height * (1 - recharge_prog)))
            seeds.append((SEED_DARK_OVERLAY, seed_pos, overlay_rect))
        if recharge_prog < 1 or sun < seed['cost']:
            seeds.append((SEED_DARK_OVERLAY, seed_pos))
        if i is not None and i == selected_idx:
            seeds.append((SEED_LIGHT_OVERLAY, seed_pos))

    return seeds


def render_plants(plant_state: NDArray[np.void], damage_array: NDArray[np.float64]):
    plants = []
    for row, col in np.argwhere(plant_state['type'] > 0):
        p = plant_state[row, col]

        name: str = PLANT_NAMES[int(p['type'])]
        name = STATE_TEXTURES.get(name, {}).get(int(p['special_state']), name)

        health_textures = HEALTH_TEXTURES.get(name, {})
        if health_textures and (ks := [h for h in health_textures if p['health'] < h]):
            name = health_textures[min(ks)]

        if p['timer'] > 0:
            name = INERT_TEXTURES.get(name, name)

        if name not in IMGS:
            if name not in img_reported_missing:
                print(f"Image '{name}' not found")
                img_reported_missing.append(name)
            name = 'unknownp'
        img = IMGS[name]

        mid = GRID_START_X + assets.TILE_W * (col + PLANT_X_OFFSET)
        bottom = GRID_START_Y + assets.TILE_H * (row + PLANT_Y_OFFSET)
        rect = img.get_rect(midbottom=(mid, bottom))
        plants.append((img, rect))
        
        if damage_array[row, col] > 0:
            plants.append((IMGS[f"{name}_hit"], rect))

        if p['timer'] > 0.1:
            continue

        if (sp := p['sun_prod']) > 0:
            sun_type = 'sun_small' if sp == 25 else 'sun' if sp == 50 else 'sun_large'
            plants.append((IMGS[sun_type], rect))

    return plants


def render_zombies(zomb_state: NDArray[np.void], damage_array: NDArray[np.float64]):
    zombies = []
    for row, col in np.argwhere(zomb_state['type'] > 0):
        z = zomb_state[row, col]

        name: str = ZOMB_NAMES[int(z['type'])]
        name = STATE_TEXTURES.get(name, {}).get(int(z['special_state']), name)

        health_texture_pack = HEALTH_TEXTURES if z['shield_health'] <= 0 else SHIELD_HEALTH_TEXTURES
        health = z['shield_health'] or z['health']
        health_textures = health_texture_pack.get(name, {})
        if health_textures and (ks := [h for h in health_textures if health <= h]):
            name = health_textures[min(ks)]

        if name not in IMGS:
            if name not in img_reported_missing:
                print(f"Image '{name}' not found")
                img_reported_missing.append(name)
            name = 'unknownz'
        img = IMGS[name]

        mid = GRID_START_X + assets.TILE_W * (float(z['x']) + ZOMBIE_X_OFFSET)
        bottom = GRID_START_Y + assets.TILE_H * (row + ZOMBIE_Y_OFFSET)
        rect = img.get_rect(midbottom=(mid, bottom))

        zombies.append((img, rect))
        if z['slow_timer'] > 0:
            zombies.append((IMGS[f"{name}_cool"], rect))
        if damage_array[row, col] > 0:
            zombies.append((IMGS[f"{name}_hit"], rect))

    return zombies


def draw_grid():
    for i in range(assets.GRID_ROWS + 1):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR,
                        (GRID_START_X, GRID_START_Y + assets.TILE_H * i),
                        (GRID_START_X + assets.TILE_W * assets.GRID_COLS, GRID_START_Y + assets.TILE_H * i))
    for i in range(assets.GRID_COLS + 1):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR,
                        (GRID_START_X + assets.TILE_W * i, GRID_START_Y),
                        (GRID_START_X + assets.TILE_W * i, GRID_START_Y + assets.TILE_H * assets.GRID_ROWS))
