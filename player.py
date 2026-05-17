from pathlib import Path

import numpy as np
import pygame

from game_logic import LevelConfig, PvZGame
from pools import PLANTS, ZOMBIES


ZOMB_NAMES = np.char.decode(ZOMBIES['name'], 'utf-8')
PLANT_NAMES = np.char.decode(PLANTS['name'], 'utf-8')

IMG_SIZES = {'misc': {'lawnmower': (69, 56), 'shovel': (72, -1), 'seedpacket': (50, -1)},
             'zombies': {'conehead': (70, -1), 'polevault': (-1, 114), 'polevault2': (70, 114), 'unknown_z': (70, -1)}}
HEALTH_TEXTURES = {'wallnut': {2667: 'wallnut2', 1333: 'wallnut3'}}
STATE_TEXTURES = {'polevault': {1: 'polevault2'}}


WIDTH = 800
HEIGHT = 600
FPS = 60

TILE_W = 81
TILE_H = 96

GRID_START_X = 35
GRID_START_Y = 85
GRID_ROWS = 5
GRID_COLS = 9

LAWN_POS = pygame.Vector2(-220, 0)
LAWN_MOWER_POS = pygame.Vector2(-28, 125)
SEEDSLOT_POS_X = 48
SHOVEL_POS_X = SEEDSLOT_POS_X + 557
GRID_LINE_COLOR = 'black'

SUN_DISPLAY_X = SEEDSLOT_POS_X + 37
SUN_FONT_SIZE = 20
SUN_FONT_TYPE = 'AgencyFB'
SUN_FONT_COLOR = 'black'

SEED_START_X = SEEDSLOT_POS_X + 80
SEED_START_Y = 8
SEED_W = 52

ZOMBIE_COORD_X_OFFSET = 70
ZOMBIE_COORD_Y_OFFSET = 180
PLANT_COORD_X_OFFSET = 110
PLANT_COORD_Y_OFFSET = 180

COST_FONT_SIZE = 12
COST_FONT_TYPE = 'Consolas'
COST_FONT_COLOR = 'black'


pygame.init()
pygame.font.init()
SUN_FONT = pygame.font.SysFont(SUN_FONT_TYPE, SUN_FONT_SIZE, bold=True)
COST_FONT = pygame.font.SysFont(COST_FONT_TYPE, COST_FONT_SIZE)

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

IMGS: dict[str, pygame.Surface] = {}
for path in Path('assets').rglob('*.*'):
    dir = path.parent.name
    name, ext = path.name.split('.')
    img = pygame.image.load(path).convert_alpha() if ext == 'png' else pygame.image.load(path).convert() 
    if size := IMG_SIZES.get(dir, {}).get(name):
        curr_w, curr_h = img.size
        w = size[0] if size[0] != -1 else round(curr_w / curr_h * size[1])
        h = size[1] if size[1] != -1 else round(curr_h / curr_w * size[0])
        img = pygame.transform.scale(img, (w, h))
        
    if dir == 'plants':
        IMGS[f"seed_{name}"] = pygame.transform.scale_by(img, min(40 / img.width, 35 / img.height))

    IMGS[name] = img


def render_misc(sun: int, lawn_mowers: np.ndarray[tuple[int]]):
    sun_txt = SUN_FONT.render(str(sun), True, SUN_FONT_COLOR)
    sun_rect = sun_txt.get_rect(center=(SUN_DISPLAY_X, 70))
    to_blit = [
        (IMGS['seedslot'], (SEEDSLOT_POS_X, 0)),
        (IMGS['shovel'], (SHOVEL_POS_X, 0)),
        (sun_txt, sun_rect)
    ]
    to_blit.extend([(IMGS['lawnmower'], (LAWN_MOWER_POS.x, LAWN_MOWER_POS.y + TILE_H * i)) for i in np.where(lawn_mowers > 0)[0]])
    return to_blit


def render_seedbank(seedbank: np.ndarray[tuple[int, int]], sun: int):
    seed_img = IMGS['seedpacket']
    dark_overlay = pygame.Surface(seed_img.size, pygame.SRCALPHA)
    dark_overlay.fill((0, 0, 0, 85))

    seeds = []
    for i, seed in enumerate(seedbank):
        seed_pos = (SEED_START_X + SEED_W * i, SEED_START_Y)
        seeds.append((seed_img, seed_pos))

        name = PLANT_NAMES[int(seed['type'])]
        img = IMGS.get(f"seed_{name}", IMGS['seed_unknown_p'])
        rect = img.get_rect(center=(SEED_START_X + 24 + SEED_W * i, 38))
        seeds.append((img, rect))

        cost_txt = COST_FONT.render(str(seed['cost']), False, COST_FONT_COLOR)
        cost_rect = cost_txt.get_rect(midright=(SEED_START_X + 31 + SEED_W * i, 71))
        seeds.append((cost_txt, cost_rect))

        recharge_prog = 1 - float(seed['timer'] / seed['recharge'])
        if recharge_prog < 1 or sun < seed['cost']:
            seeds.append((dark_overlay, seed_pos))

        if recharge_prog < 1:
            overlay_rect = pygame.Rect(0, 0, seed_img.width, round(seed_img.height * (1 - recharge_prog)))
            seeds.append((dark_overlay, seed_pos, overlay_rect))

    return seeds


def render_plants(plant_state: np.ndarray[tuple[int, int]]):
    plants = []
    for row, col in np.argwhere(plant_state['type'] > 0):
        p = plant_state[row, col]
        
        name = PLANT_NAMES[int(p['type'])]
        health_textures = HEALTH_TEXTURES.get(name, {})
        if health_textures and (ks := [h for h in health_textures.keys() if p['health'] < h]):
            name = health_textures[min(ks)]
        
        img = IMGS.get(name, IMGS['unknown_p'])

        mid = PLANT_COORD_X_OFFSET + TILE_W * col
        bottom = PLANT_COORD_Y_OFFSET + TILE_H * row
        rect = img.get_rect(midbottom=(mid, bottom))

        plants.append((img, rect))
    return plants


def render_zombies(zomb_state: np.ndarray[tuple[int, int]]):
    zombies = []
    for row, col in np.argwhere(zomb_state['type'] > 0):
        z = zomb_state[row, col]

        name: str = ZOMB_NAMES[int(z['type'])]
        name = STATE_TEXTURES.get(name, {}).get(int(z['special_state']), name)
        
        health_textures = HEALTH_TEXTURES.get(name, {})
        if health_textures and (ks := [h for h in health_textures.keys() if z['health'] < h]):
            name = health_textures[min(ks)]

        img = IMGS.get(name, IMGS['unknown_p'])
        mid = ZOMBIE_COORD_X_OFFSET + TILE_W * float(z['x'])
        bottom = ZOMBIE_COORD_Y_OFFSET + TILE_H * row
        rect = img.get_rect(midbottom=(mid, bottom))

        zombies.append((img, rect))
    return zombies


def draw_grid():
    for i in range(GRID_ROWS + 1):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR, 
                        (GRID_START_X, GRID_START_Y + TILE_H * i), 
                        (GRID_START_X + TILE_W * GRID_COLS, GRID_START_Y + TILE_H * i))
    for i in range(GRID_COLS + 1):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR, 
                        (GRID_START_X + TILE_W * i, GRID_START_Y), 
                        (GRID_START_X + TILE_W * i, GRID_START_Y + TILE_H * GRID_ROWS))


def play(game_engine: PvZGame):
    dt = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.K_ESCAPE:
                running = False

        game_outcome, *_ = game_engine.update(dt)
        running = running and (game_outcome == 0)

        SCREEN.blit(IMGS['lawn'], LAWN_POS)
        draw_grid()

        to_blit = render_misc(game_engine.sun, game_engine.lawn_mowers)
        to_blit.extend(render_seedbank(game_engine.seed_bank, game_engine.sun))
        to_blit.extend(render_plants(game_engine.p))
        to_blit.extend(render_zombies(game_engine.z))
        SCREEN.blits(to_blit)

        pygame.display.flip()
        dt = CLOCK.tick(FPS) / 1000

    pygame.quit()


if __name__ == '__main__':
    p_fin = np.array([0, 1, 0, 1/2, 1/2, 1/4, 1/2, 1/4, 1/7])
    lvlconfig = LevelConfig(list(range(1, 9)), 5, np.array([0, 1]), p_fin)
    play(PvZGame(lvlconfig))
