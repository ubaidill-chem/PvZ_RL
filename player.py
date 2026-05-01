from pathlib import Path

import numpy as np
import pygame

from game_logic import LevelConfig, PvZGame
from pools import PLANTS, ZOMBIES


ZOMB_NAMES = np.char.decode(ZOMBIES['name'], 'utf-8')
PLANT_NAMES = np.char.decode(PLANTS['name'], 'utf-8')

IMG_SIZES = {'misc': {'lawnmower': (69, 56), 'shovel': (72, 72)},
             'zombies': {'conehead': (70, 134), 'polevault': (133, 114), 'polevault2': (70, 114), 'unknown_z': (70, 112)}}
HEALTH_TEXTURES = {'wallnut': {1333: 'wallnut2', 2667: 'wallnut3'}}
STATE_TEXTURES = {'polevault': {1: 'polevault2'}}


# Screen
WIDTH = 800
HEIGHT = 600
FPS = 60

# Tile and Grid
TILE_W = 81
TILE_H = 96

GRID_START_X = 35
GRID_START_Y = 85
GRID_ROWS = 6
GRID_COLS = 10

# Entity Coordinates
ZOMBIE_COORD_X_OFFSET = 70
ZOMBIE_COORD_Y_OFFSET = 125
PLANT_COORD_X_OFFSET = 110
PLANT_COORD_Y_OFFSET = 125

# Sprite Rendering 
UI_SUN_DISPLAY_POS = pygame.Vector2(85, 70)
UI_LAWN_MOWER_POS = pygame.Vector2(-28, 125)
UI_LAWN_POS = pygame.Vector2(-220, 0)
UI_SEEDSLOT_POS = pygame.Vector2(48, 0)
UI_SHOVEL_POS = pygame.Vector2(605, 0)
GRID_LINE_COLOR = 'black'

# Font
SUN_FONT_SIZE = 20
SUN_FONT_COLOR = 'black'


pygame.init()
pygame.font.init()

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

IMGS: dict[str, pygame.Surface] = {}
for path in Path('assets').rglob('*.*'):
    dir = path.parent.name
    name, ext = path.name.split('.')
    if ext == 'png':
        IMGS[name] = pygame.image.load(path).convert_alpha()
    else:
        IMGS[name] = pygame.image.load(path).convert()
    if size := IMG_SIZES[dir].get(name):
        IMGS[name] = pygame.transform.scale(IMGS[name], size)

SUN_FONT = pygame.font.SysFont('AgencyFB', SUN_FONT_SIZE, bold=True)

def z_coords(row: int, col: float, width: int, height: int) -> tuple[int, int]:
    x = round(ZOMBIE_COORD_X_OFFSET + TILE_W * col - width / 2)
    y = ZOMBIE_COORD_Y_OFFSET + TILE_H * row - height
    return x, y

def p_coords(row: int, col: int, width: int, height: int) -> tuple[int, int]:
    x = round(PLANT_COORD_X_OFFSET + TILE_W * col - width / 2)
    y = PLANT_COORD_Y_OFFSET + TILE_H * row - height
    return x, y

def render_misc(sun: int, lawn_mowers: np.ndarray[tuple[int]]):
    sun_txt = SUN_FONT.render(str(sun), True, SUN_FONT_COLOR)
    sun_rect = sun_txt.get_rect(center=UI_SUN_DISPLAY_POS)
    SCREEN.blits([
        (IMGS['lawn'], UI_LAWN_POS),
        (IMGS['seedslot'], UI_SEEDSLOT_POS),
        (IMGS['shovel'], UI_SHOVEL_POS),
        (sun_txt, sun_rect)
    ])
    SCREEN.blits([(IMGS['lawnmower'], (UI_LAWN_MOWER_POS.x, UI_LAWN_MOWER_POS.y + TILE_H * i)) for i in np.where(lawn_mowers > 0)[0]])

def render_plants(plant_state: np.ndarray[tuple[int, int]]):
    plants = []
    for row, col in np.argwhere(plant_state['type'] > 0):
        p = plant_state[row, col]
        name = PLANT_NAMES[p['type']][0]
        img = IMGS.get(name, IMGS['unknown_p'])
        plants.append((img, p_coords(row, col, *img.get_size())))
    SCREEN.blits(plants)

def render_zombies(zomb_state: np.ndarray[tuple[int, int]]):
    zombies = []
    for row, col in np.argwhere(zomb_state['type'] > 0):
        z = zomb_state[row, col][0]
        name: str = ZOMB_NAMES[z['type']][0]
        name = STATE_TEXTURES.get(name, {}).get(z['special_state'], name)
        img = IMGS.get(name, IMGS['unknown_z'])
        zombies.append((img, z_coords(row, z['x'], *img.get_size())))
    SCREEN.blits(zombies)

def draw_grid():
    for i in range(GRID_ROWS):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR, 
                        (GRID_START_X, GRID_START_Y + TILE_H * i), 
                        (GRID_START_X + TILE_W * GRID_COLS, GRID_START_Y + TILE_H * i))
    for i in range(GRID_COLS):
        pygame.draw.line(SCREEN, GRID_LINE_COLOR, 
                        (GRID_START_X + TILE_W * i, GRID_START_Y), 
                        (GRID_START_X + TILE_W * i, GRID_START_Y + TILE_H * (GRID_ROWS - 1)))

def play(game_engine: PvZGame):
    dt = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event == pygame.QUIT or event == pygame.K_ESCAPE:
                running = False

        game_outcome, *_ = game_engine.update(dt)
        running = (game_outcome == 0)

        render_misc(game_engine.sun, game_engine.lawn_mowers)
        # draw_grid()
        render_plants(game_engine.p)
        render_zombies(game_engine.z)

        pygame.display.flip()
        dt = CLOCK.tick(FPS) / 1000

    pygame.quit()


if __name__ == '__main__':
    zombie_points = np.array([0, 1, 0, 2, 2, 4, 2, 4, 7])
    p_fin = 1 / zombie_points
    lvlconfig = LevelConfig(5, np.array([0, 1]), p_fin)
    play(PvZGame(lvlconfig))
