from pathlib import Path

import numpy as np
import pygame

from pools import PLANTS, ZOMBIES


IMG_SIZES = {'misc': {'lawnmower': (87, -1), 'shovel': (60, -1), 'seedpacket': (-1, 58), 'sun_icon': (56, -1),
                      'sun_small': (35, -1),'sun': (53, -1),'sun_large': (70, -1),},
             'zombies': {'basic': (70, -1), 'basic2': (70, -1), 'flag': (100, -1), 'flag2': (100, -1), 'unknownz': (70, -1),
                         'conehead': (70, -1), 'conehead2': (70, -1), 'conehead3': (70, -1),
                         'polevault': (-1, 114), 'polevault2': (-1, 114), 
                         'bucket': (70, -1), 'bucket2': (70, -1), 'bucket3': (70, -1),
                         'newspaper': (-1, 112), 'newspaper2': (-1, 112),'newspaper3': (-1, 112),'newspaper4': (-1, 112),
                         'screendoor': (-1, 112), 'screendoor2': (-1, 112), 'screendoor3': (-1, 112),
                         'football': (-1, 112), 'football2': (-1, 112), 'football3': (-1, 112), 'football4': (-1, 112),
                         'disco': (100, -1), 'disco2': (100, -1), 'backup': (-1, 112)},
             'plants': {'peashooter': (70, -1), 'unknownp': (70, -1), 'sunflower': (70, -1), 'cherrybomb': (-1, 75),
                        'wallnut': (-1, 75), 'wallnut2': (-1, 75), 'wallnut3': (-1, 75),
                        'potatomine': (70, -1), 'potatomine2': (70, -1), 'snowpea': (70, -1),
                        'chomper': (70, -1), 'chomper2': (70, -1), 'repeater': (70, -1),
                        'puffshroom': (35, -1), 'puffshroom2': (35, -1), 'puffshroom3': (35, -1),
                        'sunshroom': (25, -1), 'sunshroom2': (42, -1), 'sunshroom3': (70, -1),
                        'fumeshroom': (70, -1), 'scaredy': (60, -1), 'scaredy2': (60, -1)}}
HEALTH_TEXTURES = {'wallnut': {2667: 'wallnut2', 1333: 'wallnut3'},
                   'basic': {190: 'basic', 100: 'basic2'}, 'flag': {190: 'flag', 100: 'flag2'},
                   'newspaper': {290: 'newspaper2', 240: 'newspaper3', 190: 'newspaper4'},
                   'conehead': {440: 'conehead2', 310: 'conehead3', 190: 'basic', 100: 'basic2'},
                   'bucket': {940: 'bucket2', 600: 'bucket3', 190: 'basic', 100: 'basic2'},
                   'screendoor': {940: 'screendoor2', 600: 'screendoor3', 190: 'basic', 100: 'basic2'},
                   'football': {940: 'football2', 190: 'football3', 100: 'football4'}}
INERT_TEXTURES = {'potatomine': 'potatomine2', 'chomper': 'chomper2'}
STATE_TEXTURES = {'polevault': {1: 'polevault2'}, 'disco': {0: 'disco2'}, 
                  'puffshroom': {1: 'puffshroom2', 2: 'puffshroom3'},
                  'sunshroom': {1: 'sunshroom2', 2: 'sunshroom3'},
                  'scaredy': {1: 'scaredy2'}}


ZOMB_NAMES = np.char.decode(ZOMBIES['name'], 'utf-8')
PLANT_NAMES = np.char.decode(PLANTS['name'], 'utf-8')

WIDTH = 800
HEIGHT = 600
FPS = 60

GRID_START_X = 139
GRID_START_Y = 116
GRID_END_X = 778
GRID_END_Y = 541

GRID_ROWS = 5
GRID_COLS = 9
TILE_W = round((GRID_END_X - GRID_START_X) / GRID_COLS)
TILE_H = round((GRID_END_Y - GRID_START_Y) / GRID_ROWS)

LAWN_POS = pygame.Vector2(-362, -102)
LAWN_MOWER_POS = pygame.Vector2(46, 131)
SHOVEL_POS = pygame.Vector2(705, 537)
GRID_LINE_COLOR = 'black'

SUN_DISPLAY_POS = pygame.Vector2(91, 24)
SUN_FONT_SIZE = 24
SUN_FONT_TYPE = 'AgencyFB'
SUN_FONT_COLOR = 'white'

PROG_BAR_X = 225
PROG_BAR_Y = 10
PROG_BAR_W = 200
PROG_BAR_H = 15
PBAR_MARGIN = 5

SEED_START_X = 6
SEED_START_Y = 69
SEED_ICON_SIZE = (60, 52)

ZOMBIE_X_OFFSET = 30
ZOMBIE_Y_OFFSET = 72
PLANT_X_OFFSET = 35
PLANT_Y_OFFSET = 63

COST_FONT_SIZE = 24
COST_FONT_TYPE = 'AgencyFB'
COST_FONT_COLOR = 'black'

TRANSPARENT = (0, 0, 0, 0)
OVERLAY_BLACK = (0, 0, 0, 85)
HALF_BLACK = (0, 0, 0, 128)
OVERLAY_WHITE = (255, 255, 255, 85)
OVERLAY_BLUE = (20, 40, 100, 85)
NIGHT_BLUE = (10, 20, 50, 170)
LIGHT_GREEN = (80, 245, 50)


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

    IMGS[name] = img

    if dir not in ['zombies', 'plants']:
        continue

    mask = pygame.mask.from_surface(img)
    hit_overlay = mask.to_surface(setcolor=OVERLAY_WHITE, unsetcolor=TRANSPARENT)
    IMGS[f"{name}_hit"] = hit_overlay

    if dir == 'zombies':
        cool_overlay = mask.to_surface(setcolor=OVERLAY_BLUE, unsetcolor=TRANSPARENT)
        IMGS[f"{name}_cool"] = cool_overlay

    if dir == 'plants' and name.isalpha():
        new_w, new_h = SEED_ICON_SIZE
        if (w_scale := new_w / img.width) < (h_scale := new_h / img.height):
            new_h = img.height * w_scale
        else:
            new_w = img.width * h_scale

        IMGS[f"seed_{name}"] = pygame.transform.smoothscale(img, (new_w, new_h))


SUN_TXT_BOX = pygame.Surface((97, 28), pygame.SRCALPHA)
pygame.draw.rect(SUN_TXT_BOX, HALF_BLACK, SUN_TXT_BOX.get_rect(), border_radius=6)

PROGRESS_BAR = pygame.Surface((PROG_BAR_W + 2 * PBAR_MARGIN, PROG_BAR_H + 2 * PBAR_MARGIN), pygame.SRCALPHA)
pygame.draw.rect(PROGRESS_BAR, HALF_BLACK, PROGRESS_BAR.get_rect(), border_radius=6)
pygame.draw.rect(PROGRESS_BAR, TRANSPARENT, ((PBAR_MARGIN, PBAR_MARGIN), (PROG_BAR_W, PROG_BAR_H)))

SEED_DARK_OVERLAY = pygame.Surface(IMGS["seedpacket"].size, pygame.SRCALPHA)
SEED_LIGHT_OVERLAY = pygame.Surface(IMGS["seedpacket"].size, pygame.SRCALPHA)
SEED_DARK_OVERLAY.fill(OVERLAY_BLACK)
SEED_LIGHT_OVERLAY.fill(OVERLAY_WHITE)

LAWN_OVERLAY_V = pygame.Surface((TILE_W, TILE_H * GRID_ROWS), pygame.SRCALPHA)
LAWN_OVERLAY_H = pygame.Surface((TILE_W * GRID_COLS, TILE_H), pygame.SRCALPHA)
LAWN_OVERLAY_V.fill(OVERLAY_WHITE)
LAWN_OVERLAY_H.fill(OVERLAY_WHITE)

NIGHT_OVERLAY = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
NIGHT_OVERLAY.fill(NIGHT_BLUE)

SHOVEL_OVERLAY = pygame.Surface(IMGS["shovel"].size, pygame.SRCALPHA)
pygame.draw.circle(SHOVEL_OVERLAY, OVERLAY_WHITE, (30, 30), 30)
