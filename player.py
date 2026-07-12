from pathlib import Path

import numpy as np
import pygame

from game_logic import LevelConfig, PvZGame
from pools import PLANTS, ZOMBIES


IMG_SIZES = {'misc': {'lawnmower': (-1, 77), 'shovel': (60, -1), 'seedpacket': (-1, 58), 'sun': (56, -1)},
             'zombies': {'basic': (70, -1), 'basic2': (70, -1), 'flag': (80, -1), 'unknownz': (70, -1),
                         'conehead': (70, -1), 'conehead2': (70, -1), 'conehead3': (70, -1),
                         'polevault': (-1, 114), 'polevault2': (70, 114), 'bucket': (70, -1), 'bucket2': (70, -1), 'bucket3': (70, -1), 
                         'newspaper': (-1, 112), 'newspaper2': (-1, 112),'newspaper3': (-1, 112),'newspaper4': (-1, 112), 
                          'screendoor': (-1, 112), 'screendoor2': (-1, 112), 'screendoor3': (-1, 112),
                          'football': (-1, 112), 'football2': (-1, 112), 'football3': (-1, 112), 'football4': (-1, 112),},
             'plants': {'sunflower': (-1, 75), 'wallnut': (-1, 75), 'wallnut2': (-1, 75), 'wallnut3': (-1, 75),
                        'peashooter': (-1, 75), 'cherrybomb': (-1, 75), 'potatomine': (70, -1), 'potatomine2': (70, -1), 
                        'snowpea': (-1, 75), 'chomper': (70, -1), 'chomper2': (70, -1), 'repeater': (-1, 75), 'unknownp': (-1, 75)}}
HEALTH_TEXTURES = {'wallnut': {2667: 'wallnut2', 1333: 'wallnut3'},
                   'basic': {190: 'basic', 100: 'basic2'}, 
                   'newspaper': {290: 'newspaper2', 240: 'newspaper3', 190: 'newspaper4'},
                   'conehead': {440: 'conehead2', 310: 'conehead3', 190: 'basic', 100: 'basic2'},
                   'bucket': {940: 'bucket2', 600: 'bucket3', 190: 'basic', 100: 'basic2'},
                   'screendoor': {940: 'screendoor2', 600: 'screendoor3', 190: 'basic', 100: 'basic2'},
                   'football': {940: 'football2', 190: 'football3', 100: 'football4'}}
INERT_TEXTURES = {'potatomine': 'potatomine2', 'chomper': 'chomper2'}
STATE_TEXTURES = {'polevault': {1: 'polevault2'}}


ZOMB_NAMES = np.char.decode(ZOMBIES['name'], 'utf-8')
PLANT_NAMES = np.char.decode(PLANTS['name'], 'utf-8')

WIDTH = 1065
HEIGHT = 600
FPS = 60

TILE_W = 76
TILE_H = 91

GRID_START_X = 358
GRID_START_Y = 84
GRID_ROWS = 5
GRID_COLS = 9

LAWN_POS = pygame.Vector2(-178, -150)
LAWN_MOWER_POS = pygame.Vector2(255, 102)
SHOVEL_POS = pygame.Vector2(978, 537)
GRID_LINE_COLOR = 'black'

SUN_DISPLAY_POS = pygame.Vector2(91, 24)
SUN_FONT_SIZE = 24
SUN_FONT_TYPE = 'AgencyFB'
SUN_FONT_COLOR = 'white'

SEED_START_X = 6
SEED_START_Y = 69

ZOMBIE_X_OFFSET = 35
ZOMBIE_Y_OFFSET = 85
PLANT_X_OFFSET = 75
PLANT_Y_OFFSET = 80

COST_FONT_SIZE = 24
COST_FONT_TYPE = 'AgencyFB'
COST_FONT_COLOR = 'black'

OVERLAY_BLACK = (0, 0, 0, 85)
OVERLAY_WHITE = (255, 255, 255, 85)


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
    hit_overlay = mask.to_surface(setcolor=OVERLAY_WHITE, unsetcolor=(0, 0, 0, 0))
    IMGS[f"{name}_hit"] = hit_overlay

    if dir == 'zombies':
        cool_overlay = mask.to_surface(setcolor=(20, 40, 100, 85), unsetcolor=(0, 0, 0, 0))
        IMGS[f"{name}_cool"] = cool_overlay

    if dir == 'plants' and name.isalpha():
        IMGS[f"seed_{name}"] = pygame.transform.scale_by(img, min(60 / img.width, 52 / img.height))


SUN_TXT_BOX = pygame.Surface((97, 28), pygame.SRCALPHA)
pygame.draw.rect(SUN_TXT_BOX, (0, 0, 0, 170), SUN_TXT_BOX.get_rect(), border_radius=6)

SEED_DARK_OVERLAY = pygame.Surface(IMGS["seedpacket"].size, pygame.SRCALPHA)
SEED_LIGHT_OVERLAY = pygame.Surface(IMGS["seedpacket"].size, pygame.SRCALPHA)
SEED_DARK_OVERLAY.fill(OVERLAY_BLACK)
SEED_LIGHT_OVERLAY.fill(OVERLAY_WHITE)

LAWN_OVERLAY_V = pygame.Surface((TILE_W, TILE_H * GRID_ROWS), pygame.SRCALPHA)
LAWN_OVERLAY_H = pygame.Surface((TILE_W * GRID_COLS, TILE_H), pygame.SRCALPHA)
LAWN_OVERLAY_V.fill(OVERLAY_WHITE)
LAWN_OVERLAY_H.fill(OVERLAY_WHITE)

SHOVEL_OVERLAY = pygame.Surface(IMGS["shovel"].size, pygame.SRCALPHA)
pygame.draw.circle(SHOVEL_OVERLAY, OVERLAY_WHITE, (30, 30), 30)

is_shovel = False
img_reported_missing = []


def render_misc(sun: int, lawn_mowers: np.ndarray[tuple[int]]):
    sun_txt = SUN_FONT.render(str(sun), True, SUN_FONT_COLOR)
    sun_txt_rect = sun_txt.get_rect(center=SUN_DISPLAY_POS)
    to_blit = [
        (IMGS['shovel'], SHOVEL_POS),
        (SUN_TXT_BOX, (31, 10)),
        (IMGS['sun'], (11, -2)),
        (sun_txt, sun_txt_rect)
    ]

    if is_shovel:
        to_blit.append((SHOVEL_OVERLAY, SHOVEL_POS))

    x, y = pygame.mouse.get_pos()
    if (GRID_START_X <= x <= GRID_START_X + TILE_W * GRID_COLS) and (GRID_START_Y <= y <= GRID_START_Y + TILE_H * GRID_ROWS):
        lawn_col = (x - GRID_START_X) // TILE_W
        lawn_row = (y - GRID_START_Y) // TILE_H
        to_blit.append((LAWN_OVERLAY_V, (GRID_START_X + TILE_W * lawn_col, GRID_START_Y)))
        to_blit.append((LAWN_OVERLAY_H, (GRID_START_X, GRID_START_Y + TILE_H * lawn_row)))

    for i in np.where(lawn_mowers > 0)[0]:
        to_blit.append((IMGS['lawnmower'], (LAWN_MOWER_POS.x, LAWN_MOWER_POS.y + TILE_H * i)))
    return to_blit


def render_seedbank(seedbank: np.ndarray[tuple[int, int]], sun: int, selected_idx: int | None):
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
        
        rect = img.get_rect(center=(SEED_START_X + round(seed_w * 0.36), seed_y + seed_h // 2))
        seeds.append((img, rect))

        cost_txt = COST_FONT.render(str(seed['cost']), False, COST_FONT_COLOR)
        cost_rect = cost_txt.get_rect(bottomright=(SEED_START_X + round(seed_w * 0.92), seed_y + round(seed_h * 0.95)))
        seeds.append((cost_txt, cost_rect))

        recharge_prog = 1.0
        if seed['recharge'] > 0:
            recharge_prog = 1 - float(seed['timer'] / seed['recharge'])

        if recharge_prog < 1:
            overlay_rect = pygame.Rect(0, 0, seed_img.width, round(seed_img.height * (1 - recharge_prog)))
            seeds.append((SEED_DARK_OVERLAY, seed_pos, overlay_rect))
        if recharge_prog < 1 or sun < seed['cost']:
            seeds.append((SEED_DARK_OVERLAY, seed_pos))
        if i is not None and i == selected_idx:
            seeds.append((SEED_LIGHT_OVERLAY, seed_pos))

    return seeds


def render_plants(plant_state: np.ndarray[tuple[int, int]], damage_array: np.ndarray[tuple[int, int]]):
    plants = []
    for row, col in np.argwhere(plant_state['type'] > 0):
        p = plant_state[row, col]
        
        name: str = PLANT_NAMES[int(p['type'])]

        health_textures = HEALTH_TEXTURES.get(name, {})
        if health_textures and (ks := [h for h in health_textures.keys() if p['health'] < h]):
            name = health_textures[min(ks)]

        if p['timer'] > 0:
            name = INERT_TEXTURES.get(name, name)
        
        if name not in IMGS:
            if name not in img_reported_missing:
                print(f"Image '{name}' not found")
                img_reported_missing.append(name)
            name = 'unknownp'
        img = IMGS[name]

        mid = GRID_START_X + PLANT_X_OFFSET + TILE_W * (col - 0.5)
        bottom = GRID_START_Y + PLANT_Y_OFFSET +TILE_H * row
        rect = img.get_rect(midbottom=(mid, bottom))

        plants.append((img, rect))
        if damage_array[row, col] > 0:
            plants.append((IMGS[f"{name}_hit"], rect))

    return plants


def render_zombies(zomb_state: np.ndarray[tuple[int, int]], damage_array: np.ndarray[tuple[int, int]]):
    zombies = []
    for row, col in np.argwhere(zomb_state['type'] > 0):
        z = zomb_state[row, col]

        name: str = ZOMB_NAMES[int(z['type'])]
        
        name = STATE_TEXTURES.get(name, {}).get(int(z['special_state']), name)

        health_textures = HEALTH_TEXTURES.get(name, {})
        if health_textures and (ks := [h for h in health_textures.keys() if z['health'] + z['shield_health'] < h]):
            name = health_textures[min(ks)]

        if name not in IMGS:
            if name not in img_reported_missing:
                print(f"Image '{name}' not found")
                img_reported_missing.append(name)
            name = 'unknownz'
        img = IMGS[name]

        mid = GRID_START_X + ZOMBIE_X_OFFSET + TILE_W * float(z['x'])
        bottom = GRID_START_Y + ZOMBIE_Y_OFFSET + TILE_H * row
        rect = img.get_rect(midbottom=(mid, bottom))

        zombies.append((img, rect))
        if z['slow_timer'] > 0:
            zombies.append((IMGS[f"{name}_cool"], rect))
        if damage_array[row, col] > 0:
            zombies.append((IMGS[f"{name}_hit"], rect))

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
        

def process_clicks(game_engine: PvZGame, x: int, y: int):
    global is_shovel 
    if (GRID_START_X <= x <= GRID_START_X + TILE_W * GRID_COLS) and (GRID_START_Y <= y <= GRID_START_Y + TILE_H * GRID_ROWS):
        # Clicked lawn
        lawn_row = (y - GRID_START_Y) // TILE_H
        lawn_col = (x - GRID_START_X) // TILE_W
        if is_shovel:
            game_engine.shovel_plant(lawn_row, lawn_col)
            is_shovel = False
        else:
            game_engine.place_plant(lawn_row, lawn_col)
        return
    
    shovel_w, shovel_h = IMGS["shovel"].size
    if (0 <= (x - SHOVEL_POS.x) <= shovel_w) and (0 <= (y - SHOVEL_POS.y) <= shovel_h):
        # Clicked shovel
        is_shovel = not is_shovel
        return
    
    seed_w, seed_h = IMGS["seedpacket"].size
    n_seeds = game_engine.seed_bank.shape[0]
    if (0 <= (x - SEED_START_X) <= seed_w) and (0 <= (y - SEED_START_Y) <= seed_h * n_seeds):
        # Clicked seed slot
        idx = (y - SEED_START_Y) // seed_h
        if idx != game_engine.selected_plant_idx:
            game_engine.select_plant(idx)
            return
    
    game_engine.deselect_plant()
    if is_shovel:
        is_shovel = False
    

def play(game_engine: PvZGame):
    dt = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                process_clicks(game_engine, *event.pos)

        game_outcome, zombie_dmg_arr, plant_dmg_arr, *_ = game_engine.update(dt)
        running = running and (game_outcome == 0)

        SCREEN.blit(IMGS['lawn'], LAWN_POS)
        # draw_grid()

        to_blit = render_misc(game_engine.sun, game_engine.lawn_mowers)
        to_blit.extend(render_seedbank(game_engine.seed_bank, game_engine.sun, game_engine.selected_plant_idx))
        to_blit.extend(render_plants(game_engine.p, plant_dmg_arr))
        to_blit.extend(render_zombies(game_engine.z, zombie_dmg_arr))
        SCREEN.blits(to_blit)

        pygame.display.flip()
        dt = CLOCK.tick(FPS) / 1000

    pygame.quit()


if __name__ == '__main__':
    p_fin = np.array([0, 1, 0, 1/2, 1/2, 1/4, 1/2, 1/4, 1/7])
    lvlconfig = LevelConfig(list(range(1, 9)), 5, np.array([0, 1]), p_fin)
    game_engine = PvZGame(lvlconfig)
    play(game_engine)
