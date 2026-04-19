from pathlib import Path

import numpy as np
import pygame

from game_logic import LevelConfig, PvZGame
from pools import ZOMBIES

WIDTH = 800
HEIGHT = 600
FPS = 60

IMG_SIZES = {'lawnmower': (69, 56), 'shovel': (72, 72)}
TILE_W = 81
TILE_H = 96

ZOMB_NAMES = ZOMBIES['name'].astype(str)

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

imgs: dict[str, pygame.Surface] = {}
for path in Path('assets').rglob('*.*'):
    name, ext = path.name.split('.')
    if ext == 'png':
        imgs[name] = pygame.image.load(path).convert_alpha()
    else:
        imgs[name] = pygame.image.load(path).convert()

sun_font = pygame.font.SysFont('AgencyFB', 20, bold=True)

def z_coords(row: int, col: float, width: int, height: int):
    return (round(70 + TILE_W * col - width / 2), 85 + TILE_H * row - 20)

def play(lvlconfig: LevelConfig):
    game_logic = PvZGame(lvlconfig)
    
    dt = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event == pygame.QUIT or event == pygame.K_ESCAPE:
                running = False

        game_outcome, *_ = game_logic.update(dt)
        running = (game_outcome == 0)

        sun_txt = sun_font.render(str(game_logic.sun), True, 'black')
        sun_rect = sun_txt.get_rect(center=(85, 70))
        screen.blits([
            (imgs['lawn'], (-220, 0)),
            (imgs['seedslot'], (48, 0)),
            (imgs['shovel'], (605, 0)),
            (sun_txt, sun_rect)
        ])
        screen.blits([(imgs['lawnmower'], (-28, 125 + TILE_H * i)) for i in np.where(game_logic.lawn_mowers > 0)[0]])

        # for i in range(6):
        #     pygame.draw.line(screen, 'black', (35, 85 + TILE_H * i), (35 + TILE_W * 9, 85 + TILE_H * i))
        # for i in range(10):
        #     pygame.draw.line(screen, 'black', (35 + TILE_W * i, 85), (35 + TILE_W * i, 85 + TILE_H * 5))

        zombies = []
        for r, row in enumerate(game_logic.z):
            if (row['type'] == 0).all():
                continue
            for z in row[row['type'] > 0]:
                img = imgs.get(ZOMB_NAMES[z['type']].item(), imgs['unknown_z'])
                zombies.append((img, z_coords(r, z['x'], *img.size)))
        screen.blits(zombies)

        pygame.display.flip()
        dt = clock.tick(FPS) / 1000

    pygame.quit()


if __name__ == '__main__':
    p_fin = [0, 1, 0, 0.5, 0.5, 0.25, 0.5, 0.25, 1/7]
    lvlconfig = LevelConfig(5, np.array([0, 1]), np.array(p_fin))
    play(lvlconfig)
