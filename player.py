import numpy as np

from assets import (CLOCK, FPS, GRID_COLS, GRID_ROWS, GRID_START_X, GRID_START_Y, IMGS, LAWN_POS, NIGHT_OVERLAY, SCREEN, 
                    SEED_START_X, SEED_START_Y, SHOVEL_POS, TILE_H, TILE_W)
from game_logic import LevelConfig, PvZGame
from render import render_misc, render_plants, render_seedbank, render_zombies

import pygame


is_shovel = False


def process_clicks(game_engine: PvZGame, x: int, y: int):
    global is_shovel
    if (GRID_START_X <= x <= GRID_START_X + TILE_W * GRID_COLS) and (GRID_START_Y <= y <= GRID_START_Y + TILE_H * GRID_ROWS):
        # Clicked lawn
        lawn_row = (y - GRID_START_Y) // TILE_H
        lawn_col = (x - GRID_START_X) // TILE_W
        if is_shovel:
            game_engine.shovel_plant(lawn_row, lawn_col)
        else:
            game_engine.place_plant(lawn_row, lawn_col)
            is_shovel = False
        return

    seed_w, seed_h = IMGS["seedpacket"].size
    n_seeds = game_engine.seed_bank.shape[0]
    if (0 <= (x - SEED_START_X) <= seed_w) and (0 <= (y - SEED_START_Y) <= seed_h * n_seeds):
        # Clicked seed slot
        is_shovel = False
        idx = (y - SEED_START_Y) // seed_h
        if idx != game_engine.selected_plant_idx:
            game_engine.select_plant(idx)
            return
    
    game_engine.deselect_plant()
    shovel_w, shovel_h = IMGS["shovel"].size
    if (0 <= (x - SHOVEL_POS.x) <= shovel_w) and (0 <= (y - SHOVEL_POS.y) <= shovel_h):
        # Clicked shovel
        is_shovel = not is_shovel
        return
    
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
        if game_engine.lvlconfig.sun_cooldown == 'night':
            SCREEN.blit(NIGHT_OVERLAY, (0, 0))
        # draw_grid()

        to_blit = render_misc(game_engine.sun, game_engine.prog, game_engine.lvlconfig.n_flags, game_engine.lawn_mowers, is_shovel)
        to_blit.extend(render_seedbank(game_engine.seed_bank, game_engine.sun, game_engine.selected_plant_idx))
        to_blit.extend(render_plants(game_engine.p, plant_dmg_arr))
        to_blit.extend(render_zombies(game_engine.z, zombie_dmg_arr))
        SCREEN.blits(to_blit)

        pygame.display.flip()
        dt = CLOCK.tick(FPS) / 1000

    pygame.quit()


if __name__ == '__main__':
    p_fin = np.array([0, 1, 0, 1/2, 1/2, 1/4, 1/2, 1/4, 1/7])
    lvlconfig = LevelConfig(list(range(3, 11)), 5, np.array([0, 1]), p_fin, sun_cooldown='night')
    game_engine = PvZGame(lvlconfig)
    play(game_engine)
