import numpy as np

import assets
from assets import (CLOCK, FPS, GRID_START_X, GRID_START_Y, IMGS, LAWN_POS, NIGHT_OVERLAY, SCREEN, SEED_START_X, 
                    SEED_START_Y, SHOVEL_POS)
from game_logic import LevelConfig, PvZGame
from render import render_misc, render_plants, render_seedbank, render_zombies

import pygame


is_shovel = False


def process_clicks(game_engine: PvZGame, x: int, y: int):
    global is_shovel

    x_in_lawn = (GRID_START_X <= x <= GRID_START_X + assets.TILE_W * assets.GRID_COLS)
    y_in_lawn = (GRID_START_Y <= y <= GRID_START_Y + assets.TILE_H * assets.GRID_ROWS)
    if x_in_lawn and y_in_lawn:
        # Clicked lawn
        lawn_row = (y - GRID_START_Y) // assets.TILE_H
        lawn_col = (x - GRID_START_X) // assets.TILE_W
        if is_shovel:
            game_engine.shovel_plant(lawn_row, lawn_col)
        else:
            game_engine.place_plant(lawn_row, lawn_col)
            is_shovel = False
        return

    seed_w, seed_h = IMGS["seedpacket"].size
    n_seeds = game_engine.seed_bank.shape[0]
    if (0 <= (x - SEED_START_X) <= seed_w) and (0 <= (y - SEED_START_Y) < seed_h * n_seeds):
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
    assets.set_grid_size(game_engine.lvlconfig.n_rows, game_engine.lvlconfig.n_cols)
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
    plants = [9, 10, 11, 13, 3, 4, 6, 14]
    p_init = np.array([1])
    p_fin = np.array([1, 0, 1/2, 1/2, 1/4, 1/2, 1/4, 1/7, 1/5])
    # p_init = p_fin
    lvlconfig = LevelConfig(plants, 1, p_init, p_fin, sun_cooldown='night', wave_size_ramp=0.7)
    game_engine = PvZGame(lvlconfig)
    play(game_engine)
