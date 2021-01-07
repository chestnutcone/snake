# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:43:40 2020

@author: Oliver
"""
import os
import numpy as np
from matplotlib import cm
from PIL import Image

from tqdm import tqdm
from snake import SnakeGame, GameOver


class BasicSnakeGame(SnakeGame):
    
    def board_status(self):
        print_board = self.board.copy()
        for s in self.snake.body:
            print_board[s] = 0.5
        print_board[self.snake.body[0]] = 1 # define head
        # now pad the boarder with negative for the border
        print_board = np.pad(print_board, ((1,1), (1,1)), constant_values=-1)
        return print_board
    
    def play(self):
        # for debug purpose
        try:
            self.generate_reward()
            
            while True:
                # self.display_board()
                print(self.board_status())
                
                inp = input("move: ")
                if inp == 'bye':
                    raise GameOver
                direction_map = {'w': 0,
                                 's': 1,
                                 'a': 2,
                                 'd': 3}
                direction = direction_map[inp]
                status, reward, done = self.move(direction)
                print("Reward:", reward, "Done", done)
        except GameOver:
            if len(self.snake.body) == self.board_size**2:
                print("YOU WON")
            else:
                print("Game Over. You lost")
                

class BasicSnakeBot:
    def __init__(self, board_size):
        # initialize your game here
        pass

    def assess_model(self):
        # run x games and get average snake length
        snake_length = []
        steps_taken = []
        for game in tqdm(range(1000)):
            game_data = self.play_random_game(print_output=False, store_data=True)
            steps_taken.append(len(game_data))
            snake_length.append(len(self.game.snake.body))
            
        snake_length = np.array(snake_length)
        steps_taken = np.array(steps_taken)
        
        avg_snake_len = np.average(snake_length)
        avg_steps_taken = np.average(steps_taken)
        
        avg_steps_per_reward = avg_steps_taken / avg_snake_len
        avg_steps_per_reward = np.average(avg_steps_per_reward)
        print("\nAverage reward", avg_snake_len)
        print("Average steps taken", avg_steps_taken)
        print("Average steps per reward", avg_steps_per_reward)
        
    def make_gif(self, frames, gif_name='', scale_factor=100, duration=100,
                 img_size=(500,500), cmap=cm.hsv):
        gif_folder = 'gif'
        file_name = os.path.join(gif_folder, gif_name)
                   
        
        # normalize frames
        new_frames = []
        for frame in tqdm(frames):
            upscaled = frame + 1
            normalize_frame = upscaled / np.linalg.norm(upscaled) # normlize
            # now scale up the numpy array
            normalize_frame = np.kron(normalize_frame, np.ones((scale_factor, scale_factor)))
            # apply color map, rescale to 0-255, convert to int
            # then use Image.fromarray
            im = Image.fromarray(np.uint8(cmap(normalize_frame)*255))
            im = im.resize(img_size)
            new_frames.append(im)
        # save as gif
        new_frames[0].save(file_name, format='GIF', save_all=True, duration=duration,
                           append_images=new_frames[1:], loop=0)
        
    def animate_game(self, gif_name='', scale_factor=100):
        frames = self.play_random_game(print_output=False, store_data=True)
        self.make_gif(frames, gif_name, scale_factor)
        
    def play_random_game(self, initial_rewards=10, print_output=True, store_data=False):
        pass
    