# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:58:55 2020

@author: Oliver
"""
import numpy as np
from scipy.spatial.distance import cdist
import pdb
from tqdm import tqdm
from matplotlib import cm
from basic_snake_bot import BasicSnakeGame, BasicSnakeBot, GameOver


class GreedySnakeBot(BasicSnakeBot):
    def __init__(self, board_size):
        self.game = BasicSnakeGame(board_size)
        self.max_steps_per_episodes = 1000
    
    def get_action(self):
        """This will cause looping. Need a memory to the nearest target?"""
        head = self.game.snake.body[0]
        reward_locations = np.argwhere(self.game.board == self.game.snake_reward)
        
        distances = cdist([head], reward_locations, metric='cityblock')
        dist_reward = [(d,r) for d,r in zip(distances[0], reward_locations)] # tuple (distance, reward_location)
        dist_reward.sort(key=lambda x: (x[0], x[1][0], x[1][1]))
        
        # pdb.set_trace()
        for possible in dist_reward:
            # figure out to where to move and if it is valid
            # figure up/down first
            vert, hori = possible[1] - head
            moves = self.get_direction(vert, hori)
            
            for direction in moves:
                next_head = self.game.next_place(head, direction)
                if self.game.check_valid(next_head):
                    return direction
            # if no moves towards nearest works, double check no moves left
            all_moves = [0, 1, 2, 3]
            for direction in all_moves:
                next_head = self.game.next_place(head, direction)
                if self.game.check_valid(next_head):
                    return direction
                    
        raise GameOver  # if no moves found
    
    def get_direction(self, vert, hori):
        """
        {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        """
        moves = []
        
        if vert > 0:
            moves.append(1)
        elif vert < 0:
            moves.append(0)
            
        if hori > 0:
            moves.append(3)
        elif hori < 0:
            moves.append(2)
            
        # if moves is empty, it means it is on a reward, which shouldn't happen?
        return moves
        
    def make_gif(self, frames, gif_name='', scale_factor=50, duration=100):
        super().make_gif(frames, gif_name, scale_factor, duration)
        
    def play_random_game(self, initial_rewards=10, print_output=True, store_data=False):
        game_data = []
        self.game.reset(num_rewards=initial_rewards)
        state = self.game.board_status()
        try:
            for step in range(self.max_steps_per_episodes):
                if print_output:
                    print(state)
                if store_data:
                    game_data.append(state)
                    
                direction = self.get_action()
                self.game.move(direction)
                state = self.game.board_status()
        except GameOver:
            pass
        
        return game_data