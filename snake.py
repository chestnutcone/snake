# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:31:48 2020

@author: Oliver
"""

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import random


class Snake:
    def __init__(self, initial_position, initial_len):
        # have to make sure initial_len isn't longer than board
        end_point = max(-1, initial_position - initial_len)
        
        self.body = [(initial_position, i) for i in range(initial_position, end_point, -1)]
        
    def move(self, new_position, reward=False):
        self.body.insert(0, new_position)
        if not reward:
            self.body.pop()


class GameOver(Exception):
    pass


class SnakeGame:
    def __init__(self, board_size, snake_reward=2, initial_snake_len=3, initial_num_rewards=1):
        assert board_size > 1
        
        self.board_size = board_size
        self.initial_snake_len = initial_snake_len

        # reward will take value of 2
        # snake take value of 1
        # empty space take value of 0
        self.snake_reward = snake_reward
        
        self.reset(initial_num_rewards)

        
    def reset(self, num_rewards=1):
        self.board = np.zeros((self.board_size, self.board_size))
        self.snake = Snake(int(self.board_size/2), self.initial_snake_len)
        self.generate_reward(num_rewards)
        # plt.ion()
        # fig, ax = plt.subplots()
        # self.graph = ax.imshow(self.board_status(), vmin=0, vmax=2)
        # plt.pause(0.01)
        
    def next_place(self, cur_place, direction):
        """
        {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        """
        # gives direction of next place
        # cur_place is tuple of index indicating current place on the board
        # arbitrary label 0 1 2 3 maps to up down left right
        (i, j) = cur_place
        if direction == 0:
            return (i-1, j)
        elif direction == 1:
            return (i+1, j)
        elif direction == 2:
            return (i, j-1)
        elif direction == 3:
            return (i, j+1)
        else:
            raise ValueError
        
    def check_valid(self, new_place):
        # check if new_place is a valid position on the board
        (i,j) = new_place
        if not 0 <= i < self.board_size:
            # out of bound
            return False
        if not 0 <= j < self.board_size:
            return False
        if new_place in self.snake.body:
            return False
        return True
    
    def move(self, direction):
        head = self.snake.body[0] # position of head
        next_head = self.next_place(head, direction)
        if self.check_valid(next_head):
            # check if reward exists
            on_reward = self.board[next_head] == self.snake_reward
            self.snake.move(next_head, on_reward) 
            if on_reward:
                self.board[next_head] = 0
                self.generate_reward()
        else:
            raise GameOver("Game Over")
    
    def new_reward(self):
        available = np.argwhere(self.board == 0)
        if len(available):
            random_int = random.randint(0,len(available)-1)
            position = tuple(available[random_int])
            self.board[position] = self.snake_reward
        
    def generate_reward(self, num_rewards=1):
        # if len(self.snake.body) % 5 == 0:
        #     # increase extra reward every 5 increase
        #     num_rewards += 1
            
        for i in range(num_rewards):
            self.new_reward()
    
    def board_status(self):
        print_board = self.board.copy()
        for s in self.snake.body:
            print_board[s] = 1
        return print_board
    
    def display_board(self):
        # need to print snake together with baord
        # pprint(self.board_status())
        pprint(self.board_status())
        self.graph.set_data(self.board_status())
        plt.pause(0.01)
        
    def play(self):
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
                self.move(direction)
        except GameOver:
            if len(self.snake.body) == self.board_size**2:
                print("YOU WON")
            else:
                print("Game Over. You lost")

if __name__ == '__main__':
    game = SnakeGame(5)
    game.play()
        
        
        