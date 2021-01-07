# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:27:30 2020

@author: Oliver
"""
import os
import numpy as np
# from PIL import Image
# import time
# from matplotlib import cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from snake import GameOver
from basic_snake_bot import BasicSnakeBot, BasicSnakeGame
from tqdm import tqdm


class RLSnakeGame(BasicSnakeGame):
    def __init__(self, board_size, snake_reward=1, initial_snake_len=3, initial_num_rewards=1, **kwargs):
        # ,reward_value, penalty_value, finish_value, stagnant_value
        print('kwargs', kwargs)
        super().__init__(board_size, snake_reward, initial_snake_len, initial_num_rewards)
        # this is for ml
        self.board_size_with_border = self.board_size + 2
        self.reward_value = kwargs['reward_value']  # eveery time it eats a point
        self.penalty_value = kwargs['penalty_value']  # invalid moves
        self.finsh_value = kwargs['finish_value']  # reward for finishing game
        self.stagnant_value = kwargs['stagnant_value']  # reward for not getting any reward
        
        # self.snake_reward = 2  # redefine the snake reward as 2 on print_board        
        
    def is_finished(self):
        # define finish game criteria here
        if len(self.snake.body) > (self.board_size ** 2) * 0.3:
            return True
        return False
    
    def move(self, direction):
        # for reinforcement learning
        reward = self.stagnant_value  # penalty for not doing anything?
        done = False
        
        head = self.snake.body[0] # position of head
        next_head = self.next_place(head, direction)
        if self.check_valid(next_head):
            # check if reward exists
            on_reward = self.board[next_head] == self.snake_reward
            self.snake.move(next_head, on_reward) 
            if on_reward:
                self.board[next_head] = 0
                self.generate_reward()
                reward = self.reward_value
                
                # check if game is finished
                if self.is_finished():
                    done = True
                    reward = self.finish_value
        else:
            # all invalid moves should receive negative penalities
            done = True
            reward = self.penalty_value
            
        return (self.board_status(), reward, done)
    
    def board_status(self):
        print_board = self.board.copy()
        for s in self.snake.body:
            print_board[s] = -1
        print_board[self.snake.body[0]] = -0.5 # define head
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
            if len(self.snake.body) == self.board_size**0.2:
                print("YOU WON")
            else:
                print("Game Over. You lost")
        

class SnakeBot(BasicSnakeBot):
    def __init__(self, board_size):
        self.game = RLSnakeGame(
            board_size,
            **{'reward_value': 100,
             'penalty_value': -20,
             'finish_value': 1000,
             'stagnant_value': -0.15
            }
        ) # set reward value here
        
        # configuration parameters
        self.seed = 42
        self.gamma = 0.99 # Discount factor for past rewards
        self.max_steps_per_episode = 500
        self.max_episodes = 10000
        self.eps = np.finfo(np.float32).eps.item()
        
        self.num_actions = 4
        self.model = None  # Placeholder
        
        # self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.optimizer = keras.optimizers.RMSprop()
        self.huber_loss = keras.losses.Huber()
        
        self.episode_count = 0
        self.running_reward = 0
        
        self.folder_name = 'models'
        
        
    def build_model(self):
        self.input_shape = (self.game.board_size_with_border,
                            self.game.board_size_with_border, 1, )
        
        inputs = layers.Input(shape=self.input_shape)
        layer1 = layers.Conv2D(64, (3,3), activation='relu')(inputs)
        # layer1 = layers.Conv2D(64, (3,3), activation='relu')(layer1)
        # layer1 = layers.Conv2D(128, (2,2), activation='relu')(layer1)
        # layer1 = layers.Conv2D(256, (2,2), activation='relu')(layer1)
        layer1 = layers.AveragePooling2D((2,2))(layer1)
        # layer2 = layers.Conv2D(64, 4, strides=2, activation='relu')(layer1)
        
        pooled = layers.Flatten()(layer1)
        # flat_layer1 = layers.Flatten()(layer1)
        # layer2 = layers.Concatenate()([pooled, flat_layer1])
        
        layer3 = layers.Dense(128, activation='relu')(pooled)
        layer3 = layers.Dense(64, activation='relu')(layer3)
        
        actor = layers.Dense(self.num_actions, activation='softmax')(layer3)
        critic = layers.Dense(1)(layer3)
        
        model = keras.Model(inputs=inputs, outputs=[actor, critic])
        self.model = model
        
    def train_model(self, initial_rewards=10):
        for episode in tqdm(range(self.max_episodes)):
            self.game.reset(num_rewards=initial_rewards)
            state = self.game.board_status()
            episode_reward = 0
            
            action_probs_history = []
            critic_value_history = []
            rewards_history = []
            with tf.GradientTape() as tape:
                for timestep in range(1, self.max_steps_per_episode):
                    # add a line to display the board                    
                    state = np.reshape(state, (-1, *self.input_shape))
                    action_probs, critic_value = self.model(state)
                    critic_value_history.append(critic_value[0,0])
                    
                    # sample action from probability distribution
                    action = np.random.choice(
                        self.num_actions, p=np.squeeze(action_probs))
                    # if episode % 500 == 0:
                    #     print(self.game.board_status())
                    #     print("Proob distrib", action_probs)
                    action_probs_history.append(
                        tf.math.log(action_probs[0, action]))
                    
                    # apply the sampled action in our environment
                    state, reward, done = self.game.move(action)
                    rewards_history.append(reward)
                    episode_reward += reward
                    
                    if done:
                        break
    
                # update running reward to check condition for solving
                self.running_reward = 0.05 * episode_reward + (1-0.05) * self.running_reward
                
                # calculate expected value from rewards
                # - At each timestep what was the total reward received after that time
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)
                    
                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()
                
                # calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = 'value' in the future. We took an action with log probability
                    # of 'log_prob' and ended up receiving a total reward = 'ret'
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability
                    diff = ret - value
                    actor_losses.append(-log_prob*diff)  # actor loss
                    
                    # the critic must be updated so that it predicts a better
                    # estimate of the future rewards
                    critic_losses.append(
                        self.huber_loss(tf.expand_dims(value, 0),
                        tf.expand_dims(ret, 0))
                    )
                
                # backgpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))
            # log details
            self.episode_count += 1
            if self.episode_count % 500 == 0:
                print(f"Running reward: {round(self.running_reward, 2)} at episode {self.episode_count}")
                
            if self.running_reward > 2500:
                print(f"Solved at episode {self.episode_count}!")
                break
     
    def play_random_game(self, initial_rewards=10, print_output=True, store_data=False):
        game_data = []
        
        self.game.reset(num_rewards=initial_rewards)
        state = self.game.board_status()
        for step in range(self.max_steps_per_episode):
            if print_output:
                print(state)
            if store_data:
                game_data.append(state)
            state = np.reshape(state, (-1, *self.input_shape))
            
            action_probs, _ = self.model(state)
            action = np.random.choice(
                self.num_actions, p=np.squeeze(action_probs))
            state, _, done = self.game.move(action)
            if done:
                # game end
                break
        return game_data
        
    def save_model(self, model_name=None, weight_name=None):
        if model_name:
            model_name = os.path.join(self.folder_name, model_name)
            with open(model_name, 'w') as f:
                f.write(self.model.to_json())
                
        if weight_name:
            weight_name = os.path.join(self.folder_name, weight_name)
            self.model.save_weights(weight_name)
            
    def load_model(self, model_name=None, weight_name=None):
        model_name = os.path.join(self.folder_name, model_name)
        weight_name = os.path.join(self.folder_name, weight_name)
        with open(model_name, 'r') as f:
            self.model = keras.models.model_from_json(f.read())
            _, h, w, d = self.model.input_shape
            self.input_shape = (h,w,d)
            self.game.board_size = h - 2 # minus border
            self.game.board_size_with_border = h
            
        self.model.load_weights(weight_name)