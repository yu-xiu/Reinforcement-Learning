# -*- coding: utf-8 -*-
"""
ETERNAL RETURN PROJECT.ipynb
# This is James lee's and Yu Xiu's project for CS 271
"""

import gym, ray
import numpy as np
from ray.rllib.agents import ppo,dqn
import math
from collections import Counter
import random

class gameMap(gym.Env):
  def __init__(self, config):
    self.container_locations = [] # if there is 5 containers, 0-4 are the 5 containers and 5 is exit
    self.container_contents = [] 
    self.container_location_contents = []
    self.cur_position = [1,1] # will be max + 2
    self.ending_point = [10,10] # is max + 1
    self.shopping_list = [] # need to ABCCCCCCCCC
    self.total_item_list = "AAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCCDDDDDDDDDDDDDDEEEEEEEEEEEEEEFFFFFFFFFFFFFFGGGGGGGGGHHHHIIIIIIJJJJJJJKKKKKKKKKKKLLLLLLLLLLLLLLMMMMMMMMMMMNNNNNNNNNOOOOPPPP" # list of all items to be distributed among containers. AAAABBBBBCCCD
    # 11a 14b 9c 14d 14e 14f 9g 4h 6i 7j 11k 14l 11m 9n 4o 4p
    to_shuffle = list(self.total_item_list)
    random.shuffle(to_shuffle)
    result = ''.join(to_shuffle)
    self.total_item_list = result
    self.step_number = 0
    self.obstacle_list = []
    self.time = 0
    self.speed = 1
    self.location_index = 0
    
    self.item_multiplier = 16
    self.completion_bonus = 200
    self.flat_time_cost = 1
    self.time_multiplier = 0.001

    # CURRENTLY COMMENTED OUT BECAUSE ITS HAVING A PROBLEM READING FILES
    mapFile = open("DockContainerlist.txt", "r")
    for line in mapFile:
      splitLine = line.split()
      x = float(splitLine[0])
      y = float(splitLine[1])
      
      oneContainer = (x,y)
      self.container_locations.append(oneContainer)
    # print(self.container_locations)
    shoppingListFile = open("ShoppingList.txt", "r")
    for line in shoppingListFile:
      item = line.split()[0]
      self.shopping_list.append(item)
   
    self.num_items = len(self.total_item_list)
    self.num_container = len(self.container_locations)
    self.num_of_sub_contents = int(self.num_items / self.num_container) # This only works in evenly divisible lists also there's no randomness to it so we should implement a few setups.

    self.location_index = len(self.container_locations) + 2
   
    
    # evenly distribute total items into all the containers of a zone
    self.start_index = 0
    self.container_contents_list = []
    for n in range(len(self.container_locations)):
      self.container_contents_list.append(self.total_item_list[self.start_index: (self.start_index + self.num_of_sub_contents)])
      self.start_index += self.num_of_sub_contents
   
    
    for num in range(len(self.container_locations)):
      temp = []
      temp.append(self.container_locations[num])
      temp.append(list(self.container_contents_list[num]))
      
      self.container_location_contents.append(temp)
    #print(f'list of container locations assoicated with their container contents: {self.container_location_contents}')
    

    # Each 'map' will have predetermined locations for containers.
    # the contents will be randomized each time
    # the agent will always start at 'start' and end at 'stop'
    # the scoring is based on a 'shopping list' that is also static.
    # action speed is determined around going from A-B
    # collding with a barrier increases time taken.

    # define action/state space
    self.action_space = gym.spaces.Discrete(len(self.container_locations)+1) # AI can choose to go or move to any container
    self.observation_space = gym.spaces.Discrete(len(self.container_locations) + 2) # AI can go to anywhere but the start
    
  def reset(self):
    self.step_number = 0
    # Scramble the total list
    to_shuffle = list(self.total_item_list)
    random.shuffle(to_shuffle)
    result = ''.join(to_shuffle)
    self.total_item_list = result
    self.cur_position = [1,1]
    
    # self.shopping_list = [] # TODO FILL
    self.shopping_list.clear()
    shoppingListFile = open("ShoppingList.txt", "r")
    for line in shoppingListFile:
      item = line.split()[0]
      self.shopping_list.append(item)

    # evenly distribute total items into all the containers of a zone
    self.start_index = 0
    #self.container_locations.clear()
    for n in range(len(self.container_locations)):
      self.container_contents_list.append(self.total_item_list[self.start_index: (self.start_index + self.num_of_sub_contents)])
      self.start_index += self.num_of_sub_contents
    
    self.location_index = len(self.container_locations) + 1 # length 31, and +1 for exit for example
    self.container_location_contents.clear()
    for num in range(len(self.container_locations)):
      temp = []
      temp.append(self.container_locations[num])
      temp.append(list(self.container_contents_list[num]))
      
      self.container_location_contents.append(temp)

    self.time = 0

    return self.location_index

  def step(self, action):
    self.step_number += 1
    self.time = self.flat_time_cost 
    pseudo_info = {}
    
    score_for_step = 0 # for return value

    if action >= len(self.container_locations):
      # Then we are at the end.
      destination = self.ending_point
    else:
      destination = self.container_locations[action]
      
    if self.location_index > len(self.container_locations):
      current_position = [1,1]
    else:
      current_position = self.container_locations[self.location_index]
    
    
    distance = math.sqrt(math.pow((destination[0] - current_position[0]), 2) + math.pow((destination[1] - current_position[1]), 2))
    self.cur_position = destination
    #print(distance)
    self.time += distance / self.speed
    #print(action, ' ', len(self.container_locations))
    if action >= len(self.container_locations):
      if(len(self.shopping_list)):
        score_for_step = 0 - self.time * (1 + pow(self.step_number, 2)* self.time_multiplier)
      else:
        score_for_step += self.completion_bonus - self.time * (1 + pow(self.step_number, 2)* self.time_multiplier)
      
      return 0, score_for_step, True, pseudo_info
    else:
      reward_for_items = 0
      # This means that we are not at the exit, which means we need to check the container.
      containerContents = self.container_location_contents[action][1] # accessing the string inside
      ## The below code checks the contents of the container then it deletes from both the shopping list and the container
      # We do not care about the user's capactiy at the time of writing this code
      intersection = Counter(self.shopping_list) & Counter(containerContents)
      multiset_shopping_list_minus_intersect = Counter(self.shopping_list) -  intersection

      multiset_container_minus_intersect = Counter(containerContents) -  intersection

      self.shopping_list = list(multiset_shopping_list_minus_intersect.elements())
      self.container_location_contents[action][1] = list(multiset_container_minus_intersect.elements())
      num_items_obtained = len(list(intersection.elements()))
      #print(num_items_obtained, 'Number of items obtained')
      
      score_for_step = num_items_obtained * self.item_multiplier - self.time * (1 + pow(self.step_number, 2)* self.time_multiplier)
      return action, score_for_step, False, pseudo_info
      
ray.shutdown()
ray.init()

config = {
    "env": gameMap,
    "env_config": {},
    "gamma": 0.4,
    "lr": 0.0006,
}

trainer = dqn.DQNTrainer(config=config)
#output = open("outputfile.txt", "x")

for _ in range(int(10)):
  print(trainer.train())

trainer.evaluate()
