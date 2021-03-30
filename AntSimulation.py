#############################
#
# Dynamic models in Biology Project
# Gibson Olbrys, Robert Gomez
#
#############################


#Grid based approach

import numpy as np

import math

import matplotlib.pyplot as plt

import random


grid_size = 20

num_colonies = 1

max_ants_per_cell = 5

ants_per_colony = 50

ants = [] #list of numpy arrays, one for each ant colony 

ants_with_food = [] #seperate arrays for ants with food since their behavior changes

pheromones = [] #list of numpy arrays, one for each ant colonies pheromone trails

for i in range(num_colonies):
    ants.append(np.zeros((grid_size, grid_size)))
    pheromones.append(np.zeros((grid_size, grid_size)))
    ants_with_food.append(np.zeros((grid_size, grid_size)))




#function for initializing ant colony positions, food source positions, obstacle positions
def init_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell):

        for i in range(num_colonies):

            home = ( random.randint(0, grid_size), random.randint(0, grid_size) )
            ants_left_to_place = ants_per_colony

            if ants_left_to_place <= max_ants_per_cell:
                ants[i][home[0], home[1]] = ants_left_to_place
                continue
            else:
                ants[i][home[0], home[1]] = max_ants_per_cell
                ants_left_to_place = ants_left_to_place - max_ants_per_cell
                
            while(ants_left_to_place):
                new_cell = rand_direction(home)

#function for returning a random direction (up, down, left, right, diagnals). Takes as input a tuple of coordinates, returns a tuple of coordinates
def rand_direction(cell):
    if cell[0] == 0:
        ret1 = cell[0] + random.randint(0,1)
    else:
        ret1 = cell[0] + random.randint(-1,1)

    if cell[1] == 0:
        ret2 = cell[1] + random.randint(0,1)
    else:
        ret2 = cell[1] + random.randint(-1,1)

    return ( ret1, ret2)



test_tuple = (0, 0)
print(rand_direction(test_tuple))

