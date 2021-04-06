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

alpha = 1

beta = .0001

w = .5

evap = 1


# Environment Variables

ants = [] #list of numpy arrays, one for each ant colony 

ants_with_food = [] #seperate arrays for ants with food since their behavior changes

pheromones = [] #list of numpy arrays, one for each ant colonies pheromone trails

food_loc = ( 0, 0)

colony_homes = []

maxdiag = math.sqrt( 2 * (grid_size ** 2) ) #length of the diagonal for normalizing euclidean distance to food source

#initialize empty environment variable matricies
for i in range(num_colonies):
    ants.append(np.zeros((grid_size, grid_size)))
    pheromones.append(np.ones((grid_size, grid_size)))
    ants_with_food.append(np.zeros((grid_size, grid_size)))




#function for initializing ant colony positions, food source positions, obstacle positions
def init_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes):

        #initialize ant colonies
        for i in range(num_colonies):

            home = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )
            colony_homes.append(home)
            ants_left_to_place = ants_per_colony

            while(ants_left_to_place):
                if ants[i][home] != 0: #if the cell is not empty, find a new one
                    home = rand_direction(home)
                    continue
                if ants_left_to_place <= max_ants_per_cell:
                    ants[i][home] = ants_left_to_place
                    break # when all the ants are placed, end the while loop
                else:
                    ants[i][home] = max_ants_per_cell
                    ants_left_to_place = ants_left_to_place - max_ants_per_cell

                home = rand_direction(home) #get a random adjacent cell

        #initialize food stores
        food_loc = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )    


#function for returning a random direction (up, down, left, right, diagnals). Takes as input a tuple of coordinates, returns a tuple of coordinates
def rand_direction(cell):
    if cell[0] == 0:
        ret1 = cell[0] + random.randint(0,1)
    elif cell[0] == 19:
        ret1 = cell[0] + random.randint(-1,0)
    else:
        ret1 = cell[0] + random.randint(-1,1)

    if cell[1] == 0:
        ret2 = cell[1] + random.randint(0,1)
    elif cell[1] == 19:
        ret2 = cell[1] + random.randint(-1,0)
    else:
        ret2 = cell[1] + random.randint(-1,1)

    return ( ret1, ret2)

#this function takes as input a cell and the grid size and returns the indicies of all surrounding cells
def get_surrounding_cells(cell, grid_size):

    tuple_retlst = []
    for x,y in [( cell[0]+i, cell[1]+j ) for i in (-1,0,1) for j in (-1,0,1) if i != 0 or j != 0]:
        if  0 <= x < grid_size and 0 <= y < grid_size:
            tuple_retlst.append((x, y))
            
    return tuple_retlst



#function for moving the simulation forward one timestep
def update_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes):

    for i in range(num_colonies):
        (rowi, coli) = np.nonzero(ants[i])
        ant_i = list(zip(rowi, coli)) # list of tuples containing coordinates with ants
        print('num of cells with ants: ' + str(len(ant_i)))

        for j in ant_i:

            ant_count = ants[i][j]
            neighbors = get_surrounding_cells(j, grid_size)
            
            vallst = []
            cellst = []

            for nei in neighbors:              

                dist = math.sqrt(((food_loc[0] - nei[0])**2) + ( (food_loc[1] - nei[1]) ** 2)) #gets euclidean distance to foodsource
                heur = (maxdiag - dist) / maxdiag #normalize the measurement
                vallst.append( (pheromones[i][nei] ** alpha) * ( heur ** beta ) )
                cellst.append(nei)
                
            sumval = sum(vallst)

            prob_list = [(w * gs / sumval) for gs in vallst]
            #print('prob_list: ', prob_list)

            ants[i][j] = 0
            for an in range(int(ant_count)):
                choice = random.choices(cellst, weights=prob_list)
                #while(ants[i][choice] >= max_ants_per_cell):
                    #choice = random.choices(cellst, weights=prob_list)
                ants[i][choice[0]] = ants[i][choice[0]] + 1




init_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes)#print(ants[0])

print(ants[0])
print(type(ants[0]))

for q in range(20):
    update_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes)

print(type(ants[0]))
print(ants[0])



