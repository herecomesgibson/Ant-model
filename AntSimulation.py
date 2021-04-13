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

'''
Simulation Constants
'''

grid_size = 20

num_colonies = 1

max_ants_per_cell = 5

ants_per_colony = 50

alpha = 1

beta = 1

w = .5

'''
Pheromone constants, created sepperate constants for global and local pheromones, will probably keep their values the same but its nice to have the control
'''

evap = .9

pher_trail = 1 

g_evap = evap

g_pher_trail = pher_trail
##############################################
# Environment Variables

paths = []

id_loc = []

food_loc = ( 1, 1)

ants = [] #list of numpy arrays, one for each ant colony 

ants_with_food = [] #seperate arrays for ants with food since their behavior changes

pheromones = [] #list of numpy arrays, one for each ant colonies pheromone trails

g_pheromones = np.zeros((grid_size, grid_size)) # array for keeping track of global pheromones

colony_homes = []

ants_total = np.zeros((grid_size, grid_size)) # array for keeping ant totals accross colonies

maxdiag = math.sqrt( 2 * (grid_size ** 2) ) #length of the diagonal for normalizing euclidean distance to food source

#initialize empty environment variable matricies
for i in range(num_colonies):
    ants.append(np.zeros((grid_size, grid_size)))
    pheromones.append(np.zeros((grid_size, grid_size)))
    ants_with_food.append(np.zeros((grid_size, grid_size)))
    id_loc.append( np.zeros((grid_size, grid_size)) )
    antlst = []
    for m in range(ants_per_colony):
        antlst.append((m, 0))
    paths.append(dict(antlst))



#function for initializing ant colony positions, food source positions, obstacle positions
def init_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes):

        #initialize food stores
        #commenting out for now so that the food location is (0,0) so that I can determine in trials whether the ants path finding is working
        #food_loc = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )

        #initialize ant colonies
        for i in range(num_colonies):

            while True: #loop to generate a new colony home until it finds cell not already occupied by the food source or another colonies home
                home = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )
                if home != food_loc and home not in colony_homes:
                    break
            colony_homes.append(home)
            ants_left_to_place = ants_per_colony

            while(ants_left_to_place):

                if ants_total[home] != 0: #if the cell is not empty, find a new one
                    home = rand_direction(home)
                    continue
                if ants_left_to_place <= max_ants_per_cell:
                    ants[i][home] = ants_left_to_place
                    ants_total[home] = ants_left_to_place
                    
                    break # when all the ants are placed, end the while loop
                else:
                    ants[i][home] = max_ants_per_cell
                    ants_total[home] = max_ants_per_cell
                    ants_left_to_place = ants_left_to_place - max_ants_per_cell

                home = rand_direction(home) #get a random adjacent cell

        #initialize food stores
        #commenting out for now so that the food location is (0,0) so that I can determine in trials whether the ants path finding is working
        #food_loc = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )    


#function for returning a random direction (up, down, left, right, diagnals) from a given cell. Takes as input a tuple of coordinates, returns a tuple of coordinates
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
def update_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes, pher_trail, evap, g_evap, ants_with_food, pheromones, g_pheromones):

    food_deliveries = 0

    for i in range(num_colonies):

        (rowi, coli) = np.nonzero(ants_total)
        ant_i = list(zip(rowi, coli)) # list of tuples containing coordinates with ants

        #print('num of cells with ants: ' + str(len(ant_i)))

        for j in ant_i:

            neighbors = get_surrounding_cells(j, grid_size)

            food_ant_probs = []
            food_ant_cells = []



            #ANTS WITH FOOD
            if ants_with_food[i][j] != 0:

                #FOOD ARRAY -> ANTS ARRAY                
                if (colony_homes[i] in neighbors):
                    chome = colony_homes[i]
                    while (ants_total[chome] < max_ants_per_cell) and (ants_with_food[i][j] != 0):

                        ants_total[j] -= 1
                        ants_with_food[i][j] -= 1

                        ants_total[chome] += 1
                        ants[i][chome] += 1
                        food_deliveries += 1
                    break #after filling home cell with as many as possible, end food ant movement since the colony home is within range



                #FOOD ARRAY -> FOOD ARRAY
                fa_count = ants_with_food[i][j]
                for cel in neighbors:
                    if ants_total[cel] < max_ants_per_cell:
                        food_ant_cells.append(cel)
                        food_ant_probs.append( pheromones[i][cel] + 1)


                if food_ant_cells:
                    for p in range(int(fa_count)):

                        fchoice = random.choices(food_ant_cells, weights=food_ant_probs)

                        if ants_total[fchoice[0]] >= max_ants_per_cell: #if cell is full, too bad move on
                            break
                        
                        ants_total[j] -= 1
                        ants_with_food[i][j] -= 1
                        ants_total[fchoice[0]] += 1
                        ants_with_food[i][fchoice[0]] += 1
                else:#if there are no choices for the ants to move, pass
                    pass

                

            ant_count = ants[i][j]
            if ant_count == 0: # If there are no ants not carrying food, continue
                continue
            
            checksum = sum(sum(ants_total))



            #ANTS ARRAY -> FOOD ARRAY i.e. picking up a snack
            if food_loc in neighbors:
                if (ants_total[food_loc] + ant_count) <= max_ants_per_cell:#if all ants fit into food cell
                    
                    ants_total[j] -= ant_count
                    ants[i][j] = 0

                    ants_with_food[i][food_loc] = ants_with_food[i][food_loc] + ant_count
                    ants_total[food_loc] = ants_total[food_loc] + ant_count
                    
                    continue 
                elif ants_total[food_loc] < max_ants_per_cell: #if only some will fit
                    room = max_ants_per_cell - ants_total[food_loc] #space left in food cell

                    ants_total[food_loc] = max_ants_per_cell
                    ants_with_food[i][food_loc] = max_ants_per_cell

                    ants[i][j] = ant_count - room
                    ant_count = ant_count - room
                    ants_total[j] -= room
                else:#or if no ants will fit, pass
                    pass
                    
            if checksum != sum(sum(ants_total)):
                    print('UH OH!!!!!!!!!!!!!!!!!!!!!!!!!!!' + str(j))
                    print(sum(sum(ants_total)))

            #FOODLESS ANT MOVEMENT 
            vallst = []
            cellst = []

            heuristic_dist = [(math.sqrt(((food_loc[0] - nei[0])**2) + ( (food_loc[1] - nei[1]) ** 2))) for nei in neighbors] #get list of euclidean distances to food source
            hmin = min(heuristic_dist)

            #calc probability for each of j's neighbors
            for nei in range(len(neighbors)):              

                heur =  3 - (heuristic_dist[nei] - hmin)#normalize the measurement
                if ants_total[neighbors[nei]] < max_ants_per_cell: # only add a cell if there is space in the cell for ants to move into
                    vallst.append( ( (pheromones[i][neighbors[nei]] + 1) ** alpha) * ( heur ** beta ) )
                    cellst.append(neighbors[nei])

                    
            #NO MOVEMENT OPTIONS == SKIP ANTS
            if not vallst or not cellst:
                continue

            sumval = sum(vallst)

            prob_list = [(w * gs / sumval) for gs in vallst]

            #update ant position
            for an in range(int(ant_count)):

                try:
                    choice = random.choices(cellst, weights=prob_list)
                except:
                    print('random choice error. prob_list:  ')
                    print(prob_list)
                    
                if ants_total[choice[0]] >= max_ants_per_cell:#if the ants choice is full, do nothing
                    continue

                ants[i][choice[0]] += 1
                ants_total[choice[0]] += 1

                ants[i][j] -= 1
                ants_total[j] -= 1

                #add pheromone trails when ants move, ants deposit pheromones apon ariving in a new cell, updates global and local pheromone values
                pheromones[i][choice[0]] += pher_trail
                g_pheromones[choice[0]] +=  g_pher_trail

    #update pheromone trails at the same time, after ants have moved                                
    for i in range(num_colonies):
        pheromones[i] = pheromones[i] * evap
        g_pheromones = g_pheromones * g_evap

    return food_deliveries

init_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes)#print(ants[0])

print('sum:  ' + str(sum(sum(ants_total))))
print(ants[0])

print(type(ants[0]))

delv = 0

for q in range(1000):
    delv += update_model(ants, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, food_loc, colony_homes, pher_trail, evap, g_evap, ants_with_food, pheromones, g_pheromones)
testt = np.ones((grid_size, grid_size))



print('sum:  ' + str(sum(sum(ants_total))))
#print(ants_with_food[0])
print(ants_total)
print('Deliveries:  ' + str(delv))
print(pheromones[0])

