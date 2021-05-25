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

import statistics as st

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


'''
first_delivery = False # variable for checking for first food delivery, used in calcualting the global heuristic

total_steps = 0 # number of times the update_model function has been called

first_d_col = -1 # colony that delivered the first food source

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
'''

class AntModel:

    def __init__(self):
        self.first_delivery = False # variable for checking for first food delivery, used in calcualting the global heuristic

        self.total_steps = 0 # number of times the update_model function has been called

        self.first_d_col = -1 # colony that delivered the first food source

        self.paths = []

        self.id_loc = []

        self.food_loc = ( 1, 1)

        self.ants = [] #list of numpy arrays, one for each ant colony

        self.ants_with_food = [] #seperate arrays for ants with food since their behavior changes

        self.pheromones = [] #list of numpy arrays, one for each ant colonies pheromone trails

        self.g_pheromones = np.zeros((grid_size, grid_size)) # array for keeping track of global pheromones

        self.colony_homes = []

        self.ants_total = np.zeros((grid_size, grid_size)) # array for keeping ant totals accross colonies

        self.maxdiag = math.sqrt( 2 * (grid_size ** 2) ) #length of the diagonal for normalizing euclidean distance to food source

        #initialize empty environment variable matricies
        for i in range(num_colonies):
            self.ants.append(np.zeros((grid_size, grid_size)))
            self.pheromones.append(np.zeros((grid_size, grid_size)))
            self.ants_with_food.append(np.zeros((grid_size, grid_size)))
            self.id_loc.append( np.zeros((grid_size, grid_size)) )
            antlst = []
            for m in range(ants_per_colony):
                antlst.append((m, 0))
            self.paths.append(dict(antlst))



#function for initializing ant colony positions, food source positions, obstacle positions
def init_model( ants_per_colony, grid_size, num_colonies, max_ants_per_cell):

        x = AntModel()

        #initialize ant colonies
        for i in range(num_colonies):

            while True: #loop to generate a new colony home until it finds cell not already occupied by the food source or another colonies home
                home = ( random.randint(0, grid_size-1), random.randint(0, grid_size-1) )
                if home != x.food_loc and home not in x.colony_homes:
                    break
            x.colony_homes.append(home)
            ants_left_to_place = ants_per_colony

            while(ants_left_to_place):

                if x.ants_total[home] != 0: #if the cell is not empty, find a new one
                    home = rand_direction(home)
                    continue
                if ants_left_to_place <= max_ants_per_cell:
                    x.ants[i][home] = ants_left_to_place
                    x.ants_total[home] = ants_left_to_place

                    break # when all the ants are placed, end the while loop
                else:
                    x.ants[i][home] = max_ants_per_cell
                    x.ants_total[home] = max_ants_per_cell
                    ants_left_to_place = ants_left_to_place - max_ants_per_cell

                home = rand_direction(home) #get a random adjacent cell
        return x


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


def check_sums(Model, i, outstr ):
    if Model.ants_total[(1,1)] != (Model.ants_with_food[i][(1,1)] + Model.ants[i][(1,1)]):
        print('colony: ' + str(i) + '\n loc: ' + outstr)
        print('total steps: ' + str(Model.total_steps))
    return 0

#function for moving the simulation forward one timestep
def update_model(Model, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, pher_trail, evap, g_evap):

    food_deliveries = 0
    Model.total_steps += 1

    #update colonies one at a time
    for i in range(num_colonies):
        #get all cells with ants
        (rowi, coli) = np.nonzero(Model.ants_total)
        ant_i = list(zip(rowi, coli)) # list of tuples containing coordinates with ants

        for j in ant_i:

            neighbors = get_surrounding_cells(j, grid_size)
            food_ant_probs = []
            food_ant_cells = []

            #ANTS WITH FOOD
            if Model.ants_with_food[i][j] != 0:

                #FOOD ARRAY -> ANTS ARRAY
                if (Model.colony_homes[i] in neighbors):
                    chome = Model.colony_homes[i]
                    while (Model.ants_total[chome] < max_ants_per_cell) and (Model.ants_with_food[i][j] != 0):
                        if Model.first_delivery:#This is where, if it's the first delivery, we grab the colony number to use for g_pheromones update
                            Model.first_d_col = i

                        Model.ants_total[j] -= 1
                        Model.ants_with_food[i][j] -= 1

                        Model.ants_total[chome] += 1
                        Model.ants[i][chome] += 1
                        food_deliveries += 1
                    #continue #after filling home cell with as many as possible, end food ant movement since the colony home is within range

                check_sums(Model, i, 'FOOD -> ANTS!!')


                #FOOD ARRAY -> FOOD ARRAY
                fa_count = Model.ants_with_food[i][j]
                for cel in neighbors:
                    if Model.ants_total[cel] < max_ants_per_cell:
                        food_ant_cells.append(cel)
                        food_ant_probs.append( (Model.pheromones[i][cel] + 1) )


                if food_ant_cells:
                    for p in range(int(fa_count)):

                        fchoice = random.choices(food_ant_cells, weights=food_ant_probs)

                        if Model.ants_total[fchoice[0]] >= max_ants_per_cell: #if cell is full, too bad move on
                            break

                        Model.ants_total[j] -= 1
                        Model.ants_with_food[i][j] -= 1
                        Model.ants_total[fchoice[0]] += 1
                        Model.ants_with_food[i][fchoice[0]] += 1
                else:#if there are no choices for the ants to move, pass
                    pass

                check_sums(Model, i, 'FOOD -> FOOD!!')

            ant_count = Model.ants[i][j]
            if ant_count == 0: # If there are no ants not carrying food, continue, skipping this itteration
                continue

            checksum = sum(sum(Model.ants_total))



            #ANTS ARRAY -> FOOD ARRAY i.e. picking up a snack
            #Continue after each option, since if the food cell is full, we want to just wait and not do anything else
            if Model.food_loc in neighbors:
                if (Model.ants_total[Model.food_loc] + ant_count) <= max_ants_per_cell:#if all ants fit into food cell

                    Model.ants_total[j] -= ant_count
                    Model.ants[i][j] = 0

                    Model.ants_with_food[i][Model.food_loc] = Model.ants_with_food[i][Model.food_loc] + ant_count
                    Model.ants_total[Model.food_loc] += ant_count

                    continue
                elif Model.ants_total[Model.food_loc] < max_ants_per_cell: #if only some will fit
                    room = max_ants_per_cell - Model.ants_total[Model.food_loc] #space left in food cell

                    Model.ants_total[Model.food_loc] = max_ants_per_cell
                    Model.ants_with_food[i][Model.food_loc] = max_ants_per_cell

                    Model.ants[i][j] = ant_count - room
                    ant_count = ant_count - room
                    Model.ants_total[j] -= room
                    continue
                else:#or if no ants will fit, pass
                    continue
                check_sums(Model, i, 'ANTS -> FOOD!!')


            #FOODLESS ANT MOVEMENT
            vallst = []
            vallstg = []
            cellst = []

            heuristic_dist = [(math.sqrt(((Model.food_loc[0] - nei[0])**2) + ( (Model.food_loc[1] - nei[1]) ** 2))) for nei in neighbors] #get list of euclidean distances to food source
            hmin = min(heuristic_dist)

            #calc probability for each of j's neighbors
            for nei in range(len(neighbors)):

                heur =  3 - (heuristic_dist[nei] - hmin)#normalize the measurement
                if Model.ants_total[neighbors[nei]] < max_ants_per_cell: # only add a cell if there is space in the cell for ants to move into

                    vallst.append( ( (Model.pheromones[i][neighbors[nei]] + 1) ** alpha) * ( heur ** beta ) )

                    vallstg.append( ( (Model.g_pheromones[neighbors[nei]] + 1) ** alpha) * ( heur ** beta ) )
                    cellst.append(neighbors[nei])


            #NO MOVEMENT OPTIONS == SKIP ANTS
            if not vallst or not cellst:
                continue

            sumval = sum(vallst)
            sumvalg = sum(vallstg)


            prob_list = []
            num_cells_to_calc = len(vallst)
            for op in range(num_cells_to_calc):
                prob_list.append( w*(vallst[op] / sumval) + (1-w)*(vallstg[op] / sumvalg) )

            #update ant position
            for an in range(int(ant_count)):

                try:
                    choice = random.choices(cellst, weights=prob_list)
                except:
                    print('random choice error. prob_list:  ')
                    print(prob_list)

                if Model.ants_total[choice[0]] >= max_ants_per_cell:#if the ants choice is full, do nothing
                    continue

                Model.ants[i][choice[0]] += 1
                Model.ants_total[choice[0]] += 1

                Model.ants[i][j] -= 1
                Model.ants_total[j] -= 1

                #add pheromone trails when ants move, ants deposit pheromones apon ariving in a new cell, updates global and local pheromone values
                Model.pheromones[i][choice[0]] += pher_trail
                #Model.g_pheromones[choice[0]] +=  g_pher_trail

    #update pheromone trails at the same time, after ants have moved
    for i in range(num_colonies):
        Model.pheromones[i] = Model.pheromones[i] * evap
        #Model.g_pheromones = Model.g_pheromones * g_evap

    #global pheromone implimentation
    #This should only run once untill init_model is called again resetting the sim
    if food_deliveries != 0 and Model.first_delivery == True:
        print('steps at time of g_pher update: ' + str(Model.total_steps))
        Model.first_delivery = False
        print('col home: ' + str(Model.colony_homes[Model.first_d_col]))
        distance_t = ((Model.food_loc[0]-Model.colony_homes[Model.first_d_col][0]), (Model.food_loc[1] -  Model.colony_homes[Model.first_d_col][1]))
        min_steps = max( abs(distance_t[0]), abs(distance_t[1]) )#minimum number of steps to reach food source and back
        max_steps = abs(distance_t[0]) + abs(distance_t[1])#max number of steps accepted before the "shortest path" found is not a helpful measure for future ants
        print('min: ' + str(min_steps) + ' max: ' + str(max_steps))

        start = Model.colony_homes[Model.first_d_col] # for keeping track of our location
        step_diff = Model.total_steps - min_steps



        '''
        distance_t is a tuple of two distances the first being the number of rows away the second being the number of columns away eg ( 4, -3).
        This While loop decreases the numbers in distance_t untill they are both zero i.e
        '''
        while(distance_t[0] != 0 and distance_t[1] != 0):
            bigger = max(distance_t)


            if abs(distance_t[0]) == abs(distance_t[1]):
                start = (( start[0] + int(distance_t[0] / abs(distance_t[0])) ),(start[1] + int(distance_t[1] / abs(distance_t[1]))) )

                Model.g_pheromones[start] += g_pher_trail
                if step_diff > 0: #add corner of path to account for potential step difference
                    step_diff -= 1
                    Model.g_pheromones[(start[0], (start[1] - int(distance_t[1]/abs(distance_t[1]))))] += g_pher_trail
                distance_t = ( (distance_t[0] - int(distance_t[0] / abs(distance_t[0]))) , (distance_t[1] - int(distance_t[1] / abs(distance_t[1]))) )

            elif abs(distance_t[0]) > abs(distance_t[1]):
                start = ( int(start[0] + int(distance_t[0] / abs(distance_t[0])) ), start[1])

                Model.g_pheromones[start] += g_pher_trail
                distance_t = ( (distance_t[0] - int(distance_t[0] / abs(distance_t[0]))), distance_t[1] )

            elif abs(distance_t[0]) < abs(distance_t[1]):
                start = ( start[0], int(start[1] + int(distance_t[1] / abs(distance_t[1]))) )

                Model.g_pheromones[start] += g_pher_trail
                distance_t = ( distance_t[0], (distance_t[1] - int(distance_t[1] / abs(distance_t[1]))) )
            else:
                print('This should never print')


    return food_deliveries



ModelObj = init_model(ants_per_colony, grid_size, num_colonies, max_ants_per_cell)

print(ModelObj.ants_total)

delivs = 0
for i in range(100):
    delivs += update_model(ModelObj, ants_per_colony, grid_size, num_colonies, max_ants_per_cell, pher_trail, evap, g_evap)


print(ModelObj.ants_total)
print('Deliveries:  ' + str(delivs))
