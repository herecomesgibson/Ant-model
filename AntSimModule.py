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
'''
grid_size = 20

num_colonies = 3

max_ants_per_cell = 5

ants_per_colony = 50

alpha = 1

beta = 1

w = .5

#Pheromone constants, created sepperate constants for global and local pheromones, will probably keep their values the same but its nice to have the control

evap = .9

pher_trail = 1

g_evap = evap

g_pher_trail = pher_trail
'''


#class for holding variable parameters
class Params:
    def __init__(self, grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail):

        self.grid_size = grid_size

        self.num_colonies = num_cols

        self.max_ants_per_cell = max_oc

        self.ants_per_colony = ants_per

        self.alpha = alpha

        self.beta = beta

        self.w = w

        #Pheromone constants, created sepperate constants for global and local pheromones, will probably keep their values the same but its nice to have the control
        self.evap = evap

        self.pher_trail = trail

        self.g_evap = self.evap

        self.g_pher_trail = self.pher_trail




#class for holding all state info for the model, this class inherits from Params
class AntModel(Params):

    def __init__(self, grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail):

        #inherit parameters from Params
        super().__init__(grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail)

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

        self.col_deliv = [] #array to hold the number of deliveries per colony


        #initialize empty environment variable matricies
        for i in range(self.num_colonies):
            self.ants.append(np.zeros((self.grid_size, self.grid_size)))
            self.pheromones.append(np.zeros((self.grid_size, self.grid_size)))
            self.ants_with_food.append(np.zeros((self.grid_size, self.grid_size)))
            self.id_loc.append( np.zeros((self.grid_size, self.grid_size)) )
            self.col_deliv.append(0)
            antlst = []
            for m in range(self.ants_per_colony):
                antlst.append((m, 0))
            self.paths.append(dict(antlst))



#function for initializing ant colony positions, food source positions, obstacle positions
def init_model( grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail):

        x = AntModel( grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail )

        x.food_loc = ( random.randint(0, x.grid_size-1), random.randint(0, x.grid_size-1) )

        #initialize ant colonies
        for i in range(x.num_colonies):

            while True: #loop to generate a new colony home until it finds cell not already occupied by the food source or another colonies home
                home = ( random.randrange(x.grid_size), random.randrange(x.grid_size) )
                if home != x.food_loc and home not in x.colony_homes:
                    break
            x.colony_homes.append(home)
            ants_left_to_place = x.ants_per_colony

            while(ants_left_to_place):
                if x.ants_total[home] != 0: #if the cell is not empty, find a new one
                    home = ( random.randrange(x.grid_size), random.randrange(x.grid_size) )
                    continue
                if ants_left_to_place <= x.max_ants_per_cell:
                    x.ants[i][home] = ants_left_to_place
                    x.ants_total[home] = ants_left_to_place

                    break # when all the ants are placed, end the while loop
                else:
                    x.ants[i][home] = x.max_ants_per_cell
                    x.ants_total[home] = x.max_ants_per_cell
                    ants_left_to_place = ants_left_to_place - x.max_ants_per_cell

                #home = rand_direction(home) #get a random adjacent cell

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
def get_surrounding_cells(cell, Modelobj):

    tuple_retlst = []
    for x,y in [( cell[0]+i, cell[1]+j ) for i in (-1,0,1) for j in (-1,0,1) if i != 0 or j != 0]:
        if  0 <= x < Modelobj.grid_size and 0 <= y < Modelobj.grid_size:
            tuple_retlst.append((x, y))

    return tuple_retlst


def check_sums(Model, i, outstr ):

    if sum(sum(Model.ants_total)) != Model.num_colonies * Model.ants_per_colony:
        print('colony: ' + str(i) + '\n loc: ' + outstr)
        print('sum: ' + str(sum(sum(Model.ants_total))))
        print('total steps: ' + str(Model.total_steps))
        print('Ants total\n' + str(Model.ants_total))
        print('ants: ' + str(Model.ants[i]))

        raise ValueError(':(')
    return 0

#function for moving the simulation forward one timestep
def update_model(Model):

    food_deliveries = 0

    Model.total_steps += 1

    #update colonies one at a time
    for i in range(Model.num_colonies):

        check_sums(Model, i, 'VERY BEGINING')
        #get all cells with ants
        (rowi, coli) = np.nonzero(Model.ants_total)
        ant_i = list(zip(rowi, coli)) # list of tuples containing coordinates with ants

        for j in ant_i:

            neighbors = get_surrounding_cells(j, Model)

            check_sums(Model, i, 'b4 everything')
            #ANTS WITH FOOD
            if Model.ants_with_food[i][j] != 0:

                #FOOD ARRAY -> ANTS ARRAY
                if (Model.colony_homes[i] in neighbors):
                    chome = Model.colony_homes[i]
                    while (Model.ants_total[chome] < Model.max_ants_per_cell) and (Model.ants_with_food[i][j] != 0):
                        if Model.first_delivery:#This is where, if it's the first delivery, we grab the colony number to use for g_pheromones update
                            Model.first_d_col = i

                        Model.ants_total[j] -= 1
                        Model.ants_with_food[i][j] -= 1

                        Model.ants_total[chome] += 1
                        Model.ants[i][chome] += 1
                        food_deliveries += 1
                        Model.col_deliv[i] += 1
                    #continue #after filling home cell with as many as possible, end food ant movement since the colony home is within range

                check_sums(Model, i, 'FOOD -> ANTS!!')


                #FOOD ARRAY -> FOOD ARRAY

                f_probs_l = []
                f_probs_g = []
                food_ant_cells = []

                this_col_home = Model.colony_homes[i]

                f_heuristic_dist = [(math.sqrt(((this_col_home[0] - nei[0])**2) + ( (this_col_home[1] - nei[1]) ** 2))) for nei in neighbors]
                f_hmin = min(f_heuristic_dist)



                for cel in range(len(neighbors)):

                    heur =  3 - (f_heuristic_dist[cel] - f_hmin)#normalize the measurement
                    if Model.ants_total[neighbors[cel]] < Model.max_ants_per_cell: # only add a cell if there is space in the cell for ants to move into

                        f_probs_l.append( ((Model.pheromones[i][neighbors[cel]] + 1) ** Model.alpha) * ( heur ** Model.beta ) )

                        f_probs_g.append( ((Model.g_pheromones[neighbors[cel]] + 1) ** Model.alpha) * ( heur ** Model.beta ) )
                        food_ant_cells.append(neighbors[cel])

                #add local and global probs for the sum
                f_comb_vals = [ x + y for x,y in zip(f_probs_l, f_probs_g)]
                f_sumval = sum(f_comb_vals)
                #list to hold final probabilities for each cell option
                f_prob_list = []
                num_cells_to_calc = len(f_probs_l)
                for op in range(num_cells_to_calc):

                    f_prob_list.append( (Model.w*(f_probs_l[op]) + (1-Model.w)*(f_probs_g[op])) / f_sumval )

                check_sums(Model, i, 'Pre-FOOD -> FOOD!!')
                #update ant position
                fa_count = Model.ants_with_food[i][j]
                #print('facount: ' + str(fa_count))
                #print(Model.ants_total)
                #print(Model.ants_with_food[i])
                #print(Model.ants[i])

                if f_prob_list:

                    for an in range(int(fa_count)):

                        try:
                            choice = random.choices(food_ant_cells, weights=f_prob_list)
                        except:
                            print('random choice error. f_prob_list:  ')
                            print(f_prob_list)
                            print('food ant cells: ' + str(food_ant_cells))

                        if Model.ants_total[choice[0]] >= Model.max_ants_per_cell:#if the ants choice is full, do nothing
                            continue

                        Model.ants_with_food[i][choice[0]] += 1
                        Model.ants_total[choice[0]] += 1

                        Model.ants_with_food[i][j] -= 1
                        Model.ants_total[j] -= 1

                        #add pheromone trails when ants move, ants deposit pheromones apon ariving in a new cell, updates global and local pheromone values
                        Model.pheromones[i][choice[0]] += Model.pher_trail
                        #Model.g_pheromones[choice[0]] +=  Model.g_pher_trail

                    #print('J: ' + str(j))
                    check_sums(Model, i, 'FOOD -> FOOD!!')

            check_sums(Model, i, 'AFTER FOOD')
            ant_count = Model.ants[i][j]
            if ant_count == 0: # If there are no ants not carrying food, continue, skipping this itteration
                continue

            checksum = sum(sum(Model.ants_total))


            check_sums(Model, i, 'B4 FOOD -> ANTS2222222!!')
            #ANTS ARRAY -> FOOD ARRAY i.e. picking up a snack
            #Continue after each option, since if the food cell is full, we want to just wait and not do anything else
            if Model.food_loc in neighbors:
                if (Model.ants_total[Model.food_loc] + ant_count) <= Model.max_ants_per_cell:#if all ants fit into food cell

                    #print('b4\n' )
                    #print(Model.ants_total)
                    #print(Model.ants_with_food[i])
                    Model.ants_total[j] -= ant_count
                    Model.ants[i][j] = 0

                    Model.ants_with_food[i][Model.food_loc] = Model.ants_with_food[i][Model.food_loc] + ant_count
                    Model.ants_total[Model.food_loc] += ant_count
                    #print('ant count: ' + str(ant_count))
                    check_sums(Model, i, 'b4 continue 1')
                    continue
                elif Model.ants_total[Model.food_loc] < Model.max_ants_per_cell: #if only some will fit

                    room = Model.max_ants_per_cell - Model.ants_total[Model.food_loc] #space left in food cell

                    Model.ants_total[Model.food_loc] += room
                    Model.ants_with_food[i][Model.food_loc] += room

                    Model.ants[i][j] = ant_count - room
                    ant_count = ant_count - room
                    Model.ants_total[j] -= room

                    check_sums(Model, i, 'b4 continue 2')
                    continue
                else:#or if no ants will fit, pass
                    check_sums(Model, i, 'b4 continue 3')
                    continue
                check_sums(Model, i, 'ANTS -> FOOD!!')

            check_sums(Model, i, 'FOOD -> ANTS2222222!!')

            #FOODLESS ANT MOVEMENT
            vallst = []
            vallstg = []
            cellst = []

            heuristic_dist = [(math.sqrt(((Model.food_loc[0] - nei[0])**2) + ( (Model.food_loc[1] - nei[1]) ** 2))) for nei in neighbors] #get list of euclidean distances to food source
            hmin = min(heuristic_dist)

            #calc probability for each of j's neighbors
            for nei in range(len(neighbors)):

                heur =  3 - (heuristic_dist[nei] - hmin)#normalize the measurement
                if Model.ants_total[neighbors[nei]] < Model.max_ants_per_cell: # only add a cell if there is space in the cell for ants to move into

                    vallst.append( ((Model.pheromones[i][neighbors[nei]] + 1) ** Model.alpha) * ( heur ** Model.beta ) )

                    vallstg.append( ((Model.g_pheromones[neighbors[nei]] + 1) ** Model.alpha) * ( heur ** Model.beta ) )
                    cellst.append(neighbors[nei])


            #NO MOVEMENT OPTIONS == SKIP ANTS
            if not vallst or not cellst:
                continue

            #get list of combined values
            comb_vals = [ x + y for x,y in zip(vallst, vallstg)]
            #print('======================================')
            #print(comb_vals)

            sumval = sum(comb_vals)
            #sumvalg = sum(vallstg)

            check_sums(Model, i, 'b4 Ants -> Ants')

            prob_list = []
            num_cells_to_calc = len(vallst)
            for op in range(num_cells_to_calc):

                prob_list.append( (Model.w*(vallst[op]) + (1-Model.w)*(vallstg[op])) / sumval )

            #update ant position
            for an in range(int(ant_count)):

                try:
                    choice = random.choices(cellst, weights=prob_list)
                except:
                    print('random choice error. prob_list:  ')
                    print(prob_list)

                if Model.ants_total[choice[0]] >= Model.max_ants_per_cell:#if the ants choice is full, do nothing
                    check_sums(Model, i, 'CONTINUE-normal-1')
                    continue

                Model.ants[i][choice[0]] += 1
                Model.ants_total[choice[0]] += 1

                Model.ants[i][j] -= 1
                Model.ants_total[j] -= 1

                #add pheromone trails when ants move, ants deposit pheromones apon ariving in a new cell, updates global and local pheromone values
                Model.pheromones[i][choice[0]] += Model.pher_trail
                #Model.g_pheromones[choice[0]] +=  Model.g_pher_trail
                check_sums(Model, i, 'CONTINUE-normal-2')

            check_sums(Model, i, 'After ANTS -> ANTS')

    check_sums(Model, i, 'After position updates')
    #update pheromone trails at the same time, after ants have moved
    for i in range(Model.num_colonies):
        Model.pheromones[i] = Model.pheromones[i] * Model.evap
        #Model.g_pheromones = Model.g_pheromones * Model.g_evap

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

                Model.g_pheromones[start] += Model.g_pher_trail
                if step_diff > 0: #add corner of path to account for potential step difference
                    step_diff -= 1
                    Model.g_pheromones[(start[0], (start[1] - int(distance_t[1]/abs(distance_t[1]))))] += Model.g_pher_trail
                distance_t = ( (distance_t[0] - int(distance_t[0] / abs(distance_t[0]))) , (distance_t[1] - int(distance_t[1] / abs(distance_t[1]))) )

            elif abs(distance_t[0]) > abs(distance_t[1]):
                start = ( int(start[0] + int(distance_t[0] / abs(distance_t[0])) ), start[1])

                Model.g_pheromones[start] += Model.g_pher_trail
                distance_t = ( (distance_t[0] - int(distance_t[0] / abs(distance_t[0]))), distance_t[1] )

            elif abs(distance_t[0]) < abs(distance_t[1]):
                start = ( start[0], int(start[1] + int(distance_t[1] / abs(distance_t[1]))) )

                Model.g_pheromones[start] += Model.g_pher_trail
                distance_t = ( distance_t[0], (distance_t[1] - int(distance_t[1] / abs(distance_t[1]))) )
            else:
                print('This should never print')

    check_sums(Model, i, 'after pher')

    return food_deliveries


'''
Parameter order for model init so i dont have to scroll all the way back up
grid_size, num_cols, max_oc, ants_per, alpha, beta, w, evap, trail
'''
#best values kept from parameter sweepss

'''
alpha = 1
beta = 1


paramlst = [ 10, 20, 30, 40, 50 ]

best = 0
best_score = 0

best_avg_m = 0
best_avg_score = 0

for m in paramlst:


    vals = []
    for pe in range(3):
        ModelObj = init_model(m, 3, 5, 50, alpha, beta, .5, .9, 1)
        delivery_total = 0
        for ex in range(1000):
            delivery_total += update_model(ModelObj)
        if delivery_total < 200:
            print('Delivery small, m: ' + str(m))
            print('food_loc: ' + str(ModelObj.food_loc))
            print('colony homes: ' + str(ModelObj.colony_homes))
            print(ModelObj.ants_total)

        vals.append(delivery_total)

    print( str(m) + '  ' + str(max(vals)) + '     ' + str(vals))

    score = sum(vals)/len(vals)

    if max(vals) > best_score:
        best_score = max(vals)
        best = m

    if score > best_avg_score:
        best_avg_score = score
        best_avg_m = m

print('Best: ' + str(best))
print(best_score)

print('best avg: ' + str(best_avg_m))
print(best_avg_score)


'''











'''m'''
