MAX_COST = 99999
MIN_TABU_SIZE = 50
MAX_TABU_SIZE = 52
TABU_STEP = 1
NUM_OF_TESTS = 3
N_ITERS = 5000
FILE_NAME = "C:/Users/Mohamed/Desktop/PFE M2/test_data.txt"
NUM_DRONES = 8
DRONE_CAPACITY = 4
NUM_CLIENTS = 30

# Structures for storing history
best_of_all = MAX_COST
size = MIN_TABU_SIZE
costs_history = []
best_costs_history = []
fitness_history = []
tabu_size_average_costs = {}
best_model_result = None
best_tabu_size = None

# Main loop for testing parameters
while size <= MAX_TABU_SIZE:
    for _ in range(NUM_OF_TESTS):
        ts = TabuSearch(NUM_DRONES, DRONE_CAPACITY, NUM_CLIENTS, FILE_NAME)
        ts.search(tabu_size=size, n_iters=N_ITERS)
        best_cost = ts._fitness(ts.best_solution)
        print(f'Tabu size: {size}, Best_cost: {best_cost}')
        fitness_history.append(ts.best_cost)
        best_costs_history.append(ts.best_costs)
        costs_history.append(ts.costs)
        if best_cost < best_of_all:
            best_of_all = best_cost
            best_tabu_size = size
            best_model_result = ts
    hist_arr = np.array(fitness_history)
    tabu_size_average_costs[size] = hist_arr.mean()
    fitness_history = []
    print('=========================')
    print(f'Average: {tabu_size_average_costs[size]}')
    print(f'Best: {hist_arr.min()}\n')
    size += TABU_STEP
print(f"BEST COST: {best_of_all} | TABU SIZE: {best_tabu_size} | NUM OF ITERS: {N_ITERS}")