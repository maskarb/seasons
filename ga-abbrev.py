
# ga.py -- a simple GA that solves a knapsack problem

from numpy import average, array, copy, empty, ndarray, random
from random import choice
from typing import List, Tuple

## GA parameters

POPULATION_SIZE = 100
SELECT_FITTEST_PROBABILITY = 0.95
MUTATION_PROBABILITY = 0.008
CROSSOVER_PROBABILITY = 0.75
# crossover = crossover_uniform

## Example knapsack problem

PENALTY = 4
MAX_COST = 200

BA = array([ 1,  13, 12, 19,  1,  6,  6, 15,  8,  4,
             5,  12, 11, 18,  9,  9, 12, 16,  2,  8,
             11,  3,  9, 17,  2, 16,  2,  5,  3, 17,
             11, 10, 13, 18,  2,  7, 13, 11, 10, 16])

CA = array([ 7,   3,  5,  7,  9,  1, 18, 12, 12, 19,
             1,  18, 12, 13, 16,  6, 17,  6, 15, 16,
             10, 10, 10,  6, 11, 16, 10,  5,  1, 12,
             4,  19,  6,  4, 13,  1, 17, 11,  9, 11])

X = array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
           1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
           1, 0, 1, 1, 0, 1, 1, 1, 1, 1])

def fitness(x: ndarray) -> int:
    b, c = x.dot(BA), x.dot(CA)
    return b - PENALTY * max(c - MAX_COST, 0)

def random_binary_array(n: int) -> ndarray:
    return random.randint(2, size=n)

# modifies: x and y
def crossover_uniform(x: ndarray, y: ndarray) -> None:
    for i in range(len(x)):
        if random.randint(2) == 0:
            x[i], y[i] = y[i], x[i]

# modifies: x
def mutate(x: ndarray) -> None:
    for i in range(len(x)):
        if random.random() < MUTATION_PROBABILITY:
            x[i] = 1 - x[i]

def pick_winner(x: ndarray, y: ndarray) -> ndarray:
    select_fittest = random.random() < SELECT_FITTEST_PROBABILITY
    x_is_better = fitness(x) > fitness(y)
    return x if select_fittest == x_is_better else y

def selection(p: List[ndarray]) -> List[ndarray]:
    return [copy(pick_winner(choice(p), choice(p))) for _ in p]

def make_random_population(pop_size: int, org_size: int) -> List[ndarray]:
    return [random_binary_array(org_size) for _ in range(pop_size)]

def get_fittest(p: List[ndarray]) -> ndarray:
    return max(p, key=fitness)

def run(max_generations: int) -> Tuple[ndarray, int]:
    p = make_random_population(POPULATION_SIZE, len(BA))
    # incumbent = get_fittest(p)
    for i in range(1, max_generations + 1):
        q = selection(p)
        for j in range(0, len(q)-1, 2):
            if random.random() < CROSSOVER_PROBABILITY:
                crossover_uniform(q[j], q[j+1])
        for j in range(len(q)):
            mutate(q[j])
        p = q
    x = get_fittest(p)
    return x, fitness(x)

def main():
    b, c = X.dot(BA), X.dot(CA)
    print(b, c)
    print(fitness(X))

    x, fit = run(50)
    print(x.dot(BA), x.dot(CA))
    print(fit)

if __name__ == '__main__':
    main()
