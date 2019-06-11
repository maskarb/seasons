# grid search sarima hyperparameters for monthly car sales dataset
from copy import deepcopy
from itertools import product
from math import sqrt
from multiprocessing import cpu_count
from random import choice, randint, random, sample
from warnings import catch_warnings, filterwarnings

from joblib import Parallel, delayed
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

inf = float("inf")
POPULATION_SIZE = 100
SELECT_FITTEST_PROBABILITY = 0.95
MUTATION_PROBABILITY = 0.008
CROSSOVER_PROBABILITY = 0.75
PARAM_DICT = {
    0: range(5),
    1: range(2),
    2: range(5),
    3: range(5),
    4: range(2),
    5: range(5),
    6: range(0, 49, 6),
    7: ["n", "c", "t", "ct"],
}

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=sorder,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg_dict, cfg, debug=False):
    result = cfg_dict.get(cfg, None)
    # show all warnings and fail on exception if debugging
    if debug and result is None:
        result = walk_forward_validation(data, n_test, cfg)
        cfg_dict[cfg] = result
    elif result is None:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
                cfg_dict[cfg] = result
        except:
            error = None
    print(f"Model: {cfg} {result}")
    return (cfg, result)


# grid search configs
def grid_search(data, cfg_list, cfg_dict, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
        tasks = (delayed(score_model)(data, n_test, cfg_dict, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg_dict, cfg) for cfg in cfg_list]
    return_scores = []
    for r in scores:
        if r[1] is not None:
            return_scores.append(r)
        else:
            cfg_dict[r[0]] = inf
    # remove empty results
    # scores = [r for r in scores if r[1] != None]
    # return scores
    return return_scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = PARAM_DICT[0]
    d_params = PARAM_DICT[1]
    q_params = PARAM_DICT[2]
    P_params = PARAM_DICT[3]
    D_params = PARAM_DICT[4]
    Q_params = PARAM_DICT[5]
    seasonal = PARAM_DICT[6]
    t_params = PARAM_DICT[7]
    # create config instances
    for x in product(p_params, d_params, q_params):
        for y in product(P_params, D_params, Q_params, seasonal):
            for t in t_params:
                cfg = (x, y, t)
                models.append(cfg)
    cfg_dict = {model: None for model in models}
    return models, cfg_dict


def pick_winner(cfg_x, cfg_y):
    select_fittest = random() < SELECT_FITTEST_PROBABILITY
    x_is_better = (cfg_x)[1] < (cfg_y)[1]
    return cfg_x[0] if select_fittest == x_is_better else cfg_y[0]


def selection(population):
    return [
        deepcopy(pick_winner(choice(population), choice(population)))
        for _ in population
    ]


# modifies: x and y
def crossover_uniform(x, y):
    x, y = list(x), list(y)
    for i in range(len(x)):
        if randint(0, 1) == 0:
            x[i], y[i] = y[i], x[i]
    return tuple(x), tuple(y)


# modifies: x
def mutate(x):
    temp1, temp2, temp3 = x
    temp = list(temp1) + list(temp2) + [temp3]
    for i in range(len(temp)):
        if random() < MUTATION_PROBABILITY:
            temp[i] = mut_vals(i)
    return (tuple(temp[0:3]), tuple(temp[3:7]), temp[7])


def mut_vals(i):
    lis = PARAM_DICT[i]
    return choice(lis)


def get_fittest(p):
    lis = sorted(p, key=lambda tup: tup[1])
    return lis[0:3]


def re_add_incumbent(p, incumbent):
    print(" > Incumbent: %s %.3f" % incumbent)
    p.add(incumbent[0])


def run(data, n_test, config_list, cfg_dict, max_generations, parallel=False):
    p = sample(config_list, POPULATION_SIZE)
    print("initial scores")
    scores = grid_search(data, p, cfg_dict, n_test, parallel)
    incumbent = get_fittest(scores)
    for i in range(1, max_generations + 1):
        print(f"generation {i} start.")
        q = selection(scores)
        for j in range(0, len(q) - 1, 2):
            if random() < CROSSOVER_PROBABILITY:
                q[j], q[j + 1] = crossover_uniform(q[j], q[j + 1])
        for j in range(len(q)):
            q[j] = mutate(q[j])
        p = set(q)
        if len(p) < POPULATION_SIZE:
            p.update(sample(config_list, POPULATION_SIZE - len(p)))
        if incumbent[1] != get_fittest(scores)[0][1]:
            incumbent = get_fittest(scores)[0]
            re_add_incumbent(p, incumbent)
        scores = grid_search(data, p, cfg_dict, n_test, parallel)
        print(f"generation {i} complete.")
    x = get_fittest(scores)
    return x


if __name__ == "__main__":
    # load dataset
    df = read_csv("res-mgmt-0-s-0.8-0.csv", header=0, usecols=["storage"])
    data = df.values

    # series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
    # data = series.values
    print(data.shape)
    # data split
    n_test = 12
    # model configs
    cfg_list, cfg_dict = sarima_configs(seasonal=[0, 6, 12])
    # grid search
    # scores = grid_search(data, cfg_list, n_test)
    # ga search
    scores = run(data, n_test, cfg_list, cfg_dict, 10)
    print("done")
    # list top 3 configs
    for score in scores:
        cfg, error = score
        print(cfg, error)
