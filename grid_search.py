# grid search sarima hyperparameters for monthly car sales dataset
from math import sqrt
from multiprocessing import cpu_count
from random import choice, randint, random, sample
from copy import deepcopy
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

POPULATION_SIZE = 50
SELECT_FITTEST_PROBABILITY = 0.95
MUTATION_PROBABILITY = 0.008
CROSSOVER_PROBABILITY = 0.75
PARAM_DICT = {0: [0, 1, 2], 
              1: [0, 1], 
              2: [0, 1, 2], 
              3: [0, 1, 2],
              4: [0, 1],
              5: [0, 1, 2],
              6: [0,6,12],
              7: ['n','c','t','ct'] }

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
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
    # convert config to a key
    key = str(cfg)
    result = cfg_dict.get(key, None)
    # show all warnings and fail on exception if debugging
    if debug and result is None:
        result = walk_forward_validation(data, n_test, cfg)
        cfg_dict[key] = result
    elif result is None:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
                cfg_dict[key] = result
        except:
            error = None
    return (cfg, result)

# grid search configs
def grid_search(data, cfg_list, cfg_dict, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg_dict, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg_dict, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    cfg_dict = {str(model): None for model in models}
    return models, cfg_dict

def pick_winner(cfg_x, cfg_y):
    select_fittest = random() < SELECT_FITTEST_PROBABILITY
    x_is_better = (cfg_x)[1] < (cfg_y)[1]
    return cfg_x[0] if select_fittest == x_is_better else cfg_y[0]

def selection(population):
    return [deepcopy(pick_winner(choice(population), choice(population))) for _ in population]

# modifies: x and y
def crossover_uniform(x, y):
    for i in range(len(x)):
        if randint(0, 1) == 0:
            x[i], y[i] = y[i], x[i]

# modifies: x
def mutate(x):
    temp1, temp2, temp3 = x
    temp = list(temp1) + list(temp2) + list(temp3)
    for i in range(len(temp)):
        if random() < MUTATION_PROBABILITY:
            temp[i] = mut_vals(i)
    return [(temp[0:3]), (temp[3:7]), temp[7]]

def mut_vals(i):
    lis = PARAM_DICT[i]
    return choice(lis)

def get_fittest(p):
    lis = sorted(p, key=lambda tup: tup[1])
    return lis[0:3]

def re_add_incumbent(p, incumbent):
    print(' > Incumbent: %s %.3f' % incumbent)
    configs = [str(r) for r in p]
    if str(incumbent[0]) not in configs:
        p.append(incumbent[0])

def run(data, n_test, config_list, cfg_dict, max_generations, parallel=True):
    p = sample(config_list, POPULATION_SIZE)
    print("initial scores")
    scores = grid_search(data, p, cfg_dict, n_test, parallel)
    incumbent = get_fittest(scores)
    for i in range(1, max_generations + 1):
        print(f"generation {i} start.")
        q = selection(scores)
        for j in range(0, len(q)-1, 2):
            if random() < CROSSOVER_PROBABILITY:
                crossover_uniform(q[j], q[j+1])
        for j in range(len(q)):
            q[j] = mutate(q[j])
        p = q
        if len(p) < POPULATION_SIZE:
            p.extend(sample(config_list, POPULATION_SIZE - len(p)))
        if incumbent[1] != get_fittest(scores)[0][1]:
            incumbent = get_fittest(scores)[0]
            re_add_incumbent(p, incumbent)
        scores = grid_search(data, p, cfg_dict, n_test, parallel)
        print(f"generation {i} complete.")
    x = get_fittest(scores)
    return x


if __name__ == '__main__':
    # load dataset
    series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
    data = series.values
    print(data.shape)
    # data split
    n_test = 12
    # model configs
    cfg_list, cfg_dict = sarima_configs(seasonal=[0,6,12])
    # grid search
    # scores = grid_search(data, cfg_list, n_test)
    # ga search
    scores = run(data, n_test, cfg_list, cfg_dict, 5)
    print('done')
    # list top 3 configs
    for score in scores:
        cfg, error = score
        print(cfg, error)