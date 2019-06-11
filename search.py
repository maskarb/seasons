from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings

from joblib import Parallel, delayed
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# one-step sarima forecast
def sarima_forecast(endog, config):
    order, sorder = config
    # define model
    model = SARIMAX(endog, order=order, seasonal_order=sorder)
    # fit model
    model_fit = model.fit(disp=False)
    return model_fit.llf

# score a model, return None on failure
def score_model(data, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = sarima_forecast(data, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = sarima_forecast(data, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(" > Model[%s] %.3f" % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
        tasks = (delayed(score_model)(data, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0]
    d_params = [2]
    q_params = [1]
    P_params = [0]
    D_params = [2]
    Q_params = [2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for P in P_params:
                    for D in D_params:
                        for Q in Q_params:
                            for m in m_params:
                                cfg = [(p, d, q), (P, D, Q, m)]
                                models.append(cfg)
    return models


if __name__ == "__main__":
    # load dataset
    series = read_csv("res-mgmt-0-s-0.8-0.csv", header=0, usecols=["storage"])
    data = series.values
    print(data.shape)
    # model configs
    cfg_list = sarima_configs(seasonal=list(range(37)))
    # grid search
    scores = grid_search(data, cfg_list)
    print("done")
    # list top 3 configs
    for cfg, error in scores[-3:]:
        print(cfg, error)
