#%%
from pandas import read_csv
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import statsmodels.api as sm

plot_acf = sm.graphics.tsa.plot_acf
plot_pacf = sm.graphics.tsa.plot_pacf

#%%
data = read_csv("res-mgmt-0-s-0.8-0.csv", header=0, usecols=["inflow", "storage"])
data["D.storage"] = data["storage"].diff()

# mod = SARIMAX(
#     endog=data["storage"],
#     exog=data["inflow"],
#     trend="c",
#     order=(1, 1, 1),
#     enforce_stationarity=False,
#     enforce_invertibility=False,
# )
mod = SARIMAX(
    data['storage'],
    order=(4, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    simple_differencing=True,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
res = mod.fit(cov_type='oim')
print(res.summary())

#%%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fig = plot_acf(data.iloc[1:]["D.storage"], lags=40, ax=axes[0])
fig = plot_pacf(data.iloc[1:]["D.storage"], lags=40, ax=axes[1])

plt.show()

#%%
from statsmodels.tsa.stattools import adfuller

print(adfuller(data['D.storage']))

#%%

model = SARIMAX(
    endog=data["storage"], 
    order=(0, 1, 2),
    seasonal_order=(0, 1, 3, 12),
    )
results = model.fit()
print(results.llf)
print(results.summary())

#%%
results.resid.plot()
plt.show()
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
fig = plot_acf(results.resid, lags=100, ax=ax1)
ax2 = fig.add_subplot(122)
fig = plot_pacf(results.resid, lags=100, ax=ax2)
plt.show()
#%%
