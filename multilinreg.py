import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))


#multippel lineær regresjonsmodell for alle forkalringsvariabler med TotalSleep
X = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
y = df['TotalSleep']

# Legg til konstant for intercept
X_const = sm.add_constant(X)

model_tot = sm.OLS(y, X_const).fit()
print(f'Multippel lineær regresjonsmodell for alle forkalringsvariabler med TotalSleep:\n{model_tot.summary()}\n')


#multippel lineær regresjonsmodell for alle forkalringsvariabler med REM sleep
X = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
y = df['REM']

# Legg til konstant for intercept
X_const = sm.add_constant(X)

model_rem = sm.OLS(y, X_const).fit()
print(f'Multippel lineær regresjonsmodell for alle forkalringsvariabler med REM sleep:\n{model_rem.summary()}\n')