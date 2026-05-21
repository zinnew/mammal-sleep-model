import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))


#enkel lineær regresjonsmodell for utvalgte forkalringsvariabler med TotalSleep 
for var in ['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation']: 
    X = df[var]
    y = df['TotalSleep']
    
    # Legg til konstant for intercept
    X_const = sm.add_constant(X)
    
    # Pass på at X_const og y har samme lengde
    X_const = X_const.iloc[:len(y)]
    y = y.iloc[:len(X_const)]
    
    model = sm.OLS(y, X_const).fit()
    
    print(f'Enkel lineær regresjonsmodell for {var} med TotalSleep:\n{model.summary()}\n')


#enkel lineær regresjonsmodell for utvalgte forkalringsvariabler med REM sleep
for var in ['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation']: 
    X = df[var]
    y = df['REM']
    
    # Legg til konstant for intercept
    X_const = sm.add_constant(X)
    
    # Pass på at X_const og y har samme lengde
    X_const = X_const.iloc[:len(y)]
    y = y.iloc[:len(X_const)]
    
    model = sm.OLS(y, X_const).fit()
    
    print(f'Enkel lineær regresjonsmodell for {var} med REM sleep:\n{model.summary()}\n')