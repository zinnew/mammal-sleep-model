import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))


def stegvis_reg_for(X, y, features):
    Xinit = sm.add_constant(df[features])
    model = sm.OLS(y, Xinit[X]).fit()
    adj_r2 = model.rsquared_adj
    print(f'Justert R²: {adj_r2:.4f}\n')

    while True: 
        best_feature = None
        best_adj_r2 = adj_r2

        for feature in features: 
            if feature not in X: 
                tempX = X + [feature] #midlertidig liste med variabler 

                #kjør modellen 
                temp_model = sm.OLS(y, Xinit[tempX]).fit()
                temp_adj_r2 = temp_model.rsquared_adj

                #sjekker om denne gir høyest score 
                if temp_adj_r2 > best_adj_r2: 
                    best_adj_r2 = temp_adj_r2
                    best_feature = feature
        
        #sjekker om modellen ble bedre
        if best_feature is not None: 
            X.append(best_feature)
            adj_r2 = best_adj_r2
            print(f'la til {best_feature} med justert R²: {adj_r2:.4f}\n')
        else: 
            print(f'ingen flere variabler forbedre modellen')
            break
    
    print(f'endelig modell: {X}')
    print(f'endelig justert R²: {adj_r2:.4f}')
    model = sm.OLS(y, Xinit[X]).fit()
    return model

features=['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']
X = ['const']
y_tot = df['TotalSleep']
y_rem = df['REM']

model_tot = stegvis_reg_for(X.copy(), y_tot, features)
print(f'Stegvis regresjon for TotalSleep:\n {model_tot.summary()} \n')

model_rem = stegvis_reg_for(X.copy(), y_rem, features)
print(f'Stegvis regresjon for REM sleep:\n {model_rem.summary()} \n')