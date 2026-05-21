import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))


#stegvis regresjon backward 
def stegvis_reg_back(X, y): 
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    adj_r2 = model.rsquared_adj
    print(f'Justert R²: {adj_r2:.4f}')
    print(f'p-verdier:\n{model.pvalues.iloc[1:]}\n')
    
    while True: 
        pvalues = model.pvalues.drop('const', errors='ignore')
        pvalues = pvalues.sort_values(ascending=False)

        if pvalues.empty: 
            break

        feature_removed = False

        #itererer gjennom listen
        for feature in pvalues.index: 
            current_pval = pvalues[feature]
            tempX = X.drop(columns=[feature])

            #kjører modellen
            temp_model = sm.OLS(y, tempX).fit()
            temp_adj_r2 = temp_model.rsquared_adj

            #sjekker om fjerning forbedret modellen
            if temp_adj_r2 >= adj_r2: 
                print(f'fjerner {feature} med p={current_pval:.4f}')
                print(f'ny justert R²: {temp_adj_r2:.4f}\n')
                adj_r2 = temp_adj_r2
                model = temp_model
                X = tempX
               
                feature_removed = True
                break
                #starter while løkken på nytt
            #hvsi det ikke blir bedre går løkken videre til neste i listen
            else: 
                pass

        if not feature_removed: 
            print('ingen av de gjenværende variablene kan fjernes uten å redusere modellens ytelse ')
            break

    print(f'endelig modell: {list(X.columns)}')
    return model

X = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
y_tot = df['TotalSleep']
y_rem = df['REM']

model_tot = stegvis_reg_back(X.copy(), y_tot)
model_rem = stegvis_reg_back(X.copy(), y_rem)

print(f'TotalSleep:\n {model_tot.summary()}\n')
print(f'REM sleep:\n {model_rem.summary()}\n')