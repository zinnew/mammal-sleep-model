import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))

from forreg import model_tot, model_rem

#prediksjoner for mennesker 
new_data_rem = pd.DataFrame({
    'BodyWt': [62],
    'BrainWt': [1320],
    'LifeSpan': [100],
    'Gestation': [267],
    'Exposure': [1],
    'Danger': [1]
})

new_data_tot = pd.DataFrame({
    "Exposure": [1],
    "Gestation": [267],
    "Danger": [1],
    "Predation": [1],
    "BodyWt": [62]
})

new_data_tot = sm.add_constant(new_data_tot, has_constant='add')
new_data_rem = sm.add_constant(new_data_rem, has_constant='add')

prediction_tot = model_tot.predict(new_data_tot)
prediction_rem = model_rem.predict(new_data_rem)

print(f'Prediksjon for TotalSleep: {prediction_tot[0]:.4f} timer')
print(f'Prediksjon for REM sleep: {prediction_rem[0]:.4f} timer')