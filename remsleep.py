import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))

#tabell med gjennomsnitt, median, std, max og min for REM sleep
rem_sleep = pd.DataFrame({
    'mean': df['REM'].mean(), 
    'median': df['REM'].median(), 
    'std': df['REM'].std(),
    'max': df['REM'].max(),
    'min': df['REM'].min()
}, index=['REM'])

print(f'Tabell med gjennomsnitt, median, std, max og min for REM sleep:\n{rem_sleep}')


#histogram
plt.hist(df['REM'], bins=15)
plt.xlabel('andel drømmesøvn')
plt.ylabel('antall arter')
plt.title('Histogram of REM Sleep')
plt.show()


#spredningsdiagram som viser sammenhengen mellom utvalgte variabler og REM sleep
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].scatter(df['BodyWt'], df['REM'], alpha=0.7)
axs[0,0].set_xlabel('Body Weight (kg)')
axs[0,0].set_xscale('log')  # Logaritmisk skala for bedre visualisering

axs[0,1].scatter(df['BrainWt'], df['REM'], alpha=0.7)
axs[0,1].set_xlabel('Brain Weight (g)')
axs[0,1].set_xscale('log')  # Logaritmisk skala for bedre visualisering

axs[1,0].scatter(df['LifeSpan'], df['REM'], alpha=0.7)
axs[1,0].set_xlabel('Life Span (years)')

axs[1,1].scatter(df['Gestation'], df['REM'], alpha=0.7)
axs[1,1].set_xlabel('Gestation (days)')

for ax in axs.flat:
    ax.set_ylabel('andel drømmesøvn')
    ax.grid()
plt.tight_layout()
plt.show()

