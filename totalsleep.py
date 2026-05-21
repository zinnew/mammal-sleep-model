import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess
df = preprocess('sleep.txt')


#tabell med gjennomsnitt, median, std, max og min for alle kolonnene i df
total_sleep = pd.DataFrame({
    'mean': df.mean(numeric_only=True), 
    'median': df.median(numeric_only=True), 
    'std': df.std(numeric_only=True),
    'max': df.max(numeric_only=True),
    'min': df.min(numeric_only=True)
})

print(f'Tabell med gjennomsnitt, median, std, max og min for alle kolonnene i df:\n{total_sleep}')


#histogram 
plt.hist(df['TotalSleep'], bins=15)
plt.xlabel('Total Sleep (hours)')
plt.ylabel('arter')
plt.title('Histogram of Total Sleep')

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axes = axs.flatten()

axs[0,0].hist(df['BodyWt'], bins=15)
axs[0,0].set_xlabel('Body Weight (kg)')

axs[0,1].hist(df['BrainWt'], bins=15)
axs[0,1].set_xlabel('Brain Weight (g)')

axs[1, 0].hist(df['LifeSpan'], bins=15)
axs[1, 0].set_xlabel('Life Span (years)')

axs[1, 1].hist(df['Gestation'], bins=15)
axs[1, 1].set_xlabel('Gestation (days)')

for ax in axes: 
    ax.set_ylabel('antall arter')
plt.tight_layout()
plt.show()


#spredningsdiagram som viser sammenhengen mellom utvalgte variabler og TotalSleep
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0,0].scatter(df['BodyWt'], df['TotalSleep'], alpha=0.7)
axs[0,0].set_xlabel('Body Weight (kg)')
axs[0,0].set_xscale('log')  # Logaritmisk skala for bedre visualisering

axs[0,1].scatter(df['BrainWt'], df['TotalSleep'], alpha=0.7)
axs[0,1].set_xlabel('Brain Weight (g)')
axs[0,1].set_xscale('log')  # Logaritmisk skala for bedre visualisering

axs[1,0].scatter(df['LifeSpan'], df['TotalSleep'], alpha=0.7)
axs[1,0].set_xlabel('Life Span (years)')

axs[1,1].scatter(df['Gestation'], df['TotalSleep'], alpha=0.7)
axs[1,1].set_xlabel('Gestation (days)')

plt.tight_layout()
plt.show()
