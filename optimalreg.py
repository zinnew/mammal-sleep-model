import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import statsmodels.api as sm

from preprocessing import preprocess
df = preprocess('sleep.txt')

#ny variable for rem sleep
df['REM'] = ((df['Dreaming'] / df['TotalSleep']))


#alle modeller
X = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
X = sm.add_constant(X)
y_tot = df['TotalSleep']
y_rem = df['REM']
model_tot = sm.OLS(y_tot, X).fit()
model_rem = sm.OLS(y_rem, X).fit()

#excluding Danger
X_exdanger = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure']]
X_exdanger = sm.add_constant(X_exdanger)
model_exdanger_tot = sm.OLS(y_tot, X_exdanger).fit()
model_exdanger_rem = sm.OLS(y_rem, X_exdanger).fit()

#excluding Exposure
X_exexposure = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Danger']]
X_exexposure = sm.add_constant(X_exexposure)
model_exexposure_tot = sm.OLS(y_tot, X_exexposure).fit()
model_exexposure_rem = sm.OLS(y_rem, X_exexposure).fit()

#excluding Predation
X_expredation = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation', 'Exposure', 'Danger']]
X_expredation = sm.add_constant(X_expredation)
model_expredation_tot = sm.OLS(y_tot, X_expredation).fit()
model_expredation_rem = sm.OLS(y_rem, X_expredation).fit()

#excluding Gestation
X_exgestation = df[['BodyWt', 'BrainWt', 'LifeSpan', 'Predation', 'Exposure', 'Danger']]
X_exgestation = sm.add_constant(X_exgestation)
model_exgestation_tot = sm.OLS(y_tot, X_exgestation).fit()
model_exgestation_rem = sm.OLS(y_rem, X_exgestation).fit()

#excluding LifeSpan
X_exlifespan = df[['BodyWt', 'BrainWt', 'Gestation', 'Predation', 'Exposure', 'Danger']]
X_exlifespan = sm.add_constant(X_exlifespan)
model_exlifespan_tot = sm.OLS(y_tot, X_exlifespan).fit()
model_exlifespan_rem = sm.OLS(y_rem, X_exlifespan).fit()

#excluding BrainWt
X_exbrainwt = df[['BodyWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
X_exbrainwt = sm.add_constant(X_exbrainwt)
model_exbrainwt_tot = sm.OLS(y_tot, X_exbrainwt).fit()
model_exbrainwt_rem = sm.OLS(y_rem, X_exbrainwt).fit()

#excluding BodyWt
X_exbodywt = df[['BrainWt', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger']]
X_exbodywt = sm.add_constant(X_exbodywt)
model_exbodywt_tot = sm.OLS(y_tot, X_exbodywt).fit()
model_exbodywt_rem = sm.OLS(y_rem, X_exbodywt).fit()


#tabell med adj R for alle modellene
models_tot = {
    'full model': model_tot.rsquared_adj,
    'excluding danger': model_exdanger_tot.rsquared_adj,
    'excluding exposure': model_exexposure_tot.rsquared_adj,
    'excluding predation': model_expredation_tot.rsquared_adj,
    'excluding gestation': model_exgestation_tot.rsquared_adj,
    'excluding lifespan': model_exlifespan_tot.rsquared_adj,
    'excluding brainwt': model_exbrainwt_tot.rsquared_adj,
    'excluding bodywt': model_exbodywt_tot.rsquared_adj
}

models_rem = {
    'full model': model_rem.rsquared_adj,
    'excluding danger': model_exdanger_rem.rsquared_adj,
    'excluding exposure': model_exexposure_rem.rsquared_adj,
    'excluding predation': model_expredation_rem.rsquared_adj,
    'excluding gestation': model_exgestation_rem.rsquared_adj,
    'excluding lifespan': model_exlifespan_rem.rsquared_adj,
    'excluding brainwt': model_exbrainwt_rem.rsquared_adj,
    'excluding bodywt': model_exbodywt_rem.rsquared_adj
}

adj_r2_tot_df = pd.DataFrame(list(models_tot.items()), columns=['Modell', 'Adjusted R-squared'])
adj_r2_rem_df = pd.DataFrame(list(models_rem.items()), columns=['Modell', 'Adjusted R-squared'])

print(f'Tabell med adj R for alle modellene med TotalSleep:\n{adj_r2_tot_df}\n')
print(f'Tabell med adj R for alle modellene med REM sleep:\n{adj_r2_rem_df}\n')