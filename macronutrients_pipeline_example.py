#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:58:35 2021

@author: carlo_mengucci
"""


import numpy as np
import pandas as pd
from os.path import join as pj

import seaborn as sns
import pylab as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import matplotlib

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict


# %%
data_dir=Path('path/to/Data').resolve()
results_dir=Path('path/to/results').resolve()

# %% ## Loading and cleaning data

data=pd.read_excel(pj(data_dir, 'macro_dataset'), header=0, sep=',', #### load data as pandas dataframe, set sample IDs as index
                             index_col='Subject_ID')

data.dropna(axis='columns', how='any', inplace=True)

data=data[data['report_bias'] == 0] ### dropping data from biased questionaires 

# %% ### Select features and target columns of the dataset (continuous and categorical) 

feats = [column for column in data.columns if 'delta' in column]

feats_cat = [column for column in data.columns if '_cat' in column
           and 'T0' not in column]

targets = [column for column in data.columns if '_T0' in column 
           and '_cat' not in column]

targets_cat = [column for column in data.columns if '_T0' in column and '_cat'
               in column]

# %% compute and plot feature/ target correlation heatmap ###

corr_col = feats+targets

corr_map = data.loc[:,corr_col].corr(method = 'spearman')

sns.set(font_scale=1.4)

f, ax = plt.subplots (figsize = (14,14))
sns.heatmap(data = corr_map, cmap = 'vlag', ax = ax)
ax.tick_params(labelsize = 18)
f.savefig(pj(results_dir, 'corr_heatmap.eps'), bbox_inches = 'tight', dpi= 200)

# %% ####### LEARNING #########

## Gridsearch for model selection using sklearn pipeline objects ##

### Scalers

scalers=[RobustScaler(), StandardScaler(), MinMaxScaler()]

cv=StratifiedKFold(5).split(X=feats, y=data['Cohort']) ## Crossvalidation stratified on country of origin


### Models to be learnt and evaluated

lasso=LassoCV(fit_intercept=True, cv = 5)
ridge=RidgeCV(fit_intercept=True, cv = 5)
pls=PLSRegression(n_components=10, scale=False)
gbr=GradientBoostingRegressor(loss='lad')

regressors=[lasso,ridge, pls, gbr]

# %% #### Pipeline ####

best_estimators = [] ## list of descriptions of best estimators for each target

for target in targets:

    pipe = Pipeline([('scale', StandardScaler()),
                    ('regress', lasso)])
    
    param_grid = [{ 'scale': scalers,
                    'regress': regressors}]
    
    grid = GridSearchCV(pipe, n_jobs=16, pre_dispatch=8,  param_grid=param_grid, cv=cv) ### parallelization for gridsearch speedup
    grid.fit(feats, data.loc[:,target].values)
    
    best=grid.cv_results_['params'][grid.best_index_] ## Return best estimator and its parameters
    best_estimators.append (best)

# %% #### Fine tuning best estimators (Ridge), train one model for each target ###

feats_s = StandardScaler().fit_transform(data.loc[:,feats].values)
alphas=np.linspace(0.01,30, 2000) ## regularization space
reg=RidgeCV(alphas=alphas, normalize=False, cv=5, fit_intercept=True)

scores = [] ## store model results (prediction R^2, coefficients and predictions\)
coeffs = []
preds = []

for target in targets:
    reg.fit(feats_s, data.loc[:,target].values)
    
    score=cross_val_score(reg,X=feats_s, y=data.loc[:,target].values,cv=cv)
    pred=cross_val_predict(reg, X=feats_s, y=data.loc[:,target].values,cv=cv)

    scores.append(score)
    coeffs.append(reg.coef_)
    preds.append(pred)

# %% ## Ridge Coefficients plot ##

for i in range (len (coeffs)):
    feat_rank_idx = np.argsort(coeffs[i])[::-1]

    f,ax=plt.subplots(figsize=(20,20))
    ax.scatter(np.arange(0,10),np.sort(np.abs(coeffs[i]))[::-1], marker='^', alpha=.6,
    c=np.sort(np.abs(coeffs[i]))[::-1], cmap='viridis', s=300)
    ax.plot(np.sort(np.abs(coeffs[i]))[::-1], lw=2, color='blue', alpha=.2)
    ax.set_xticks(np.arange(0,10))
    ax.tick_params(labelsize=18)
    ax.set_ylabel('Absolute Ridge Beta Magnitude', fontsize=30)
    ax.set_xlabel('Ranked Features', fontsize=30)
    for label, x, y in zip(np.asarray(feats)[feat_rank_idx],
                            np.arange(len(feats)),
                            np.sort(np.abs(coeffs[i]))[::-1] ):

                            plt.annotate(
                                label,
                                xy=(x, y), xytext=(-4, 4),
                                textcoords='offset points', ha='right', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                                fontsize=18)
    ax.set_title('Coeffs for %s'%targets[i], fontsize=30)
    f.savefig(pj(results_dir, 'abs_ridgecoeff%s.png'%targets[i]), bbox_inches='tight')

# %% Sample prediction plot (BMI) with gradient

# BMI #

for c, feat in enumerate(feats):
            
    t_data=np.zeros([2,len(data)])
    
    t_data[0,:] = data.loc[:,targets[0]].values
    t_data[1:,] = preds[0]

    score_df=pd.DataFrame(columns=['True Label', 'Predicted Label'], data=t_data.T)
    score_df[feat] = data[feat].values
    score_df['target_cat'] = data['BMI_T0'].values

    cmap = plt.get_cmap('Spectral_r') ## Create Cmap and Normalize for color extraction
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1) ## pass color as color=cmap(norm(a_certain_value))
    
    norm_feats = MinMaxScaler((-1,1)).fit_transform(data.loc[:,feat].values.reshape(-1,1))
    
    f,ax= plt.subplots(figsize=(10,10))
    sns.regplot(data=score_df,x='True Label',y='Predicted Label',
        truncate=True, scatter = False, ax = ax)
    
    ax.scatter(score_df['True Label'],
               score_df['Predicted Label'],
               c = cmap(norm(norm_feats[:,0])), marker='o', s=30)
    ax.set_title('BMI, Prediction R^2 = %.2f'%scores[0], fontsize = 10)
    ax1 = f.add_axes([0.35, 0.98, 0.3, 0.03])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    
    cb1.set_label(feat, fontsize=14)
    ax.vlines(25,28,36, color = 'black', alpha = .5, lw = .5)
    ax.vlines(30,28,36, color = 'black', alpha = .5, lw = .5)
    ax.vlines(40,28,36, color = 'black', alpha = .5, lw = .5)
    ax.set_xlim(22,44)
    ax.set_ylim(28,36)
    f.savefig(pj(results_dir, 'delta_grads_BMI_%s.png'%feat))
