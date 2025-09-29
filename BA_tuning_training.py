import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
import pickle


# Read in and split data
data = pd.read_csv("Expanded_Features.csv", index_col=0)
data = data[data['balls'] < 4]

X = data
X = X.drop(columns= ['normed_x', 'normed_z'])

Y_x = np.round( data.loc[:, 'normed_x'], 2)
Y_z = np.round( data.loc[:, 'normed_z'], 2)

Xx_train, Xx_test, Yx_train, Yx_test = train_test_split(X, Y_x, test_size=0.2, random_state=1)
Xz_train, Xz_test, Yz_train, Yz_test = train_test_split(X, Y_z, test_size=0.2, random_state=1)


# Quantile loss function

def quantile_loss(y_true, y_pred, quantile):
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual)


# Create training and testing subset to tune parameters

optimization_train_Xx, optimization_test_Xx, optimization_train_Yx, optimization_test_Yx = train_test_split(Xx_train, Yx_train, test_size = 0.25, random_state=1)
optimization_train_Xz, optimization_test_Xz, optimization_train_Yz, optimization_test_Yz = train_test_split(Xz_train, Yz_train, test_size = 0.25, random_state=1)

def objective (trial, dimension_x = True):

    quantile = 0.5
    
    params = {
        'n_jobs': 1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'alpha': quantile,
        'objective': 'quantile',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True), 
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.4, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 5, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100)
    }

    if dimension_x:
        training_data_x = optimization_train_Xx
        training_data_y = optimization_train_Yx
        testing_data_x = optimization_test_Xx
        testing_data_y = optimization_test_Yx
        
    else:
        training_data_x = optimization_train_Xz
        training_data_y = optimization_train_Yz
        testing_data_x = optimization_test_Xz
        testing_data_y = optimization_test_Yz
    
    model = LGBMRegressor(**params)
    model.fit(training_data_x, training_data_y)
    pred = model.predict(testing_data_x)

    loss = quantile_loss(testing_data_y, pred, quantile).mean()

    return loss


# Tune parameters for x- and z- axis

study_x = optuna.create_study(direction='minimize')
study_x.optimize(objective, n_trials= 100)

z_objective = lambda trial: objective(trial, dimension_x=False)
study_z = optuna.create_study(direction='minimize')
study_z.optimize(z_objective, n_trials=100)

# overview of fitting results
# optuna.visualization.plot_optimization_history(study_z)
# optuna.visualization.plot_optimization_history(study_x)


# Get tuned parameters for x and z (use the tuned parameters for 0.5 quantile everywhere- assume there isn't much difference within quantile)
quantile = 0.5    
params = {
        'n_jobs': 1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'alpha': quantile,
        'objective': 'quantile',
}

params_x, params_z = params.copy(), params.copy()
params_x.update(study_x.best_params)
params_z.update(study_z.best_params)


# Save/Load parameters
with open('params_z_expanded_features.pkl', 'wb') as fp:
    pickle.dump(params_z, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('params_x_expanded_features.pkl', 'wb') as fp:
    pickle.dump(params_x, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('params_x_expanded_features.pkl', 'rb') as f:
    params_x = pickle.load(f)
with open('params_z_expanded_features.pkl', 'rb') as g:
    params_z = pickle.load(g)


# Train models for every quantile of interest

# alphas = np.round(np.linspace(0.05, 0.95, 19), 2)
alphas = np.round(np.linspace(0.1, 0.9, 9), 1) #use this for expanded features!!

models= {}




for quantile in tqdm(alphas): #alphas[::n] only takes every n-th alpha entry
    quantiles = {}

    params_x['alpha'] = quantile

    lgbx = LGBMRegressor(**params_x)
    lgbx.fit(Xx_train, Yx_train)
    quantiles['x'] = lgbx
    # lgbx.save_model('lgbx_'+str(quantile), num_iteration = lgbx.best_iteration)

    params_z['alpha'] = quantile
    
    lgbz = LGBMRegressor(**params_z)
    lgbz.fit(Xz_train, Yz_train)
    quantiles['z'] = lgbz
    # lgbz.save_model('lgbz_'+str(quantile), num_iteration = lgbz.best_iteration)

    
    models[quantile] = quantiles

# Save/Load models
with open('models_expanded_features.pkl', 'wb') as f:
    pickle.dump(models, f)
    
with open('models_expanded_features.pkl', 'rb') as f:
    models = pickle.load(f)

alphas = np.round(np.linspace(0.1, 0.9, 9), 1)