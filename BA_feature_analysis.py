import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from BA_tuning_training import Xx_train, Xz_train



# Load models
with open('models_expanded_features.pkl', 'rb') as f:
    models = pickle.load(f)

alphas = np.round(np.linspace(0.1, 0.9, 9), 1)



x_explainer = shap.TreeExplainer(models[0.5]['x'])
x_shap_values = x_explainer(Xx_train)


z_explainer = shap.TreeExplainer(models[0.5]['z'])
z_shap_values = z_explainer(Xz_train)


# Plot main features of each axis
plt.title("X-coordinate feature influences")
shap.plots.beeswarm(x_shap_values)
plt.title("Z-coordinate feature influences")
shap.plots.beeswarm(z_shap_values)

# Plot X-axis features against each other (correlation overview)
interesting = Xx_train.loc[:, ['prev_x', '2_prev_x', '3_prev_x', 'prev_z', '2_prev_z', '3_prev_z']]
batting_order = Xx_train.loc[:, ['batting_order']]
# shap.dependence_plot('batting_order', x_shap_values.values, Xx_train)

for name in Xx_train.columns:
    shap.dependence_plot(name, x_shap_values.values, Xx_train)