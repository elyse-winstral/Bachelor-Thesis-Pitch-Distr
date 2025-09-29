import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import optimize
import scipy.stats as stats
from tqdm import tqdm
import pickle
from BA_tuning_training import Xx_test, Xz_test, Yx_train, Yz_train, Yx_test, Yz_test, quantile_loss


# Load models
with open('models_expanded_features.pkl', 'rb') as f:
    models = pickle.load(f)

alphas = np.round(np.linspace(0.1, 0.9, 9), 1)

# Get x and z coordinate predictions
preds_x = pd.DataFrame(index = Xx_test.index)
preds_z = pd.DataFrame(index = Xz_test.index)

for alpha in alphas:
    preds_x[str(alpha)] = models[alpha]['x'].predict(Xx_test)
    preds_z[str(alpha)] = models[alpha]['z'].predict(Xz_test)

 
# Frequency of quantile crossing (error) 
def quantile_crossing (preds, alphas):
    diff_x = preds.diff(axis = 1)
    diff_x = diff_x.drop(columns = ['0.1'])


    tot = len(diff_x)
    amount =[]
    rate = []
    ind = []
    for col in range(len(alphas)-1):
        amnt = (diff_x.loc[:, str(alphas[col+1])] < 0).sum()
        ind.append(str(alphas[col+1])+ '-' + str(alphas[col]))
        amount.append(int(amnt))
        rate.append(amnt/tot)

        # plt.hist(diff_x[str(alphas[col+1])], bins = 100, density = True)
        # plt.vlines(0, ymin = 0, ymax = 10, color = 'red')
        # plt.title(str(alphas[col+1]))
        # plt.show()

    amount = pd.Series(amount)
    rate = np.round(pd.Series(rate), 4)
    ind = pd.Series(ind)
    table = pd.concat([amount, rate], axis = 1)
    table = table.rename(columns = {0: 'amount', 1 : 'rate'})
    table = table.set_index(ind)
    table = table.T
    return table

#quantile_crossing(preds_x, alphas)

# Joint accuracy of a given model - frequency of a pitch being to the left of the x-quantile value and below the z-quantile value
def joint_accuracy(Yx, Yz, x_quantiles, z_quantiles, display_plot = False):

    if isinstance(x_quantiles, np.ndarray):
        return joint_accuracy_array(Yx, Yz, x_quantiles, z_quantiles, display_plot)

    accuracy = {}
    areas = {}
    for alpha in alphas: 
        alpha_acc = pd.Series()
        alpha_acc = ((Yx - x_quantiles[str(alpha)])<0).astype(int) * ((Yz - z_quantiles[str(alpha)])<0).astype(int)
        areas[str(alpha)] = str(np.round(alpha**2, 2))


        accuracy[str(alpha)] = alpha_acc.mean() 

    accuracy = pd.Series(accuracy)
    accuracy = accuracy.rename(index = areas)



    if display_plot:
        plt.plot(squares, squares, color = 'gray', alpha = 0.3)
        plt.scatter(squares, accuracy, alpha = 0.9, marker='x', color = 'red', label = "Joint Accuracy")
        plt.title("Joint Accuracy in Regression Model")
        plt.legend()
        plt.show()

    return accuracy

def joint_accuracy_array(Yx, Yz, x_quantiles, z_quantiles, display_plot = False):
    true_pitch_x = Yx
    true_pitch_x = true_pitch_x.reset_index(drop = True)
    true_pitch_z = Yz
    true_pitch_z = true_pitch_z.reset_index(drop = True)


    accuracy = {}
    areas = {}
    squares = np.round(alphas**2, 2)

    for ind in range(len(alphas)):
        alpha_acc = pd.Series()
        alpha_acc = ((true_pitch_x - x_quantiles[ind])<0).astype(int) * ((true_pitch_z - z_quantiles[ind])<0).astype(int)
        areas[str(alphas[ind])] = str(squares[ind])

        accuracy[str(alphas[ind])] = alpha_acc.mean() 

    accuracy = pd.Series(accuracy)
    accuracy = accuracy.rename(index = areas)


    if display_plot:
        plt.plot(squares, squares, color = 'gray', alpha = 0.3)
        plt.scatter(squares, accuracy, alpha = 0.9, marker='x', color = 'blue', label = "Joint Accuracy")
        plt.title("Joint Accuracy in Model")
        plt.legend()
        plt.show()
    
    return accuracy


# Accuracy of a given model's quantiles - how often a pitch is to the left of each quantile
def accuracy (Yx, Yz, x_quantiles, z_quantiles, display_plot=False):

    if isinstance(x_quantiles, np.ndarray):
        return accuracy_array(Yx, Yz, x_quantiles, z_quantiles, display_plot)
  

    dist_x = pd.DataFrame()
    dist_z = pd.DataFrame()


    accuracy = {}
    # print(x_quantiles['0.05'])
    # print(true_pitch_x)
    for alpha in alphas:
        alpha_acc = {}
        dist_x[str(alpha)] = ((Yx - x_quantiles[str(alpha)])<0).astype(int)
        dist_z[str(alpha)] = ((Yz - z_quantiles[str(alpha)])<0).astype(int)

        alpha_acc['x'] = dist_x[str(alpha)].mean()
        alpha_acc['z'] = dist_z[str(alpha)].mean()
        accuracy[str(alpha)] = alpha_acc

    accuracy = pd.DataFrame(accuracy).T

    if display_plot:
        plt.plot(alphas, alphas, alpha = 0.3, color = 'gray')
        plt.scatter(alphas, accuracy['z'], alpha = 0.9, marker='+', color = 'blue', label = "Z- coord. accuracy")
        plt.scatter(alphas, accuracy["x"], alpha = 0.9, marker='x', color = 'red', label = "X- coord. accuracy")
        plt.title("Accuracy of quantiles in Regression Model")
        plt.legend()
        plt.show()

    return(np.round(accuracy, 5))

def accuracy_array (Yx, Yz, x_quantiles, z_quantiles, display_plot):

    true_pitch_x = Yx
    true_pitch_x = true_pitch_x.reset_index(drop = True)
    true_pitch_z = Yz
    true_pitch_z = true_pitch_z.reset_index(drop = True)


    dist_x = pd.DataFrame()
    dist_z = pd.DataFrame()


    accuracy = {}

    for ind in range(len(alphas)):
        alpha_acc = {}
        dist_x[str(alphas[ind])] = ((true_pitch_x - x_quantiles[ind])<0).astype(int)
        dist_z[str(alphas[ind])] = ((true_pitch_z - z_quantiles[ind])<0).astype(int)

        alpha_acc['x'] = dist_x[str(alphas[ind])].mean()
        alpha_acc['z'] = dist_z[str(alphas[ind])].mean()
        accuracy[str(alphas[ind])] = alpha_acc

    accuracy = pd.DataFrame(accuracy).T

    if display_plot:
        plt.plot(alphas, alphas, alpha = 0.3, color = 'gray')
        plt.scatter(alphas, accuracy['z'], alpha = 0.9, marker='+', color = 'blue', label = "Z- coord. accuracy - comparison")
        plt.scatter(alphas, accuracy["x"], alpha = 0.9, marker='x', color = 'lightblue', label = "X- coord. accuracy - comparison")
        plt.legend()
        plt.show()

    return(np.round(accuracy, 5))


# Quantile Loss of a given model
def get_quantile_loss(Yx, Yz, x_quantiles, z_quantiles, table = True):

    if isinstance(x_quantiles, np.ndarray):
        return quantile_loss_array(Yx, Yz, x_quantiles, z_quantiles, table)
    
    index = Yx.index
    x_quantiles = x_quantiles.loc[index]
    z_quantiles = z_quantiles.loc[index]
    
    x_quant_loss = pd.DataFrame()
    z_quant_loss = pd.DataFrame()
    tot_avg_loss = pd.Series()


    for alpha in alphas:
        x_quant_loss[str(alpha)] = quantile_loss(Yx, x_quantiles[str(alpha)], alpha)
        z_quant_loss[str(alpha)] = quantile_loss(Yz, z_quantiles[str(alpha)], alpha)

        tot_avg_loss[str(alpha)] = np.sqrt(x_quant_loss[str(alpha)] ** 2 + z_quant_loss[str(alpha)] ** 2).mean()
    
    
    if table: return tot_avg_loss
    return tot_avg_loss.mean()

def quantile_loss_array(Yx, Yz, x_quantiles, z_quantiles, table):

    x_quant_loss = pd.DataFrame()
    z_quant_loss = pd.DataFrame()
    tot_avg_loss = pd.Series()

    for ind in range(len(alphas)):
        x_quant_loss[str(alphas[ind])] = quantile_loss(Yx, x_quantiles[ind], alphas[ind])
        z_quant_loss[str(alphas[ind])] = quantile_loss(Yz, z_quantiles[ind], alphas[ind])

        tot_avg_loss[str(alphas[ind])] = np.sqrt(x_quant_loss[str(alphas[ind])] ** 2 + z_quant_loss[str(alphas[ind])] ** 2).mean()
        

    if table: return tot_avg_loss
    return tot_avg_loss.mean()


# Quantile distance of a given model
def quantile_distance(x_quantiles, z_quantiles, absolute = True, table = False):

    if isinstance(x_quantiles, np.ndarray):
        return quantile_distance_array(x_quantiles, z_quantiles, absolute, table)
    
    mid = int((len(alphas)-1)/2)

    if absolute: 
        x_quantiles_difference = x_quantiles.subtract(x_quantiles['0.5'], axis = 0)
        z_quantiles_difference = z_quantiles.subtract(z_quantiles['0.5'], axis = 0)

    else:
        x_quant_diff_low = x_quantiles.iloc[:, :mid+1]
        x_quant_diff_low = x_quant_diff_low.diff(axis = 1, periods = -1)
        x_quant_diff_high = x_quantiles.iloc[:, mid:]
        x_quant_diff_high = x_quant_diff_high.diff(axis = 1, periods = 1)
        x_quant_diff_high = x_quant_diff_high.drop(columns = ['0.5'])
        x_quantiles_difference = pd.concat([x_quant_diff_low, x_quant_diff_high], axis = 1)

        z_quant_diff_low = z_quantiles.iloc[:, :mid+1]
        z_quant_diff_low = z_quant_diff_low.diff(axis = 1, periods = -1)
        z_quant_diff_high = z_quantiles.iloc[:, mid:]
        z_quant_diff_high = z_quant_diff_high.diff(axis = 1, periods = 1)
        z_quant_diff_high = z_quant_diff_high.drop(columns = ['0.5'])
        z_quantiles_difference = pd.concat([z_quant_diff_low, z_quant_diff_high], axis = 1)


    quantile_distances = pd.DataFrame()

    for ind in alphas[:mid]:
        # Retrieve the specific columns and verify their contents
        alpha_mid_minus_ind = str(np.round(alphas[mid] - ind, 1))
        alpha_mid_plus_ind = str(np.round(alphas[mid] + ind, 1))
        # print(alpha_mid_minus_ind, alpha_mid_plus_ind)
        quantile_distances['C' + str(int(ind*200))] = pd.Series([np.sqrt(x_quantiles_difference[alpha_mid_minus_ind]**2 + z_quantiles_difference [alpha_mid_minus_ind]**2).mean(), 
                                                  np.sqrt(x_quantiles_difference[alpha_mid_minus_ind]**2 + z_quantiles_difference[alpha_mid_plus_ind]**2).mean(), 
                                                  np.sqrt(x_quantiles_difference[alpha_mid_plus_ind]**2 + z_quantiles_difference[alpha_mid_minus_ind]**2).mean(), 
                                                  np.sqrt(x_quantiles_difference[alpha_mid_plus_ind]**2 + z_quantiles_difference[alpha_mid_plus_ind]**2).mean()])
    
    indices = {0:"x low, z low", 1:'x low, z high', 2:'x high, z low', 3:'x high, z high'}
    quantile_distances = quantile_distances.rename(index = indices)
    quantile_distances = quantile_distances.T
    if table: return quantile_distances
    else: return quantile_distances.mean(axis=1)

def quantile_distance_array (x_quantiles, z_quantiles, absolute, table):

    mid = int(len(x_quantiles)/2)
    dist_table = pd.DataFrame()
    if absolute:
        dist_table['x low, z low'] = np.sqrt((x_quantiles[mid] - x_quantiles[mid-1::-1])**2 + (z_quantiles[mid] - z_quantiles[mid-1::-1])**2)
        dist_table['x low, z high'] = np.sqrt((x_quantiles[mid] - x_quantiles[mid-1::-1])**2 + (z_quantiles[mid] - z_quantiles[mid+1:])**2)
        dist_table['x high, z low'] = np.sqrt((x_quantiles[mid] - x_quantiles[mid+1:])**2 + (z_quantiles[mid] - z_quantiles[mid-1::-1])**2)
        dist_table['x high, z high'] = np.sqrt((x_quantiles[mid] - x_quantiles[mid+1:])**2 + (z_quantiles[mid] - z_quantiles[mid+1:])**2)

    else:
        dist_table['x low, z low'] = np.sqrt((x_quantiles[mid:0:-1] - x_quantiles[mid-1::-1])**2 + (z_quantiles[mid:0:-1] - z_quantiles[mid-1::-1])**2)
        dist_table['x low, z high'] = np.sqrt((x_quantiles[mid:0:-1] - x_quantiles[mid-1::-1])**2 + (z_quantiles[mid:-1] - z_quantiles[mid+1:])**2)
        dist_table['x high, z low'] = np.sqrt((x_quantiles[mid:-1] - x_quantiles[mid+1:])**2 + (z_quantiles[mid:0:-1] - z_quantiles[mid-1::-1])**2)
        dist_table['x high, z high'] = np.sqrt((x_quantiles[mid:-1] - x_quantiles[mid+1:])**2 + (z_quantiles[mid:-1] - z_quantiles[mid+1:])**2)

    indices = {0:'C20', 1:'C40', 2:'C60', 3:'C80'}
    dist_table = dist_table.rename(index=indices)
    if table: return dist_table
    else: return dist_table.mean(axis=1)


#Quantile area of a given model 
def quantile_area(x_quantiles, z_quantiles):

    if isinstance(x_quantiles, np.ndarray):
        return quantile_area_array(x_quantiles, z_quantiles)
    
    x_quantiles_dist = np.round(np.abs(x_quantiles.subtract(x_quantiles['0.5'], axis = 0)), 2)
    z_quantiles_dist = np.round(np.abs(z_quantiles.subtract(z_quantiles['0.5'], axis = 0)), 2)
    
    mid = int((len(alphas)-1)/2)
    
    area = pd.DataFrame()
    for ind in alphas[:mid]:
        right_index, left_index = str(np.round(alphas[mid] - ind, 1)), str(np.round(alphas[mid] + ind, 1))
        area['C'+str(int(ind*200))] = (x_quantiles_dist[left_index] + x_quantiles_dist[right_index]) * (z_quantiles_dist[left_index] + z_quantiles_dist[right_index])
    return area

def quantile_area_array (x_quantiles, z_quantiles):
    
    mid = int((len(alphas)-1)/2)
    x_quantiles_dist = np.round(np.abs(x_quantiles - x_quantiles[mid]), 3)
    z_quantiles_dist = np.round(np.abs(z_quantiles - z_quantiles[mid]), 3)

    area = pd.Series()

    for ind in range(1,len(alphas[:mid+1])):
        area['C'+str(int((ind)*20))] = (x_quantiles_dist[mid+ind] + x_quantiles_dist[mid - ind]) * (z_quantiles_dist[mid+ind] + z_quantiles_dist[mid - ind])
    return area



# Create comparison models and compare quantile loss, accuracy and distance

# Our model- quantile regression:
regression_accuracy = accuracy(Yx_test, Yz_test, preds_x, preds_z,)
regression_loss = get_quantile_loss(Yx_test, Yz_test, preds_x, preds_z)
regression_q_dist = quantile_distance(preds_x, preds_z)
regression_q_area = quantile_area(preds_x, preds_z).mean()


x_training = Yx_train.reset_index(drop=True)
z_training = Yz_train.reset_index(drop=True)

x_training = x_training.fillna(0)
z_training = z_training.fillna(0)

x_training = np.array(x_training)
z_training = np.array(z_training)

# naive model
naive_x_quantiles = np.quantile(x_training, alphas)
naive_z_quantiles = np.quantile(z_training, alphas)

    # using gaussian kdes as oppposed to univariate splines because the splines smooth too much
naive_x_pdf = stats.gaussian_kde(x_training)
naive_z_pdf = stats.gaussian_kde(z_training)

naive_accuracy = accuracy(Yx_test, Yz_test, naive_x_quantiles, naive_z_quantiles)
naive_loss = get_quantile_loss(Yx_test, Yz_test, naive_x_quantiles, naive_z_quantiles)
naive_q_dist = quantile_distance(naive_x_quantiles, naive_z_quantiles)
naive_q_area = quantile_area(naive_x_quantiles, naive_z_quantiles)


# normal distribution
mu_x = x_training.mean()
mu_z = z_training.mean() #averaged location
sigma = 1
norm_x_quantiles = stats.norm.ppf(alphas, mu_x, sigma)
norm_z_quantiles = stats.norm.ppf(alphas, mu_z, sigma)

normal_accuracy = accuracy(Yx_test, Yz_test, norm_x_quantiles, norm_z_quantiles)
normal_loss = get_quantile_loss(Yx_test, Yz_test, norm_x_quantiles, norm_z_quantiles)
normal_q_dist = quantile_distance(norm_x_quantiles, norm_z_quantiles)
normal_q_area = quantile_area(norm_x_quantiles, norm_z_quantiles)


# uniform distribution - take 10th and 90th quantile of pitches
lower_x, lower_z = naive_x_quantiles[1], naive_z_quantiles[1]
upper_x, upper_z = naive_x_quantiles[-2], naive_z_quantiles[-2]

uniform_x_quantiles = stats.uniform.ppf(alphas, lower_x, upper_x)
uniform_z_quantiles = stats.uniform.ppf(alphas, lower_z, upper_z)

uniform_accuracy = accuracy(Yx_test, Yz_test, uniform_x_quantiles, uniform_z_quantiles)
uniform_loss = get_quantile_loss(Yx_test, Yz_test, uniform_x_quantiles, uniform_z_quantiles)
uniform_q_dist = quantile_distance(uniform_x_quantiles, uniform_z_quantiles)
uniform_q_area = quantile_area(uniform_x_quantiles, uniform_z_quantiles)


# Summarize results:
Accuracy = pd.DataFrame({'Quantile Reg. x' : regression_accuracy.iloc[:,0], 'Quantile Reg. z' : regression_accuracy.iloc[:,1], 'Naive x' : naive_accuracy.iloc[:,0], 'Naive z' : naive_accuracy.iloc[:,1], 'Normal x' : normal_accuracy.iloc[:,0], 'Normal z' : normal_accuracy.iloc[:,1], 'Uniform x' : uniform_accuracy.iloc[:,0], 'Uniform z' : uniform_accuracy.iloc[:,1]})

Average_q_loss = pd.DataFrame({'Quantile Reg.' : regression_loss, 'Naive' : naive_loss, 'Normal' : normal_loss, 'Uniform' : uniform_loss})

Average_q_dist = pd.DataFrame({'Quantile Reg.' : regression_q_dist, 'Naive' : naive_q_dist, 'Normal' : normal_q_dist, 'Uniform' : uniform_q_dist})

Average_q_area = pd.DataFrame({'Quantile Reg.' : regression_q_area, 'Naive' : naive_q_area, 'Normal' : normal_q_area, 'Uniform' : uniform_q_area})


# Joint accuracy comparison between naive and regression
naive_joint_accuracy = joint_accuracy(Yx_test, Yz_test, naive_x_quantiles, naive_z_quantiles)
reg_joint_accuracy = joint_accuracy(Yx_test, Yz_test, preds_x, preds_z)
#n_vals = naive_joint_accuracy.values
#r_vals = reg_joint_accuracy.values


# Compare loss in more predictable scenario (pitcher's disadvantage)
count_Xx = Xx_test[(Xx_test['balls'] == 3) & (Xx_test['strikes'] == 0)]
count_Xx = Xx_test[(Xx_test['wld'] == 3)]
count_x = Yx_test.loc[count_Xx.index]
count_z = Yz_test.loc[count_Xx.index] 

reg_acc_easy = get_quantile_loss(count_x, count_z, preds_x, preds_z,False)
naive_acc_easy = get_quantile_loss(count_x, count_z, naive_x_quantiles, naive_z_quantiles, False)
norm_acc_easy = get_quantile_loss(count_x, count_z, norm_x_quantiles, norm_z_quantiles, False)
unif_acc_easy = get_quantile_loss(count_x, count_z, uniform_x_quantiles, uniform_z_quantiles, False)
print(reg_acc_easy, naive_acc_easy, norm_acc_easy, unif_acc_easy)


##################### -------------- Visualization and Plots -------------- #####################


# Plot some predictions against results in test set
def visualize_pred (preds_x, preds_z):
    for ind in range(3,15):
        
        x_quantiles = preds_x.iloc[ind, :]
        z_quantiles = preds_z.iloc[ind, :]


        cdf_x = UnivariateSpline(x_quantiles, alphas, k = 5)
        pdf_x = cdf_x.derivative()

        cdf_z = UnivariateSpline(z_quantiles, alphas, k = 5)
        pdf_z = cdf_z.derivative()
        

        finer_x_quantiles = np.linspace(x_quantiles[0], x_quantiles[-1], 100)
        finer_z_quantiles = np.linspace(z_quantiles[0], z_quantiles[-1], 100)

        X, Z = np.meshgrid(finer_x_quantiles, finer_z_quantiles)
        pdf = pdf_x(X) * pdf_z(Z)


        img = plt.imshow(pdf, cmap='viridis', extent = [x_quantiles[0], x_quantiles[-1], z_quantiles[0], z_quantiles[-1]], alpha = 0.9)

        mid = int(len(alphas)/2)
        plt.plot([-0.5, -0.5, 0.5, 0.5, -0.5],[0, 1, 1, 0, 0], 'k', label = 'normalized strike zone')
        plt.plot(Yx_test.iloc[ind], Yz_test.iloc[ind], 'r', marker = "x", label = 'actual pitch')
        for ind in range(mid+1):
            plt.plot([x_quantiles[mid+ind],x_quantiles[mid+ind], x_quantiles[mid-ind], x_quantiles[mid-ind], x_quantiles[mid+ind]], [z_quantiles[mid+ind], z_quantiles[mid-ind], z_quantiles[mid-ind], z_quantiles[mid+ind], z_quantiles[mid+ind]], color = 'black', alpha = 0.3, label = str(ind))
            if ind > 0:
                x_middle = (x_quantiles[mid + ind] - x_quantiles[mid - ind]) / 2 + x_quantiles[mid - ind]
                plt.text(x_quantiles[mid], z_quantiles[mid - ind] + 0.04, 'C' + str((ind)*20), ha='center', va='top', color='black', alpha = 0.8, fontsize=8)


        plt.show()
    return

#visualize_pred(preds_x, preds_z)

# Plot Centralized quantile area based on count
regs = {}
for i in range (4):
    for j in range (3):
        index = Xx_test[(Xx_test['balls'] == i) & (Xx_test['strikes'] == j)].index
        pitch_x = Yx_test.loc[index]
        pitch_z = Yz_test.loc[index]
        reg_x = preds_x.loc[index]
        reg_z = preds_z.loc[index]
        regs[str(i) + '-' + str(j)] = quantile_area(reg_x, reg_z).mean()

regs_bases = {}
for i in range (2):
    for j in range (2):
        for k in range (2):
            index = Xx_test[(Xx_test['on_3b'] == i) & (Xx_test['on_2b'] == j) & (Xx_test['on_1b'] == k)].index
            pitch_x = Yx_test.loc[index]
            pitch_z = Yz_test.loc[index]
            reg_x = preds_x.loc[index]
            reg_z = preds_z.loc[index]
            regs_bases[str(i) + str(j) + str(k)] = quantile_area(reg_x, reg_z).mean()


CCA_count = pd.DataFrame(regs)
CCA_count['naive'] = naive_q_area

# Extract relevant data for plotting
ahead = CCA_count[['0-1', '0-2', '1-2', '2-2']]
behind = CCA_count[['2-0', '3-0', '3-1']]
neutral = CCA_count[['1-0', '0-0', '2-1', '1-1', '3-2']]

# Function to plot a given DataFrame on a specified axis
def plot_data(ax, data, title):
    for column in data.columns:
        ax.plot(data.index, data[column], label=column)
        ax.scatter(data.index, data[column])
    ax.plot(data.index, CCA_count['naive'], color='black', ls='--', label='naive')
    ax.scatter(data.index, CCA_count['naive'], color='black')
    ax.set_ylim(0, 3)
    ax.set_title(title)
    ax.legend(loc='upper left')

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

plot_data(axs[0], ahead, "Pitcher favored")
plot_data(axs[1], neutral, "Neutral")
plot_data(axs[2], behind, "Batter favored")

fig.suptitle("Area of Centralized Credible Areas Based on Count")
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust top to make space for the main title
plt.show()



# Determine largest probability over a square
def get_prob_rect_naive(x_data, z_data):

    x_sorted_data = np.sort(x_data)
    x_cdf_values = np.arange(1, len(x_sorted_data) + 1) / len(x_sorted_data)
    cdf_x = interp1d(x_sorted_data, x_cdf_values, bounds_error=False, fill_value=(0, 1))
    x_quantiles = naive_x_quantiles

    z_sorted_data = np.sort(z_data)
    z_cdf_values = np.arange(1, len(z_sorted_data) + 1)/ len(x_sorted_data)
    cdf_z = interp1d(z_sorted_data, z_cdf_values, bounds_error=False, fill_value=(0, 1))
    z_quantiles = naive_z_quantiles

    max_x_length = x_quantiles[-1] - x_quantiles[0]
    max_z_length = z_quantiles[-1] - z_quantiles[0]

    x_lengths = np.linspace(0.1, 1.7, 100)
    z_lengths = np.linspace(0.1, 1.7, 100)
    areas = {}
    first_guess = [0, 0.5]
    
    for lx, lz in zip(x_lengths, z_lengths):
        applied = lambda c, lx=lx, lz=lz, cdf_x=cdf_x, cdf_z=cdf_z: -rect_area_prob(c, lx, lz, cdf_x, cdf_z)
    
        if lx > max_x_length or lz > max_z_length:
            continue

        x_lower_bound = naive_x_quantiles[0] + (lx / 2)
        x_upper_bound = naive_x_quantiles[-1] - (lx / 2)
        z_lower_bound = naive_z_quantiles[0] + (lz / 2)
        z_upper_bound = naive_z_quantiles[-1] - (lz / 2)

        if x_lower_bound > x_upper_bound or z_lower_bound > z_upper_bound:

            continue  # Skip if bounds are invalid

        bnds = (
            (x_lower_bound, x_upper_bound),
            (z_lower_bound, z_upper_bound)
        )

        best_center = optimize.minimize(applied, first_guess, bounds=bnds)
        first_guess = best_center.x
        areas[np.round(lx * lz, 5)] = applied(best_center.x) * (-1)
    areas = pd.Series(areas)
    return areas

def process_index(ind, preds_x, preds_z):
    ind_preds_x = preds_x.iloc[ind, :]
    ind_preds_z = preds_z.iloc[ind, :]

    # skip instances of quantile crossing
    if not (ind_preds_x.is_monotonic_increasing and ind_preds_z.is_monotonic_increasing):
        return None 

    # cdf_x = cdf(ind_preds_x)
    # cdf_z = cdf(ind_preds_z)
    return get_prob_rect(ind_preds_x, ind_preds_z)

def get_areas (preds_x, preds_z):
    results = []
    n = np.minimum(len(preds_x), 10)
    
    indices = pd.Series()
    for ind in tqdm(range(n)):
        result = process_index(ind, preds_x, preds_z)
        
        if result is not None:
            results.append(result)
            

    # Convert results to DataFrame
    regression_areas = pd.DataFrame(results).T
    regression_areas.sort_index(inplace=True)
    for col in regression_areas.columns:
        exceeded_threshold = False
        for i in range(len(regression_areas)):
            if (regression_areas.iloc[i, col] == regression_areas.iloc[:, col].max()):
                regression_areas.iloc[i:, col] = 0.64
                pass
            if (regression_areas.iloc[i, col] > 0.64):
                exceeded_threshold = True
            if exceeded_threshold and pd.isna(regression_areas.iloc[i, col]):
                regression_areas.iloc[i, col] = 0.64
    regression_areas = regression_areas.apply(lambda x: x.fillna((x.fillna(method='ffill') + x.fillna(method='bfill')) / 2))

    # Calculate the mean regression square
    avg_regression_rectangle = regression_areas.mean(axis=1)
    return avg_regression_rectangle

    x1, x2 = center[0] - (1/2)*length, center[0] + (1/2)*length
    z1, z2 = center[1] - (1/2)*length, center[1] + (1/2)*length
    return (cdf_x(x2) - cdf_x(x1)) * (cdf_z(z2) - cdf_z(z1))

reg_rect = get_areas(preds_x, preds_z)
naive_rect = get_prob_rect_naive(x_training, z_training)

n = 50
areas = np.linspace(0.1, 1.7, 100)
plt.plot(reg_rect.index, reg_rect, color = 'blue', label = f'Regression Model Average over {n} Cases')
# plt.scatter(regression_q_area,[0.04, 0.16, 0.36, 0.64], color = 'blue', label = "Area of CCA")
# plt.scatter(lengths, avg_regression_square, color = 'blue')
plt.plot(naive_rect.index, naive_rect, color = 'black', label = 'Naive Model')
# plt.scatter(naive_q_area, [0.04, 0.16, 0.36, 0.64], color = 'black')
# plt.scatter(lengths, naive_area, color = 'black')


plt.legend()
plt.xlabel("Area A of a Sqaure")
plt.ylabel("Largest Probability over Area A")
plt.title("Largest Probability over a Square")
plt.legend()

