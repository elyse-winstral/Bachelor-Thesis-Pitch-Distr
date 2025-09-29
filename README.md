# Bachelor-Thesis-Pitch-Distr.
## Predicting Pitch Distribution using Quantile Regression in Baseball

### Abstract: 
This thesis explores the application of quantile regression techniques to predict pitch locations in baseball. Using data from Statcast, a high accuracy tool for tracking player movement and ball information, we developed models to estimate the conditional distribution of pitch locations. The methodology incorporates training quantile regression models in Python on the dataset of features of every MLB game in the 2023 season. Using the estimated quantiles, the distribution for each pitch was approximated. The quantile regression model was found to have very high accuracy. When compared with a naive model our model was able to reflect the predictability of given situations by yielding more certain distributions. Comparison with a similar model is also discussed. Finally, an analysis of the features was conducted to understand how the quantile regression models processed inputs. This study contributes to the field of sports analytics by enhancing the understanding of predictive modeling in baseball and offering practical applications for teams and analysts.



- The data from the 2023 season was predominately collected via statcast in python. However, for the batting order wasn't available in the statcast library. Using the baseballr package available in R, the batting order could be saved as two csv files. 
- Then the data was pre-processed in the data_prep file. This consisted mostly of normalizing variables, converting variables to binary values. 
- In tuning_training the quantile regression models are tuned and trained on the data.
- In feature_analysis the model features are analyzed with the help of the shap library. This gives insight into how the model evaluates the different variables, and how the variables may interact with each other.
- Finally, in model_results, the model is compared with various other models. After comparison, some visualizations of the predictions are made, along with some analysis of the model predictions. 