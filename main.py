# A Comparison of Various Forecasting Models
# in Predicting Indian Rainfall from Spatio-temporal Data
# Author: Sanchari Biswas


# Import Libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score
import webbrowser
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import pickle

import warnings
warnings.filterwarnings('ignore')


# Code Block

weather_data = pd.read_csv('data/forecast_data.csv')
location_data = pd.read_csv('data/location_data.csv')
print(location_data.head())

# Plotting the cities for which we have the data for
INDIA_COORDINATES = [20.5937, 78.9629]

my_map = folium.Map(INDIA_COORDINATES) # Creates a map
def plot_map(df):
  # The function will add markers at the coordinates present in our dataframe
  folium.Marker(location=[df.lat, df.lon]).add_to(my_map)

# Apply the function to the dataframe
location_data.apply(plot_map, axis=1)
my_map.fit_bounds([[30.3753, 69.3451], [7.8731, 80.7718]])
filename1 = "map1.html"
my_map.save(filename1)
webbrowser.open('file://' + os.path.realpath(filename1))

print(weather_data.head())

# Extract the condition text from the json string
weather_data['condition'] = weather_data['condition'].apply(lambda x : eval(x)['text'])

print(weather_data.head())

# Many columns are same just with differnt units, for ex wind_mph and wind_kph. So, it's better to remove the redundancy
redundant_columns = ['temp_f', 'wind_kph', 'pressure_in', 'precip_in', 'feelslike_f', 'windchill_f', 'heatindex_f', 'dewpoint_f', 'vis_miles', 'chance_of_rain', 'chance_of_snow','gust_kph']

# Drop the redundant information
weather_data.drop(redundant_columns, axis=1, inplace=True)

print(weather_data.head())

# Visualise various continuous distributions
continuous_distributions = ['temp_c', 'wind_mph', 
                            'wind_degree', 'pressure_mb', 
                            'precip_mm', 'humidity', 
                            'feelslike_c', 'windchill_c', 
                            'heatindex_c', 'dewpoint_c', 
                            'vis_km', 'gust_mph']

# plt.figure(figsize=(10, 12))
# for i, dist_col in enumerate(continuous_distributions):

#   # Create subplots
#   plt.subplot(2, 6, i+1)
#   sns.histplot(weather_data[dist_col])
#   plt.title(dist_col + " Distribution")

# plt.show()

plt.style.use('seaborn')
weather_data['precip_mm'].hist(bins=25)
plt.xlabel("Precipitation (in mm)")
plt.xticks(np.arange(0,21,1))
plt.show()

# Plotting categorical discrete variables

# 1. Plotting days and nights count
plt.figure(figsize=(6,4))
sns.countplot(data=weather_data, x='is_day')
plt.show()

weather_data['condition'].value_counts().sort_values().plot(kind='barh', figsize=(6,4))
plt.xlabel("Weather Conditions")
plt.ylabel("Frequency")
plt.grid()
plt.show()

weather_data['wind_dir'].value_counts().sort_values().plot(kind='bar', figsize=(6,4))
plt.xlabel("Wind Direction")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Plot state distribution
weather_data['state'].value_counts().sort_values().plot(kind='barh', figsize=(6,4))
plt.xlabel("Frequency")
plt.ylabel("State")
plt.show()

# Plot the count of rainy forecasts
sns.countplot(data=weather_data, x='will_it_rain')
plt.show()

# Plot the count of snow forecasts
sns.countplot(data=weather_data, x='will_it_snow')
plt.show()

# Drop the snow column as it has constant variance
weather_data.drop(['will_it_snow'], axis=1, inplace=True)

# Group the data by states
grouped_state_data = weather_data.groupby('state')

# State wise analysis
state_data = weather_data.groupby('state').mean().reset_index()

# Plotting each states mean temperature in degree celsius
plt.figure(figsize=(8,6))
plt.barh(y = state_data.sort_values(by='temp_c')['state'], width = state_data.sort_values(by='temp_c')['temp_c'])
plt.xlabel("State")
plt.ylabel("Temperature in degree Celsius")
plt.show()

# store the name of the states in a list and sort them alphabetically
states = [state for state in weather_data['state'].value_counts().index]
states.sort()

continuous_distributions.append('will_it_rain')
continuous_distributions.append('cloud')

columns_to_analyze_1 = continuous_distributions
columns_to_analyze_2 = ['condition', 'wind_dir']

# Time Series Analysis
def statewise_data_analysis(columns_to_analyze_1, columns_to_analyze_2):

  label_map = ['Day {}'.format(i) for i in range(1, 15)]
  ticks = [i for i in range(1,15)]

  # Plotting average daily distributions from 15/10/2021-22/10/2021

  for i in range(len(states)):

    # Perform analysis for each state
    state = states[i]
    state_i_data = grouped_state_data.get_group(state)
    state_i_data['time'] = pd.to_datetime(state_i_data['time'])

    print("=========================================================================={}'s ANALYSIS=================================================================================".format(state.upper()))
    print()

    # Group the data by time
    time_data = state_i_data.groupby('time').mean()

    # Plot the daily average of all the continuous distributions and numerical variable
    columns_to_analyze_1 = continuous_distributions
    plt.figure(figsize=(25, 25))
    for i, col in enumerate(columns_to_analyze_1):
      plt.subplot(7, 2, i + 1)
      plt.plot(time_data[col])
      #plt.xticks(rotation=-45)
      plt.xlabel("Date")
      plt.ylabel(col)
    plt.show()

    # Plotting categorical variable across states
    plt.figure(figsize=(25, 12))
    for j in range(len(columns_to_analyze_2)):
      plt.subplot(1, 2, j+1)
      curr_column = columns_to_analyze_2[j]
      data_to_plot = state_i_data[curr_column].value_counts().sort_values()
      plt.barh(y=data_to_plot.index, width=data_to_plot.values)
      plt.xlabel("Frequency")
      plt.ylabel(curr_column)
    plt.show()

    print("====================================================================================================================================================================================".format(state.upper()))
    print()

# call the function
# statewise_data_analysis(columns_to_analyze_1, columns_to_analyze_2)

# Analysizng features and their dependence on predicting whether it will rain or not
print(weather_data.groupby('will_it_rain').mean())

columns_to_analyze_3 = list(weather_data.groupby('will_it_rain').mean().columns)
columns_to_analyze_3.remove('time_epoch')



# Analysizng features and their dependence on predicting whether it will rain or not for each state
def statewise_rain_factor_analysis(columns_to_analyze_3):

  # Plotting boxplots for the feature 'will_it_rain' against the average of all other continuous features for each state

  for i in range(len(states)):

    # Perform analysis for each state
    state = states[i]
    state_i_data = grouped_state_data.get_group(state)

    print("=========================================================================={}'s ANALYSIS=================================================================================".format(state.upper()))
    print()
    plt.figure(figsize=(25, 25))
    for i, col in enumerate(columns_to_analyze_3):
      plt.subplot(7, 2, i + 1)
      sns.boxplot(data=state_i_data, x='will_it_rain', y=col)
      #plt.xticks(rotation=-45)
      plt.xlabel("Will it Rain 0 : No ; 1 : Yes")
      plt.ylabel(col)
    plt.show()
    
    print("====================================================================================================================================================================================".format(state.upper()))
    print()

# Call the above function
# statewise_rain_factor_analysis(columns_to_analyze_3)

print(continuous_distributions)

continuous_distributions.remove('will_it_rain')
continuous_distributions.remove('cloud')

# Plotting scatter plots

'''Plotting each continuous distribution against every other continuous distribution'''
print(weather_data[continuous_distributions].head())
# for i in range(len(continuous_distributions)):

#   if i != len(continuous_distributions)-1:
#     curr_column = continuous_distributions[i]
#     print("================================================= {} SCATTER PLOT ===================================================".format(curr_column))
#     print()
#     num_rows = len(continuous_distributions) - i
#     for j in range(i+1, len(continuous_distributions)):
#       next_column = continuous_distributions[j]
#       plt.figure(figsize=(8,6))
#       sns.scatterplot(data=weather_data, x=curr_column, y=next_column)
#       plt.show()
#     print("=======================================================================================================================")
#     print()

# PLot the correlation heatmap
corr_matrix = weather_data.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cbar=False)
plt.show()

print(weather_data.head())


# Pre Processing

# Encode the categorical text values
cols_to_encode = ['condition', 'wind_dir', 'state', 'city']
for col in cols_to_encode:
  le = LabelEncoder()
  weather_data[col] = le.fit_transform(weather_data[col])

print(weather_data.head())

# drop the time column
weather_data.drop('time', axis=1, inplace=True)

# Scale the data
X = weather_data.drop('will_it_rain', axis=1)
y = weather_data['will_it_rain']

X_columns = X.columns
ss = StandardScaler()
X_scaled_arr = ss.fit_transform(X)

# Create a new dataframe called X_scaled
X_scaled = pd.DataFrame(X_scaled_arr, columns=X_columns)

print(X_scaled.head())

# Feature Selection

rfe_model = RandomForestClassifier()
rfe = RFE(rfe_model, verbose=3)
X_rfe = rfe.fit_transform(X_scaled, y)

# Extract the important features
feature_series = pd.Series(rfe.support_, index=X_columns)
important_features = list(feature_series[feature_series == True].index)

print(important_features)

# Prepare the final dataset
X_final = X_scaled[important_features]

# Classification

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# We are going to chose a model which gives maximum recall, in case of tie we are going to see which one gives maximum TPs.

# 1. Compute Recall Score
def compute_recall_score(model_dict, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
  model_name = list(model_dict.keys())[0]
  model_obj = list(model_dict.values())[0]

  # Make predictions

  # 1. Training predictions
  train_preds = model_obj.predict(X_train)

  # 2. Testing predictions
  test_preds = model_obj.predict(X_test)

  # Compute Recall Score

  # 1. Training Score
  train_recall = recall_score(y_train, train_preds)

  # 2. Testing score
  test_recall = recall_score(y_test, test_preds)

  # Display the result
  result_arr = np.array([train_recall, test_recall])
  result_df = pd.DataFrame(data = result_arr.reshape(1,2), columns = ['Train_Recall', 'Test_Recall'], index=[model_name])

  return result_df

# Plot the Confusion Matrix
def plot_confusion_matrix(model_dict, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):

  model_name = list(model_dict.keys())[0]
  model_obj = list(model_dict.values())[0]

  # Make predictions

  # 1. Training predictions
  train_preds = model_obj.predict(X_train)

  # 2. Testing predictions
  test_preds = model_obj.predict(X_test)

  # Compute Recall Score

  # 1. Training Score
  train_recall = confusion_matrix(y_train, train_preds)

  # 2. Testing score
  test_recall = confusion_matrix(y_test, test_preds)

  # Plot the heatmap
  fig, ax = plt.subplots(1, 2, figsize=(15,8))

  # PLot the trainig matrix
  sns.heatmap(train_recall, annot=True, cbar=False, ax=ax[0], fmt='g')
  ax[0].set_xlabel("Predicted Values")
  ax[0].set_ylabel("Actual Values")
  ax[0].set_title("Training Set Results")

  # Plot the testing matrix
  sns.heatmap(test_recall, annot=True, cbar=False, ax=ax[1], fmt='g')
  ax[1].set_xlabel("Predicted Values")
  ax[1].set_ylabel("Actual Values")
  ax[1].set_title("Testing Set Results")

  fig.show()
  plt.show()

# 1. Baseline Model -> KNN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'KNN' : knn_clf}
knn_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(knn_results)

# 2. Logistic Regression
lr_clf = LogisticRegressionCV()
lr_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'LogisticRegression' : lr_clf}
lr_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(lr_results)

# 3. Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'DecisionTree' : dt_clf}
dt_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(dt_results)

# 4. SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'SVM' : svm_clf}
svm_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(svm_results)

# 5. Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'RandomForest' : rf_clf}
rf_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(rf_results)

# 6. Extra Trees
ext_clf = ExtraTreesClassifier()
ext_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'ExtraTrees' : ext_clf}
ext_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(ext_results)

# 7. XGBoost
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'XGBoost' : xgb_clf}
xgb_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(xgb_results)

# 8. LightGBM
lgbm_clf = LGBMClassifier()
lgbm_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'LightGBM' : lgbm_clf}
lgbm_results = compute_recall_score(model_dict=model_dict)
plot_confusion_matrix(model_dict=model_dict)

print(lgbm_results)






# Concatenate the results
final_results = pd.concat((knn_results, lr_results, 
                           svm_results, dt_results, 
                           rf_results, ext_results, 
                           xgb_results, lgbm_results), axis=0).sort_values(by='Test_Recall', ascending=False)


print(final_results)



# Hyperparameter Tuning
# Tune the top 2 models i.e LightGBM and ExtraTrees

# 1. Light GBM
lgbm_params = {"num_leaves" : [31, 50, 70, 90, 110],
               "max_depth" : [10, 20, 30, 40, 50, 60],
               "learning_rate" : [0.1, 0.5, 1, 1.5, 2.0],
               "n_estimators" : [100, 150, 200, 250, 300, 350],
               "reg_alpha" : [0.0, 0.25, 0.50, 0.75, 1.0, 2.0],
               "reg_lambda" : [0.0, 0.25, 0.50, 0.75, 1.0, 2.0],
               "colsample_bytree" : [0.0, 0.25, 0.50, 0.75, 1.0]
               }

lgbm_clf_2 = LGBMClassifier()

# using randomised search cv
rf_lgbm_clf = RandomizedSearchCV(lgbm_clf_2, lgbm_params, n_iter=20, scoring='recall', n_jobs=-1, cv=3, verbose=3, random_state=0)
rf_lgbm_clf.fit(X_train, y_train)


# Store the Best estimator
lgbm_best = rf_lgbm_clf.best_estimator_

# Generate Results
model_dict = {'LGBM_Tuned' : lgbm_best}
lgbm_best_results = compute_recall_score(model_dict)
plot_confusion_matrix(model_dict)

print(lgbm_best_results)

# 2. Extra Trees
ext_params = { "max_depth" : [10, 20, 30, 40, 50, 60],
               "criterion" : ['gini', 'entropy'],
               "n_estimators" : [100, 150, 200, 250, 300, 350],
               "max_features" : ["auto", "sqrt", "log2"],
               "min_samples_split" : [2, 4, 6, 8, 10],
               "min_samples_leaf" : [1, 2, 3, 4, 5, 6, 7],
               "bootstrap" : [True, False]
              }

ext_clf_2 = ExtraTreesClassifier()

# using randomised search cv
rf_ext_clf = RandomizedSearchCV(ext_clf_2, ext_params, n_iter=20, scoring='recall', n_jobs=-1, cv=3, verbose=3, random_state=0)
print(rf_ext_clf.fit(X_train, y_train))

# Store the Best estimator
ext_best = rf_ext_clf.best_estimator_

# Generate Results
model_dict = {'ExtraTrees_Tuned' : ext_best}
ext_best_results = compute_recall_score(model_dict)
plot_confusion_matrix(model_dict)

print(ext_best_results)

lgbm_best.fit(X_scaled, y)

model_file = './LightGBM.pkl'
pickle.dump(lgbm_best, open(model_file, 'wb'))


