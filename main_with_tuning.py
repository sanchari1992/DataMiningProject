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
from prophet import Prophet

import pickle

import warnings
warnings.filterwarnings('ignore')


# Code Block

df_forecast = pd.read_csv('data/forecast_data.csv')
df_loc = pd.read_csv('data/df_loc.csv')
print(df_loc.head())

# Plotting the cities for which we have the data for
map_India = [20.5937, 78.9629]

India_folium = folium.Map(map_India) # Creates a map
def plot_map(df):
  # The function will add markers at the coordinates present in our dataframe
  folium.Marker(location=[df.lat, df.lon]).add_to(India_folium)

# Apply the function to the dataframe
df_loc.apply(plot_map, axis=1)
India_folium.fit_bounds([[30.3753, 69.3451], [7.8731, 80.7718]])
map_file1 = "map1.html"
India_folium.save(map_file1)
webbrowser.open('file://' + os.path.realpath(map_file1))

print(df_forecast.head())

# Extract the condition text from the json string
df_forecast['condition'] = df_forecast['condition'].apply(lambda x : eval(x)['text'])

print(df_forecast.head())

# Many columns are same just with differnt units, for ex wind_mph and wind_kph. So, it's better to remove the redundancy
unused_col_df = ['temp_f', 'wind_kph', 'pressure_in', 'precip_in', 'feelslike_f', 'windchill_f', 'heatindex_f', 'dewpoint_f', 'vis_miles', 'chance_of_rain', 'chance_of_snow','gust_kph']

# Drop the redundant information
df_forecast.drop(unused_col_df, axis=1, inplace=True)

print(df_forecast.head())

# Visualise various continuous distributions
cont_data_col_df = ['temp_c', 'wind_mph', 
                            'wind_degree', 'pressure_mb', 
                            'precip_mm', 'humidity', 
                            'feelslike_c', 'windchill_c', 
                            'heatindex_c', 'dewpoint_c', 
                            'vis_km', 'gust_mph']

plt.style.use('seaborn')
df_forecast['precip_mm'].hist(bins=25)
plt.xlabel("Precipitation (in mm)")
plt.xticks(np.arange(0,21,1))
plt.show()

# Plotting categorical discrete variables

# 1. Plotting days and nights count
plt.figure(figsize=(6,4))
sns.countplot(data=df_forecast, x='is_day')
plt.show()

df_forecast['condition'].value_counts().sort_values().plot(kind='barh', figsize=(6,4))
plt.xlabel("Weather Conditions")
plt.ylabel("Frequency")
plt.grid()
plt.show()

df_forecast['wind_dir'].value_counts().sort_values().plot(kind='bar', figsize=(6,4))
plt.xlabel("Wind Direction")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Plot state distribution
df_forecast['state'].value_counts().sort_values().plot(kind='barh', figsize=(6,4))
plt.xlabel("Frequency")
plt.ylabel("State")
plt.show()

# Plot the count of rainy forecasts
sns.countplot(data=df_forecast, x='will_it_rain')
plt.show()

# Plot the count of snow forecasts
sns.countplot(data=df_forecast, x='will_it_snow')
plt.show()

# Drop the snow column as it has constant variance
df_forecast.drop(['will_it_snow'], axis=1, inplace=True)

# Group the data by states
state_df = df_forecast.groupby('state')

# State wise analysis
state_mean_df = df_forecast.groupby('state').mean().reset_index()

# Plotting each states mean temperature in degree celsius
plt.figure(figsize=(8,6))
plt.barh(y = state_mean_df.sort_values(by='temp_c')['state'], width = state_mean_df.sort_values(by='temp_c')['temp_c'])
plt.xlabel("State")
plt.ylabel("Temperature in degree Celsius")
plt.show()

# store the name of the states in a list and sort them alphabetically
states = [state for state in df_forecast['state'].value_counts().index]
states.sort()

cont_data_col_df.append('will_it_rain')
cont_data_col_df.append('cloud')

columns_to_analyze_1 = cont_data_col_df
columns_to_analyze_2 = ['condition', 'wind_dir']

# Analysing features and their dependence on predicting whether it will rain or not
print(df_forecast.groupby('will_it_rain').mean())

columns_to_analyze_3 = list(df_forecast.groupby('will_it_rain').mean().columns)
columns_to_analyze_3.remove('time_epoch')


print(cont_data_col_df)

cont_data_col_df.remove('will_it_rain')
cont_data_col_df.remove('cloud')

# Plotting scatter plots

'''Plotting each continuous distribution against every other continuous distribution'''
print(df_forecast[cont_data_col_df].head())

# Plot the correlation heatmap
corr_matrix = df_forecast.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cbar=False)
plt.show()

print(df_forecast.head())


# Pre Processing

# Encode the categorical text values
col_enc_df = ['condition', 'wind_dir', 'state', 'city']
for col in col_enc_df:
  le = LabelEncoder()
  df_forecast[col] = le.fit_transform(df_forecast[col])

print(df_forecast.head())

# drop the time column
df_forecast.drop('time', axis=1, inplace=True)

# Scale the data
X = df_forecast.drop('will_it_rain', axis=1)
y = df_forecast['will_it_rain']

X_col = X.columns
scalermodel = StandardScaler()
X_scl1 = scalermodel.fit_transform(X)

# Create a new dataframe called X_scl2
X_scl2 = pd.DataFrame(X_scl1, columns=X_col)

print(X_scl2.head())

# Feature Selection

randomforest = RandomForestClassifier()
rfe = RFE(randomforest, verbose=3)
X_rfe = rfe.fit_transform(X_scl2, y)

# Extract the important features
feature_series = pd.Series(rfe.support_, index=X_col)
important_features = list(feature_series[feature_series == True].index)

print(important_features)

# Prepare the final dataset
X_final = X_scl2[important_features]

# Classification

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# We are going to chose a model which gives maximum recall, in case of tie we are going to see which one gives maximum TPs.

# 1. Compute Recall Score
def recall_function(model_dict, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
  name_arr = list(model_dict.keys())[0]
  object_arr = list(model_dict.values())[0]

  # Make predictions

  # 1. Training predictions
  train_preds = object_arr.predict(X_train)

  # 2. Testing predictions
  test_preds = object_arr.predict(X_test)

  # Compute Recall Score

  # 1. Training Score
  train_recall = recall_score(y_train, train_preds)

  # 2. Testing score
  test_recall = recall_score(y_test, test_preds)

  # Display the result
  result_arr = np.array([train_recall, test_recall])
  result_df = pd.DataFrame(data = result_arr.reshape(1,2), columns = ['Train_Recall', 'Test_Recall'], index=[name_arr])

  return result_df

# 2. Plot the Confusion Matrix
def cm_function(model_dict, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):

  name_arr = list(model_dict.keys())[0]
  object_arr = list(model_dict.values())[0]

  # Make predictions

  # 1. Training predictions
  train_preds = object_arr.predict(X_train)

  # 2. Testing predictions
  test_preds = object_arr.predict(X_test)

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
knn_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(knn_results)


# 2. Logistic Regression
lr_clf = LogisticRegressionCV()
lr_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'LogisticRegression' : lr_clf}
lr_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(lr_results)

# 3. Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'DecisionTree' : dt_clf}
dt_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(dt_results)

# 4. SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'SVM' : svm_clf}
svm_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(svm_results)

# 5. Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'RandomForest' : rf_clf}
rf_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(rf_results)

# 6. Extra Trees
ext_clf = ExtraTreesClassifier()
ext_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'ExtraTrees' : ext_clf}
ext_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(ext_results)

# 7. XGBoost
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'XGBoost' : xgb_clf}
xgb_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(xgb_results)

# 8. LightGBM
lgbm_clf = LGBMClassifier()
lgbm_clf.fit(X_train, y_train)

# Compute Scores and plot confusion matrix
model_dict={'LightGBM' : lgbm_clf}
lgbm_results = recall_function(model_dict=model_dict)
cm_function(model_dict=model_dict)

print(lgbm_results)


# 9. Prophet





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
lgbm_best_results = recall_function(model_dict)
cm_function(model_dict)

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
ext_best_results = recall_function(model_dict)
cm_function(model_dict)

print(ext_best_results)

lgbm_best.fit(X_scl2, y)

model_file = './LightGBM.pkl'
pickle.dump(lgbm_best, open(model_file, 'wb'))


