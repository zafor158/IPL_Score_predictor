# Importing Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
ipl_df = pd.read_csv('ipl_data.csv')
print(f"Dataset successfully Imported of Shape : {ipl_df.shape}")

"""# Exploratory Data Analysis"""

# First 5 Columns Data
ipl_df.head()

# Describing the ipl_dfset
ipl_df.describe()

# Information about Each Column
ipl_df.info()

# Number of Unique Values in each column
ipl_df.nunique()

# ipl_df types of all Columns
ipl_df.dtypes

"""# Data Cleaning

#### Removing Irrelevant Data columns
"""

# Names of all columns
ipl_df.columns

"""Here, we can see that columns _['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']_ won't provide any relevant information for our model to train.Because our model is to predict the overall a 20 overs score for a perticular team.So we really dont care about who the batsman is,who the bowler is ,who the striker end, non striker end and so on.

##Checking null values percentage of the data set
"""

percentage_null = ipl_df.isna().mean() * 100
print(percentage_null)

"""##Checking duplicated values"""

duplicates = ipl_df[ipl_df.duplicated()]
print(duplicates)

"""##Dropping Irrelevant columns from the dataset"""

irrelevant = ['mid', 'date', 'venue','batsman', 'bowler', 'striker', 'non-striker']
print(f'Before Removing Irrelevant Columns : {ipl_df.shape}')
ipl_df = ipl_df.drop(irrelevant, axis=1) # Drop Irrelevant Columns
print(f'After Removing Irrelevant Columns : {ipl_df.shape}')
ipl_df.head()

"""##List of Unique Team"""

ipl_df['bat_team'].unique()

"""#### Keeping only Consistent Teams

"""

# Define Consistent Teams
const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
              'Delhi Daredevils', 'Sunrisers Hyderabad']

print(f'Before Separating Inconsistent Teams : {ipl_df.shape}')
ipl_df = ipl_df[(ipl_df['bat_team'].isin(const_teams)) & (ipl_df['bowl_team'].isin(const_teams))]
print(f'After Removing Irrelevant Columns : {ipl_df.shape}')
print(f"Consistent Teams : \n{ipl_df['bat_team'].unique()}")
ipl_df.head()

"""Plotting a Correlation Matrix of current data"""

from seaborn import heatmap
heatmap(data=ipl_df.corr(), annot=True)

"""# Data Preprocessing and Encoding

#### Performing Label Encoding
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()#This line creates an instance of the LabelEncoder class and assigns it to the variable le.
#The LabelEncoder is used to encode categorical labels with numerical values.
for col in ['bat_team', 'bowl_team']:
  ipl_df[col] = le.fit_transform(ipl_df[col])
ipl_df.head(15)

"""#### Performing One Hot Encoding and Column Transformation"""

from sklearn.compose import ColumnTransformer
#ColumnTransformer is a useful tool for applying different transformers to different columns of a pandas DataFrame.
columnTransformer = ColumnTransformer([('encoder',
                                        OneHotEncoder(),
                                        [0, 1])],
                                      remainder='passthrough')

ipl_df = np.array(columnTransformer.fit_transform(ipl_df))
ipl_df

"""Save the Numpy Array in a new DataFrame with transformed columns"""

cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
              'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
              'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
              'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
              'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
              'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
       'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(ipl_df, columns=cols)

# Encoded Data
df.head()

"""# Model Building

## Prepare Train and Test Data
"""

features = df.drop(['total'], axis=1)
labels = df['total']

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)##
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

"""## ML  Algorithms"""

models = dict() # models is an empty dictionary here. Actually its a container for storing various models info.

"""#### 1. Decision Tree Regressor"""

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
# Train Model
tree.fit(train_features, train_labels)

# Evaluate Model
train_score_tree = float(tree.score(train_features, train_labels) * 100)
test_score_tree = float(tree.score(test_features, test_labels) * 100)
# print(f'Train Score : {train_score_tree[:5]}%\nTest Score : {test_score_tree[:5]}%')
models["tree"] = test_score_tree
print(train_score_tree)
print(test_score_tree)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("---- Decision Tree Regressor - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mean_absolute_error(test_labels, tree.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mean_squared_error(test_labels, tree.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mean_squared_error(test_labels, tree.predict(test_features)))))

"""#### Linear Regression"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
linreg = LinearRegression()
# Train Model
linreg.fit(train_features, train_labels)

# Evaluate Model
train_score_linreg = float(linreg.score(train_features, train_labels) * 100)
test_score_linreg = float(linreg.score(test_features, test_labels) * 100)

models["linreg"] = test_score_linreg
print(train_score_linreg)
print(test_score_linreg)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("---- Linear  Regressor - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mean_absolute_error(test_labels, linreg.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mean_squared_error(test_labels, linreg.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mean_squared_error(test_labels, linreg.predict(test_features)))))

"""#### Random Forest Regression"""

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
# Train Model
forest.fit(train_features, train_labels)

# Evaluate Model
train_score_forest = float(forest.score(train_features, train_labels)*100)
test_score_forest = float(forest.score(test_features, test_labels)*100)

models["forest"] = test_score_forest
print(train_score_forest)
print(test_score_forest )

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("---- Random Forest Regressor - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mean_absolute_error(test_labels, forest.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mean_squared_error(test_labels, forest.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mean_squared_error(test_labels, forest.predict(test_features)))))

"""## Best Model"""

import matplotlib.pyplot as plt
model_names = list(models.keys())
accuracy = list(map(float, models.values()))
# creating the bar plot
plt.bar(model_names, accuracy)
plt.title('Model Vs Accuracy')
plt.xlabel('Model Name');
plt.ylabel('Accuracy Score')

"""From above, we can see that **Random Forest** performed the best, closely followed by **Decision Tree**. So we will be choosing Random Forest for the final model

# Predictions
"""

def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
  prediction_array = []
  # Batting Team
  if batting_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
  prediction_array = np.array([prediction_array])
  pred = model.predict(prediction_array)
  return int(round(pred[0]))## round function rounds the predicted value to the nearest integer.

"""### Test 1
- Batting Team : **Delhi Daredevils**
- Bowling Team : **Chennai Super Kings**
- Final Score : **147/9**
"""

batting_team = 'Delhi Daredevils'
bowling_team = 'Chennai Super Kings'
actual_score = 147
features = {'overs': 10.2, 'runs': 68, 'wickets': 3, 'runs_last_5': 29, 'wickets_last_5': 1}

predicted_score = score_predict(batting_team, bowling_team, **features)## **features represents the dictionary here

# accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100

print(f'Predicted Score: {predicted_score} || Actual Score: {actual_score}')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""### Test 2
- Batting Team : **Mumbai Indians**
- Bowling Team : **Kings XI Punjab**
- Final Score : **176/7**
"""

batting_team='Mumbai Indians'
bowling_team='Kings XI Punjab'
actual_score=176
features = {'overs': 12.3, 'runs': 113, 'wickets': 2, 'runs_last_5': 55, 'wickets_last_5': 0}


predicted_score = score_predict(batting_team, bowling_team, **features)


# accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100


print(f'Predicted Score : {predicted_score} || Actual Score : 176')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""### Test 3
- Batting Team : **Kings XI Punjab**
- Bowling Team : **Rajasthan Royals**
- Final Score : **185/4**
<br/>
These Test Was done before the match and final score were added later.
"""

batting_team="Kings XI Punjab"
bowling_team="Rajasthan Royals"
actual_score=185
features = {'overs': 14.0, 'runs': 118, 'wickets': 1, 'runs_last_5': 45, 'wickets_last_5': 0}

predicted_score = score_predict(batting_team, bowling_team, **features)


# accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100


print(f'Predicted Score : {predicted_score} || Actual Score : 185')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""### Test 4
- Batting Team : **Kolkata Knight Riders**
- Bowling Team : **Chennai Super Kings**
- Final Score : **172/5**
"""

batting_team="Kolkata Knight Riders"
bowling_team="Chennai Super Kings"
actual_score = 172
features = {'overs': 18.0, 'runs': 150, 'wickets': 4, 'runs_last_5': 57, 'wickets_last_5': 1}


predicted_score = score_predict(batting_team, bowling_team, **features)


#accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100


print(f'Predicted Score : {predicted_score} || Actual Score : 172')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""### Test 5
- Batting Team : **Delhi Daredevils**
- Bowling Team : **Mumbai Indians**
- Final Score : **110/7**
"""

batting_team='Delhi Daredevils'
bowling_team='Mumbai Indians'
actual_score = 110
features = {'overs': 18.0, 'runs': 96, 'wickets': 8, 'runs_last_5': 18, 'wickets_last_5': 4}


predicted_score = score_predict(batting_team, bowling_team, **features)

#accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100

print(f'Predicted Score : {predicted_score} || Actual Score : 110')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""### Test 6
- Batting Team : **Sunrisers Hyderabad**
- Bowling Team : **Royal Challengers Banglore**
- Final Score : **146/10**
"""

batting_team='Sunrisers Hyderabad'
bowling_team='Royal Challengers Bangalore'
actual_score = 146
features = {'overs': 10.5, 'runs': 67, 'wickets': 3, 'runs_last_5': 29, 'wickets_last_5': 1}


predicted_score = score_predict(batting_team, bowling_team, **features)

#accuracy percentage
accuracy_percentage = (1 - abs(predicted_score - actual_score) / actual_score) * 100

print(f'Predicted Score : {predicted_score} || Actual Score : 146')
print(f'Prediction Accuracy: {accuracy_percentage:.2f}%')

"""# Export Model"""

import pickle
filename = "ml_model.pkl"
pickle.dump(forest, open(filename, "wb"))

"""##Streamlit app"""



#import the libraries


import math
import numpy as np
import pickle
import streamlit as st

#SET PAGE WIDE
st.set_page_config(page_title='IPL_Score_Predictor',layout="centered")

#Get the ML model

filename='ml_model.pkl'
model = pickle.load(open(filename,'rb'))

#Title of the page with CSS

st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor Using machine Learning </h1>", unsafe_allow_html=True)

#Add background image

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.app.goo.gl/r8nLcL7P42s4kqm46");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#Add description

with st.expander("Description"):
    st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match. To make sure the model results accurate score and some reliability the minimum no. of current overs considered is greater than 5 overs.

 """)

# SELECT THE BATTING TEAM


batting_team= st.selectbox('Select the Batting Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))

prediction_array = []
  # Batting Team
if batting_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
elif batting_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
elif batting_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
elif batting_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
elif batting_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
elif batting_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
elif batting_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
elif batting_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]

#SELECT BOWLING TEAM

bowling_team = st.selectbox('Select the Bowling Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))
if bowling_team==batting_team:
    st.error('Bowling and Batting teams should be different')
# Bowling Team
if bowling_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
elif bowling_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
elif bowling_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
elif bowling_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
elif bowling_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
elif bowling_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]


col1, col2 = st.columns(2)

#Enter the Current Ongoing Over
with col1:
    overs = st.number_input('Enter the Current Over',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
    if overs-math.floor(overs)>0.5:
        st.error('Please enter valid over input as one over only contains 6 balls')
with col2:
#Enter Current Run
    runs = st.number_input('Enter Current runs',min_value=0,max_value=354,step=1,format='%i')


#Wickets Taken till now
wickets =st.slider('Enter Wickets fallen till now',0,9)
wickets=int(wickets)

col3, col4 = st.columns(2)

with col3:
#Runs in last 5 over
    runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs',min_value=0,max_value=runs,step=1,format='%i')

with col4:
#Wickets in last 5 over
    wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs',min_value=0,max_value=wickets,step=1,format='%i')

#Get all the data for predicting

prediction_array = prediction_array + [runs, wickets, overs, runs_in_prev_5,wickets_in_prev_5]
prediction_array = np.array([prediction_array])
predict = model.predict(prediction_array)


if st.button('Predict Score'):
    #Call the ML Model
    my_prediction = int(round(predict[0]))

    #Display the predicted Score Range
    x=f'PREDICTED MATCH SCORE : {my_prediction-5} to {my_prediction+5}'
    st.success(x)

