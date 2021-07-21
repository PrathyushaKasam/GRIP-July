#importing all the required libraries

import numpy as np
import pandas as pd
from matplotlib import style
%matplotlib inline 
from matplotlib import pyplot as plt
#importing the dataset
link ="http://bit.ly/w-data"
data_set =pd.read_csv(link)
print("Data imported successfully")
print(data_set.shape) #prints number of rows and columns in our dataset
data_set.head(26)
data_set.info() #summary of the data
#plotting the graph for distribution of scores
data_set.plot(x='Hours',y='Scores',style='o')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.title('Study Hours vs Student Score(%)')
plt.show()
#preparing the data
x= data_set.iloc[:,:-1].values
y= data_set.iloc[:,1].values
x
y
#training the model
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test =train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("The data has been trained")
plot= regressor.coef_*x+regressor.intercept_
style.use('seaborn')
plt.scatter(x_train,y_train,color='blue')
plt.plot(x,plot,color='grey',linewidth=3)
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.title('Study Hours vs Student Score(%)')
plt.show()
#for the data we split for testing we are finding out the predicted scores for that data
print(x_test) #testing data in hours
scores_pred= regressor.predict(x_test) #predicting the scores
#comparing the Actual scores Vs the Predicted scores
diff = pd.DataFrame({'Actual Scores' :y_test,'Predicted Scored:':scores_pred})
diff
#predicted score of a student studying for 9.25 hours a day
diff=np.array(9.25)
diff=diff.reshape(-1,1)
hours=regressor.predict(diff)
print('The score of a student who studies for 9.25 hours a day is {}'.format(hours))
from sklearn import metrics
print('The Mean Absolute Error:' ,metrics.mean_absolute_error(y_test,scores_pred))