# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.

2. Download and upload required csv file or dataset for predecting Employee Churn

3. Initialize variables with required features.

4. And implement Decision tree classifier to predict Employee Churn


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Lokesh N
RegisterNumber:  212222100023
*/
```
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
### Initial Dataset:
![image](https://github.com/lokeshnarayanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393019/57e36685-c3d0-4865-be4a-cddb95775727)

### Mean Squared Error:
![image](https://github.com/lokeshnarayanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393019/f73a25c5-4f4b-41a9-8828-4a425e31194b)

### R2 (variance):
![image](https://github.com/lokeshnarayanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393019/d6182b02-f5b5-406a-8db4-d1328610cd3c)

### Data prediction:
![image](https://github.com/lokeshnarayanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393019/b41c4550-4f17-4e71-9efa-b30f81ace0c7)

### Decision Tree:
![Screenshot 2024-04-03 103909](https://github.com/lokeshnarayanan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393019/50c3e6e0-2e80-417e-be04-6a48a99dd860)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
