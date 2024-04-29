# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vikaash K S
RegisterNumber: 212223240179
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("/content/Placement_Data.csv")
d
d=d.drop("sl_no",axis=1)
d=d.drop("salary",axis=1)
d["gender"]=d["gender"].astype("category")
d["ssc_b"]=d["ssc_b"].astype("category")
d["hsc_b"]=d["hsc_b"].astype("category")
d["degree_t"]=d["degree_t"].astype("category")
d["workex"]=d["workex"].astype("category")
d["specialisation"]=d["specialisation"].astype("category")
d["status"]=d["status"].astype("category")
d["hsc_s"]=d["hsc_s"].astype("category")
d.dtypes
d["gender"]=d["gender"].cat.codes
d["ssc_b"]=d["ssc_b"].cat.codes
```
```
d["hsc_b"]=d["hsc_b"].cat.codes
d["degree_t"]=d["degree_t"].cat.codes
d["workex"]=d["workex"].cat.codes
d["specialisation"]=d["specialisation"].cat.codes
d["status"]=d["status"].cat.codes
d["hsc_s"]=d["hsc_s"].cat.codes
d
X=d.iloc[:,:-1].values
Y=d.iloc[:,-1].values
Y
theta=np.random.randn(x.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta -=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)






```
## Output:
![exp 5 op1](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/2c2dbe27-f772-421e-89e5-7fd0d61c82d4)

![exp 5 op2](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/be6d613b-f9bf-4c8f-af62-807b7a0dac86)

![exp 5 op3](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/c21f0f12-0df3-43c5-95c7-84c0c9b78642)

![exp 5 op4](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/b9b4c4bf-c563-4e20-b629-5444d0e62d66)

![exp 5 op5](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/fca05324-e1f8-4538-961f-84e109c50011)

![exp 5 op6](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/99d7c850-224f-4709-849c-8c1a2903bd11)

![exp 5 op6](https://github.com/Vikaash19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148514589/99d7c850-224f-4709-849c-8c1a2903bd11)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
