import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
df

df.head(10)
df.info()
df.describe()
df.describe(include='object')
print(f'DF with {df.duplicated().sum()} duplicated row in the dataset.')

sns.heatmap(df.corr(), annot=True)
sns.histplot(data=df, x="charges")
sns.histplot(data=df, x="charges", hue='sex')

df.groupby(['sex'])['charges'].describe()
sns.histplot(data=df, x="charges", hue='smoker')

df.groupby(['smoker'])['charges'].describe()
sns.histplot(data=df, x="charges", hue='region')

df.groupby(['region'])['charges'].describe()

sns.scatterplot(data=df, x="bmi", y='charges')
sns.scatterplot(data=df, x="age", y='charges')
sns.scatterplot(data=df, x="age", y='charges', hue='smoker')
sns.scatterplot(data=df, x="age", y='charges', hue='sex')
sns.scatterplot(data=df, x="age", y='charges', hue='region')

df_age2 = df
df_age2['age2'] = df['age'] * df['age']

sns.scatterplot(data=df_age2, x="age2", y='charges', hue='smoker')

df = pd.get_dummies(df, drop_first=True)
df.head()

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

print('Intercept:',model_simple.intercept_)
print('Columns:', model_simple.feature_names_in_)
print('Slope:', model_simple.coef_)

y_pred = model_simple.predict(X_test)
y_pred

print('MAE', metrics.mean_absolute_error(y_test, y_pred))
print('MSE', metrics.mean_squared_error(y_test, y_pred))
print('RMSE', metrics.mean_squared_error(y_test, y_pred, squared=False))

filename='../models/simple_model.sav'
pickle.dump(model_simple, open(filename, 'wb'))

X_int = sm.add_constant(X_train) 
modelo_alt = sm.OLS(y_train, X_int)
results = modelo_alt.fit()
results.summary()

# Optimization
model2 = LinearRegression(fit_intercept=False)
y_pred2 = model2.fit(X_train, y_train).predict(X_test)

# Performance metrics
print('MAE', metrics.mean_absolute_error(y_test, y_pred2))
print('MSE', metrics.mean_squared_error(y_test, y_pred2))
print('RMSE', metrics.mean_squared_error(y_test, y_pred2,squared=False))

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

param_grid = {
    'polynomialfeatures__degree': np.arange(4), # polynomial up to 4
    'linearregression__fit_intercept': [True, False], # with and without intercept
    'linearregression__normalize': [True, False]
} # normalize and not normalize

grid = GridSearchCV(PolynomialRegression(), param_grid) # 5 folds
grid.fit(X_train, y_train)
grid.best_params_

model3 = grid.best_estimator_
y_pred3 = model3.fit(X_train, y_train).predict(X_test)

print('MAE', metrics.mean_absolute_error(y_test, y_pred3))
print('MSE', metrics.mean_squared_error(y_test, y_pred3))
print('RMSE', metrics.mean_squared_error(y_test, y_pred3,squared=False))

filename='../models/model_second.sav'
pickle.dump(model2, open(filename, 'wb'))

filename='../models/model_final.sav'
pickle.dump(model3, open(filename, 'wb'))


