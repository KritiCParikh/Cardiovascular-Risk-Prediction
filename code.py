# -*- coding: utf-8 -*-

# Loading the libraries
"""

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# split the data into train and test
from sklearn.model_selection import train_test_split

# to scale the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# to do PCA
from sklearn.decomposition import PCA

# for one hot encoding
from sklearn.preprocessing import OneHotEncoder

# for imputation
from sklearn.impute import SimpleImputer, KNNImputer

# for building models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

# for metrics
import sklearn.metrics as metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


# for cross validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# To tune a model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# To supress the warning messages generated
import warnings

cd

"""# Loading the data and getting some basic understanding of the data"""

data = pd.read_csv('{mention the path}/Data.csv')
data.head()

data.shape   #shape is the attribute of the dataframe and not a function/method, hence no parantheses ()

data.info()

data.isnull().sum()

"""# Id serves no purpose for any of our analysis, so dropping the id column"""

data = data.drop('id', axis = 1)
data
# axis = 1 to drop columns, axis = 0 to drop rows
# default is axis = 0

# need to provide a list if you want drop more than one column
# example:
#data = data.drop(['id','TenYearCHD'], axis = 1)

# 5 number summary for each of the numerical columns. This value is only for numerical data
data.describe().T   # .T will just give the transpose of the dataframe, easier to view

"""## Data is not perfect

# Null Value Treatment

## We need to use different kinds of imputation for numerical and categorical columns
"""

numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
numeric_columns

"""## You should be careful not to do imputation for Target variable(s)"""

#As it is 1/0 values(Binary)
numeric_columns.remove('TenYearCHD')

numeric_columns

data[numeric_columns].shape

categorical_columns = data.select_dtypes(include=object).columns.tolist()
categorical_columns

data[categorical_columns].shape

data['sex'].value_counts()

data['is_smoking'].value_counts()

data[categorical_columns].value_counts()

data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

#For categorical there is no mean since binary datatypes, therefore we use mode.
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode())

data.isnull().sum()

data.tail()

"""# Other Methods for treating NULL values

1. Dropping the Rows/Columns containing the null values
"""

#Sometimes dropping doesn't work the best as, if a column had a lot of null values then if we drop it only based on that column, we might lost most of the important data in the other columns.
# Drop all rows containing null values
#data = data.dropna()

# Drop columns containing null values
#data = data.dropna(axis = 1)

# by default the axis is 0

# you can choose to drop the rows containing nulls in only particular columns
# data = data.dropna(subset=['TenYearCHD','glucose'])

"""If there are large number of nulls in a particular column, it is better to drop that column rather than dropping the rows associated with the nulls in that column

What does Large mean?

It DEPENDS!! on the dataset

Also Calls for your judgement

2. KNN Imputation
"""

# Create the KNN imputer
#imputer = KNNImputer(n_neighbors=3)

#imputed_data = imputer.fit_transform(data)

#imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

"""# Drop duplicate rows if any"""

data.shape

data = data.drop_duplicates(keep='first')
data.shape

"""## Exploratory Data Analysis"""

for col in numeric_columns:
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    plt.title("----------   " + col + "   ----------")
    sns.boxplot(data=data, x=col, showmeans=True, color="violet")
    plt.subplot(2, 1, 2)
    sns.histplot(data=data, x=col)
    plt.axvline(data[col].mean(), color="red", linestyle="-")
    plt.axvline(data[col].median(), color="green", linestyle="--")
    print('\n')
    plt.show()

for col in categorical_columns:
    sns.countplot(data=data[categorical_columns], x=col)
    plt.show()
    print('\n')

"""## Bivariate Analyses"""

data[numeric_columns].corr()

# correlation heatmap
plt.figure(figsize=(15, 7))
sns.heatmap(data[numeric_columns].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()

"""## OUTLIER Treatment"""

data[data['totChol'] > 500]

rows_drop = data[data['totChol'] > 500].index

data[data['totChol'] > 500]

data = data.drop(rows_drop)

"""### You can try doing that for outliers in other columns

## We need to do Encoding for Categorical columns since most of the models work with only numerical data
"""

# Using get_dummies method
data = pd.get_dummies(
    data,
    columns= ['sex','is_smoking'],
    drop_first=True,
)
data.head()

"""### Other methods: OneHotEncoder
### it will produce a numpy array
### need to convert it back to a dataframe to continue

## Standardisation or Normalisation
"""

#defining X(independent) and Y(Dependent) Variables
X = data.drop(["TenYearCHD"], axis=1)
y = data["TenYearCHD"]

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns=X.columns)
X_norm

"""# fit_transform vs transform

## Log Transformation

### If there are 0's in your column to be log transformed, it will create some errors/problems (log 0 problem)
### you can add a small constant to the column such as 0.00001 to avoid this
"""

X_norm['glucose'] = X_norm['glucose'] + 0.00001

X_norm['glucose'] = np.log(X_norm['glucose'])
X_norm['glucose']

"""## Splitting the Data between Testing and Training"""

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=7, stratify = y)
print("Number of rows in train data =", X_train.shape[0])
print("Number of rows in test data =", X_test.shape[0])

"""## It is important to stratify the data based on target variable.
## X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify = y)
## In this command the parameter   'stratify = y'   will stratify the dataset based on the y variable. This is particularly important if the target variable is imbalanced

## The random_state = 7 is a seed so that when you run the program many times (number of experiments) division of data into train and test does not vary ie in each experiment same rows go into training data/testing data. This is important for consistency in different experiments and comparing the results
"""

y.value_counts()

print(f'Percentage of people with 10-year risk of coronary heart disease in this dataset: {511/data.shape[0]}')

"""This dataset is somewhat imabalanced"""

y_train.value_counts()

print(f'Percentage of people with 10-year risk of coronary heart disease in train set: {358/X_train.shape[0]}')

y_test.value_counts()

print(f'Percentage of people with 10-year risk of coronary heart disease in test set: {153/X_test.shape[0]}')

"""# BUILDING THE MODELS

## 1. Logistic Regression
"""

logistic = LogisticRegression(max_iter = 5000)

logistic.fit(X_train, y_train)

logistic.score(X_train,y_train)   #gives training accuracy

predicted_l = logistic.predict(X_train)

ConfusionMatrixDisplay(confusion_matrix(y_train, predicted_l)).plot()

"""### Accuracy is not always a good indicator --- WHY?"""

print(classification_report(y_train, predicted_l))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(logistic, X_train, y_train, cv=cv, scoring='f1',)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(logistic, X_train, y_train, cv=cv, scoring='roc_auc',)

"""# If you want to visualise the ROC Curve"""

# ROC curve
logistic_roc_auc_train = roc_auc_score(y_train, logistic.predict_proba(X_train)[:,1])
fpr, tpr, thresholds = roc_curve(y_train, logistic.predict_proba(X_train)[:,1])

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logistic_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic (Training)")
plt.legend(loc="lower right")
#plt.show()

"""## To change the thresholds"""

# Predict probabilities on the train and test sets
pred_prob = logistic.predict_proba(X_train)[:, 1]
pred_prob_test = logistic.predict_proba(X_test)[:, 1]  # since 1 is the positive class

# Calculate ROC AUC
roc_auc = roc_auc_score(y_train, pred_prob)

# Find the optimal threshold for ROC AUC
fpr, tpr, thresholds = roc_curve(y_train, pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Use the optimal threshold for predictions
y_pred_train_l = [1 if prob >= optimal_threshold else 0 for prob in pred_prob]
y_pred_test_l = [1 if prob >= optimal_threshold else 0 for prob in pred_prob_test]

# Print the ROC AUC and the optimal threshold
print("ROC AUC:", roc_auc)
print("Optimal Threshold based on ROC AUC:", optimal_threshold)

print(classification_report(y_train, y_pred_train_l))

print(classification_report(y_test, y_pred_test_l))

"""## Another way of building a Logistic Regression Model using statsmodels library which gives a nicer output"""

# Need to add a constant term to the indepdent variables
# This is specific to the statsmodel library to calculate the intercept when using Linear and Logistic Regression Models

#X = sm.add_constant(X)

#X.head()

# Redoing this since X has changed
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify = y)
#print("Number of rows in train data =", X_train.shape[0])
#print("Number of rows in test data =", X_test.shape[0])

#logreg = sm.Logit(y_train, X_train).fit()
#print(logreg.summary())

#predicted = logreg.predict(X_test)

"""## 2. K-Nearest Neighbour (KNN) Model"""

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(X_train, y_train)

knn.score(X_train,y_train)

predicted_knn = knn.predict(X_train)

ConfusionMatrixDisplay(confusion_matrix(y_train, predicted_knn)).plot()

print(classification_report(y_train, predicted_knn))

# To ignore the warnings generated in KNN related processings
warnings.filterwarnings("ignore")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1',)

"""## Model Tuning
### Using Grid Search Technique
"""

parameters = {"n_neighbors": np.arange(2, 10),}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

#To use stratified K-fold cross validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

knn_grid = GridSearchCV(KNeighborsClassifier(), parameters, scoring=scorer, cv=cv, n_jobs= -1).fit(X_train, y_train)

params = knn_grid.best_params_
params

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(knn_grid, X_train, y_train, cv=cv, scoring='f1',)

predicted_train_knn = knn_grid.predict(X_train)

print(classification_report(y_train, predicted_train_knn))

predicted_test_knn = knn_grid.predict(X_test)

print(classification_report(y_test, predicted_test_knn))

"""## 3. Decision Tree Classifier"""

tree = DecisionTreeClassifier(random_state=7)

tree.fit(X_train, y_train)

tree.score(X_train,y_train)

tree.get_depth()

plt.figure(figsize=(30, 30))
plot_tree(tree)
plt.show()

predicted = tree.predict(X_train)

print(classification_report(y_train, predicted))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(tree, X_train, y_train, cv=cv, scoring='f1',)

"""### Model Tuning"""

parameters = {"max_depth": np.arange(2, 25),}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

#To use stratified K-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

tree_grid = GridSearchCV(DecisionTreeClassifier(), parameters, scoring=scorer, cv=5, n_jobs= -1).fit(X_train, y_train)

params = tree_grid.best_params_
params

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(tree_grid, X_train, y_train, cv=cv, scoring='f1',)

predicted_train_tree = tree_grid.predict(X_train)

print(classification_report(y_train, predicted_train_tree))

predicted_test_tree = tree_grid.predict(X_test)

print(classification_report(y_test, predicted_test_tree))

"""## 4. Random Forest Classifier  (Ensemble Method)"""

Forest = RandomForestClassifier(random_state=7)

Forest.fit(X_train, y_train)

Forest.score(X_train,y_train)

predicted = Forest.predict(X_train)

ConfusionMatrixDisplay(confusion_matrix(y_train, predicted)).plot()

print(classification_report(y_train, predicted))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cross_val_score(tree, X_train, y_train, cv=cv, scoring='recall',)

parameters = {"max_depth": np.arange(2, 25),}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

#To use stratified K-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

forest_grid = GridSearchCV(RandomForestClassifier(random_state=7), parameters, scoring=scorer, cv=5, n_jobs= -1).fit(X_train, y_train)

params = forest_grid.best_params_
params

predicted_train_forest = forest_grid.predict(X_train)

print(classification_report(y_train, predicted_train_forest))

predicted_test_forest = forest_grid.predict(X_test)

print(classification_report(y_test, predicted_test_forest))

"""## We compare the different models based on chosen metrics and then choose the best model that is most capable of making good predictions"""

print(classification_report(y_test, predicted_test_knn))

print(classification_report(y_test, predicted_test_tree))

print(classification_report(y_test, predicted_test_forest))

# Also printing the earlier Logistic Regression scores
print(classification_report(y_test, y_pred_test_l))

"""## Try Optimising the thresholds for each of the models using predict_proba (like we did for Logistic Regression)

## If one is still not satisified with the scores :-

## We can try oversampling, SMOTE, ADASYN, other Ensembles etc.
"""
