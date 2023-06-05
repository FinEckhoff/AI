import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Define the column names
column_names = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status',
                'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss',
                'Hours-per-week', 'Native-country', 'Income']

# Read the data from the CSV file
data = pd.read_csv('./adult/adult.data', names=column_names)

# Replace '?' with NaN (missing value indicator in pandas)
data = data.replace('?', np.nan)

# For simplicity, fill missing values with the mode value of each column
for column in data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

#print(data.head())

# Drop the 'Fnlwgt' column
data = data.drop(columns='Fnlwgt')

# Convert the 'Capital-gain' and 'Capital-loss' columns to a binary 'Has capital gain or loss' column
data['Has capital gain or loss'] = data.apply(lambda row: 1 if row['Capital-gain'] > 0 or row['Capital-loss'] > 0 else 0, axis=1)
data['is married'] = data['Marital-status'].apply(lambda x: 1 if x == ' Married-civ-spouse' else 1 if x == ' Married-spouse-absent' else 1 if x == ' Married-AF-spouse' else 0)
data = data.drop(columns='Native-country')

data['Work Hours'] = data['Hours-per-week'].apply(lambda x: 0 if x <= 20 else 0 if x <= 40 else 2)
data = data.drop(columns= 'Hours-per-week')
data['Employment type']= data['Workclass'].apply(lambda x: 2 if x == ' Self-emp-inc' else 2 if x == ' Self-emp-not-inc' else 0 if x == ' Never-worked' else 0 if x == ' Without-pay' else 1)
# Drop the 'Capital-gain' and 'Capital-loss' columns
data = data.drop(columns=['Capital-gain', 'Capital-loss', 'Workclass', 'Education', 'Marital-status', 'Relationship'])


#print(data['Sex'].unique())
data['Sex']= data['Sex'].apply(lambda x: 0 if x == ' Female' else 1)

#data = data.drop(columns=[ 'Sex'])



data['Income'] = data['Income'].apply(lambda x: 0 if x == ' <=50K' else 1)
# Now the 'Capital-gain' and 'Capital-loss' columns are converted to a binary 'Has capital gain or loss' column
numeric_columns = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()


data['Occupation'] = data['Occupation'].astype('category')
ohe = OneHotEncoder(sparse_output=False)
ohe_race = OneHotEncoder(sparse_output=False)
transformedRACE = ohe_race.fit_transform(data[['Race']])
transformedOCC = ohe.fit_transform(data[['Occupation']])
#print(ohe.categories_)
#print(transformed[0].size)


for index, cat in enumerate(ohe.categories_):
    data[cat] = transformedOCC[index]

for index, cat in enumerate(ohe_race.categories_):
    data[cat] = transformedRACE[index]

data = data.drop(columns=['Occupation'])
data = data.drop(columns=['Race'])


#print(data['Occupation'])
# Print the correlation matrix
"""
print(correlation_matrix)

# Draw a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Columns')
plt.show()
"""
dataset = data.values
X = dataset
Income = data['Income']

index = 0
for i, k in enumerate(data.keys()):
    if k == 'Income':
        index = i


X = np.delete(X, index, axis=1)
#print(data.keys())
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
Y = Income.values


knn = KNeighborsClassifier(n_neighbors=6)

#print(data['Income'].unique())
#print(X_scale)
#print(Y)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)



