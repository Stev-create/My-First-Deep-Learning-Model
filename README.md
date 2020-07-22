# My-First-Deep-Learning-Model

In this notebook, I try to build my first model Deep Learning Model using Tensorflow and Keras.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
```


```python
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.shape
```




    (1470, 35)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
df.isna().sum()
```




    Age                         0
    Attrition                   0
    BusinessTravel              0
    DailyRate                   0
    Department                  0
    DistanceFromHome            0
    Education                   0
    EducationField              0
    EmployeeCount               0
    EmployeeNumber              0
    EnvironmentSatisfaction     0
    Gender                      0
    HourlyRate                  0
    JobInvolvement              0
    JobLevel                    0
    JobRole                     0
    JobSatisfaction             0
    MaritalStatus               0
    MonthlyIncome               0
    MonthlyRate                 0
    NumCompaniesWorked          0
    Over18                      0
    OverTime                    0
    PercentSalaryHike           0
    PerformanceRating           0
    RelationshipSatisfaction    0
    StandardHours               0
    StockOptionLevel            0
    TotalWorkingYears           0
    TrainingTimesLastYear       0
    WorkLifeBalance             0
    YearsAtCompany              0
    YearsInCurrentRole          0
    YearsSinceLastPromotion     0
    YearsWithCurrManager        0
    dtype: int64



No missing Values. And like I said, the purpose of this notebook it's to build a deep learning model. So in this time, I won't exploring the data.  

# Dataset Splitting


```python
X = pd.get_dummies(df.drop(columns = 'Attrition'))
y = df['Attrition'].map({'Yes' : 1, 'No' : 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101, stratify = y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((1176, 55), (294, 55), (1176,), (294,))



# Preprocessing


```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

# Build the model


```python
classifier = Sequential()
classifier.add(Dense(64, activation = 'relu', input_dim = 55))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

classifier.summary()
```

    Model: "sequential_17"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_30 (Dense)             (None, 64)                3584      
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    dense_31 (Dense)             (None, 1)                 65        
    =================================================================
    Total params: 3,649
    Trainable params: 3,649
    Non-trainable params: 0
    _________________________________________________________________



```python
def make_model():
    classifier = Sequential()
    classifier.add(Dense(64, activation = 'relu', input_dim = 55))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
    return classifier

classifier = KerasClassifier(build_fn = make_model, batch_size=10, nb_epoch=1)
```


```python
acc = cross_val_score(estimator = classifier,X =X_train,y = y_train,cv = 10,n_jobs = -1)
```


```python
acc.mean()
```




    0.8495581686496735



# Running Predictions


```python
classifier.fit(X_train, y_train, epochs = 5)
```

    Epoch 1/5
    118/118 [==============================] - 1s 11ms/step - loss: 0.5126 - accuracy: 0.7662
    Epoch 2/5
    118/118 [==============================] - 1s 10ms/step - loss: 0.3574 - accuracy: 0.8656
    Epoch 3/5
    118/118 [==============================] - 1s 11ms/step - loss: 0.3186 - accuracy: 0.8801
    Epoch 4/5
    118/118 [==============================] - 1s 10ms/step - loss: 0.3015 - accuracy: 0.8861
    Epoch 5/5
    118/118 [==============================] - 1s 11ms/step - loss: 0.2864 - accuracy: 0.8946





    <tensorflow.python.keras.callbacks.History at 0x7f6f087014d0>




```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
confusion_matrix(y_test, y_pred)
```




    array([[167,  80],
           [ 35,  12]])



# Thank you!
