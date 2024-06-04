# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:54:24 2023

@author: bebo
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# faire afficher le nuages de données
# générer un partionnement du nuage

df =  pd.read_excel('indian diabetes.xlsx') 
print(df)

# Fonction pour tracer la droite de régression
def plot_regression_line(x, y):
    # Conversion en tableau numpy
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    # Créer le modèle de régression linéaire
    reg = LinearRegression().fit(x, y)
    # Prédire les valeurs y à partir des valeurs x
    y_pred = reg.predict(x)
    # Tracer la ligne de régression
    plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')

# Nuage de points Age vs Glucose avec droite de régression
x = df.Age
y = df.Glucose
fig = plt.figure(figsize=(20, 10))
plt.plot(x, y, "ob")  # point bleu
plot_regression_line(x, y)
plt.xlabel('Age')
plt.ylabel('Taux Glucose')
plt.legend()
plt.show()

# Nuage de points Age vs BloodPressure avec droite de régression
x = df.Age
z = df.BloodPressure
fig = plt.figure(figsize=(20, 10))
plt.plot(x, z, "ob")  # point bleu
plot_regression_line(x, z)
plt.xlabel('Age')
plt.ylabel('Pression sanguine')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:54:24 2023

@author: bebo
"""

from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.linear_model import LinearRegression
import numpy as np

# faire afficher le nuages de données
# générer un partitionnement du nuage

df = pd.read_excel('indian diabetes.xlsx')
print(df)

# Fonction pour tracer la droite de régression
def plot_regression_line(x, y):
    # Conversion en tableau numpy
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    # Créer le modèle de régression linéaire
    reg = LinearRegression().fit(x, y)
    # Prédire les valeurs y à partir des valeurs x
    y_pred = reg.predict(x)
    # Tracer la ligne de régression
    plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')

# Nuage de points Age vs Glucose avec droite de régression
x = df.Age
y = df.Glucose
fig = plt.figure(figsize=(20, 10))
plt.plot(x, y, "ob")  # point bleu
plot_regression_line(x, y)
plt.xlabel('Age')
plt.ylabel('Taux Glucose')
plt.legend()
plt.show()

# Nuage de points Age vs BloodPressure avec droite de régression
x = df.Age
z = df.BloodPressure
fig = plt.figure(figsize=(20, 10))
plt.plot(x, z, "ob")  # point bleu
plot_regression_line(x, z)
plt.xlabel('Age')
plt.ylabel('Pression sanguine')
plt.legend()
plt.show()

# Nuage de points Glucose vs BloodPressure avec droite de régression
y = df.Glucose
z = df.BloodPressure
fig = plt.figure(figsize=(20, 10))
plt.plot(y, z, "ob")  # point bleu
plot_regression_line(y, z)
plt.xlabel('Taux Glucose')
plt.ylabel('Pression sanguine')
plt.legend()
plt.show()


df.head()

X = df.loc[:, df.columns != 'DiabetesPresence']
Y = df.loc[:, 'DiabetesPresence']

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu')) # essai couche en plus
model.add(Dense(8, activation='relu')) # essai couche en plus
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100) # on en rajoute de 50 à 100 et plus si affinité

_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)

df_train.head()

X_train = df_train.loc[:, df.columns != 'DiabetesPresence']
Y_train = df_train.loc[:, 'DiabetesPresence']
X_test = df_test.loc[:, df.columns != 'DiabetesPresence']
Y_test = df_test.loc[:, 'DiabetesPresence']

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
print("Plot_model\n")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=10)

plt.plot(history.history['accuracy'], color='#066b8b')
plt.plot(history.history['val_accuracy'], color='#b39200')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predictions = model.predict(X_test)

predictions[0]
predictions = (model.predict(X_test) > 0.5).astype(int)
for i in range(5):
	print('%s => Prédit : %d,  Attendu : %d' % (X_test.iloc[i].tolist(), predictions[i], Y_test.iloc[i]))
    
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Ajouter des couches






