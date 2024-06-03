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
import csv

NUMBER_OF_COLUMNS = 9

file_path_csv = './indian_diabetes.csv'
file_path = './indian_diabetes.xlsx'

df = pd.read_excel(file_path)
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

model.fit(X, Y, epochs=100)

_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)

df_train.head()

X_train = df_train.loc[:, df.columns != 'DiabetesPresence']
Y_train = df_train.loc[:, 'DiabetesPresence']
X_test = df_test.loc[:, df.columns != 'DiabetesPresence']
Y_test = df_test.loc[:, 'DiabetesPresence']

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=50, batch_size=10)

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




