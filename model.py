import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import os

from keras import Input, Model
from keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.metrics import TopKCategoricalAccuracy 



# df = pd.read_csv("train_data/2004-05_team_stats.csv")
# df = df.drop(columns=["TEAM_ID", "TEAM_NAME"])
# df.fillna(0, inplace=True)
# data = df.to_numpy()



def load_data(folder_path):
    data = []

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    files.sort(key=lambda name: int(name[:4]))  # Sort chronologically

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        df = pd.read_csv(full_path)
        df = df.drop(columns=["TEAM_ID", "TEAM_NAME"])
        df.fillna(0, inplace=True)
        arr = df.to_numpy(dtype='float32')
        data.append(arr)

    return np.array(data, dtype='float32')


def make_y_train(champion_indices):
    labels = []
    for idx in champion_indices:
        arr = np.zeros(30, dtype='float32')
        arr[idx] = 1.0
        labels.append(arr)

    return np.array(labels, dtype='float32')


folder_path = "train_data"
train_champion_indices = [6, 15, 15, 26, 9, 5, 9, 9, 27, 13, 16] # 
# train_playoff_indices_combined = [
#     [1, 3, 5, 6, 7, 9, 10, 13, 14, 17, 21, 22, 24, 25, 26, 29],  # 2004-05 
#     [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 22, 24, 25, 29],  # 2005-06 
#     [3, 4, 5, 6, 7, 8, 9, 12, 14, 17, 20, 22, 25, 27, 28, 29],  # 2006-07 
#     [0, 1, 4, 5, 6, 7, 9, 12, 18, 20, 21, 22, 25, 27, 28, 29],  # 2007-08 
#     [0, 1, 3, 4, 5, 6, 7, 9, 12, 14, 18, 21, 22, 24, 26, 28],  # 2008-09 
#     [0, 1, 2, 3, 4, 5, 6, 12, 14, 15, 20, 21, 23, 24, 26, 28],  # 2009-10 
#     [0, 1, 3, 5, 6, 10, 12, 13, 14, 18, 19, 20, 21, 22, 24, 26],  # 2010-11 
#     [0, 1, 3, 5, 6, 10, 11, 12, 13, 14, 19, 20, 21, 22, 26, 28],  # 2011-12 
#     [0, 1, 2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 26],  # 2012-13 
#     [0, 1, 3, 4, 6, 9, 10, 11, 12, 14, 15, 20, 24, 26, 27, 29],  # 2013-14 
#     [0, 1, 2, 4, 5, 6, 9, 10, 12, 14, 16, 18, 24, 26, 27, 29],  # 2014-15 
#     [0, 1, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 20, 24, 26, 27],  # 2015-16 
#     [0, 1, 4, 5, 9, 10, 11, 12, 14, 16, 20, 24, 26, 27, 28, 29],  # 2016-17
#     [1, 5, 9, 10, 11, 15, 16, 17, 18, 20, 22, 24, 26, 27, 28, 29],  # 2017-18 
#     [1, 2, 7, 8, 9, 10, 11, 12, 16, 20, 21, 22, 24, 26, 27, 28],  # 2018-19 
#     [1, 2, 6, 7, 10, 11, 12, 13, 15, 16, 20, 21, 22, 24, 27, 28],  # 2019-20
#     [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 19, 22, 23, 24, 28, 29]  # 2020-21 
# ]
        

# 

X_train = load_data(folder_path)
Y_train = make_y_train(train_champion_indices)





num_features = X_train.shape[2]

# # Build the model
# inputs = Input(shape=(30, num_features))  # one szn of 30 teams
# x = Dense(30, activation='tanh')(inputs)
# x = Dense(16, activation='tanh')(x)
# x = Dense(8, activation='relu')(x)
# x = Dense (4, activation='relu')(x)
# x = Dense(1)(x)        # Output one score per team
# x = Flatten()(x)       # Flatten to (30,)
# outputs = Activation('softmax')(x)  # Probabilities across teams

# model = Model(inputs=inputs, outputs=outputs)

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',  # Categorical crossentropy for multi-class classification
#     metrics=[
#         'accuracy',
#         TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
#     ]
# )

# sample_weights = np.exp(np.linspace(1, 3, len(Y_train)))  # gives a big boost to recent
# sample_weights = np.exp(np.linspace(0, 2, len(Y_train)))  # weights from 1 to ~7.4



# history = model.fit(X_train, Y_train, epochs=75, batch_size=1)

# # Calculate average accuracy across all epochs
# avg_accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])
# avg_top3 = sum(history.history['top_3_accuracy']) / len(history.history['top_3_accuracy'])

# print(f"Average training accuracy: {avg_accuracy:.4f}")
# print(f"Average top-3 accuracy: {avg_top3:.4f}")
# model.save("nba_model.h5")

test_champion_indices = [9, 7, 1]  
nba_model = keras.models.load_model("nba_model.h5")


test_path = "test_data"
X_test = load_data(test_path)
Y_test = make_y_train(test_champion_indices)

results = nba_model.evaluate(X_test, Y_test, verbose=1)

TEAM_NAMES = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
print(f"Top-3 Accuracy: {results[2]:.4f}")

probs = nba_model.predict(X_test, verbose=0)

# For each test season, create a DataFrame with team names and their predicted probs
for i, season_probs in enumerate(probs):
    df = pd.DataFrame({
        "Team": TEAM_NAMES,
        "Probability": np.round(season_probs, 4)
    }).sort_values(by="Probability", ascending=False).reset_index(drop=True)
    print(df)
    df.to_csv(f"predictions/test_season_{i + 1}.csv", index=False)


