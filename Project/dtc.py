import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import utils

np.random.seed(123)
t_0 = time.time()

print("Loading data . . . ")
indices = np.load("processed_corpus/indices.npy")
X_train = np.load("results/doc_topic.npy")
Y_train = pd.read_csv("processed_data/data.csv", usecols=["target"]).values[indices]

X_train = X_train[Y_train.reshape((-1))!=0]
Y_train = Y_train[Y_train.reshape((-1))!=0]
print(". . . Data has", X_train.shape[0], "non-neutral observations")

print("Splitting into train and test sets . . . ")
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=456)

print("Training decision tree . . . ")
t_1 = time.time()
classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=0.001, min_samples_leaf=0.001)
classifier.fit(X_train, Y_train)
t_2 = time.time()
print(". . . Done; took", np.round((t_2-t_1)/60, 5), "min")

print("Predicting . . . ")
results = {}
pred = classifier.predict(X_train)
results.update({"train_acc": accuracy_score(Y_train, pred)})
results.update({"train_roc_auc": roc_auc_score(Y_train, pred)})

pred = classifier.predict(X_test)
results.update({"test_acc": accuracy_score(Y_test, pred)})
results.update({"test_roc_auc": roc_auc_score(Y_test, pred)})
t_3 = time.time()
print(". . . Done; took", np.round((t_3-t_2)/60, 5), "min")

print("Saving results . . . ")
utils.save_json(results, "results/dtc.json")

print("Done!")
print("Total time:", np.round((time.time()-t_0)/60, 5), "min")
