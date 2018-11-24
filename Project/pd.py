import numpy as np
import pandas as pd
import utils

n_tweets = 1000000


headers = utils.load_json("headers.json")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
months = {mon: i+1 for i, mon in enumerate(months)}

with open("DATA_PATH.txt", "r") as file:
	DATA_PATH = file.read()

print("Loading data . . . ")
data = pd.read_csv(DATA_PATH, encoding="latin1", header=None, names=headers, usecols=["target", "date", "text"])
if n_tweets is not None:
	print("Sampling ", n_tweets, " tweets . . . ")
	data = data.sample(n=n_tweets, random_state=123)

print("Processing . . . ")
data["target"] /= 2
data["target"] -= 1
data["target"] = pd.to_numeric(data["target"], downcast="integer")

data["month"] = pd.to_numeric(data["date"].apply(lambda date: months[date.split(" ")[1]]), downcast="integer")
data["day"] = pd.to_numeric(data["date"].apply(lambda date: int(date.split(" ")[2])), downcast="integer")

days = {4: 0, 5: 30, 6: 61}
data["day_index"] = pd.to_numeric(data["month"].apply(lambda month: days[month]) + data["day"] - 6, downcast="integer")

data = data[["day_index", "month", "day", "text", "target"]]

print("Saving processed data . . . ")
data.to_csv("processed_data/data.csv", index=False)


print("Getting daily sentiments . . . ")
data.drop(axis="columns", columns=["text"], inplace=True)

data = data.values
data[:,-1] += 1

data_new = np.unique(data[:,:3], axis=0)
sents = []
for i in data_new[:,0]:
	s = (data[:,-1])[data[:,0]==i]
	A = np.zeros((len(s), 3))
	A[np.arange(s.shape[0]),s] = 1
	sents.append(A.sum(axis=0))

sents = np.stack(sents, axis=0)
data_new = np.concatenate((data_new, sents), axis=1)
data_new = pd.DataFrame(data_new, columns=["day_index", "month", "day", "negative", "neutral", "positive"], dtype=int)

print("Saving daily sentiments . . . ")
data_new.to_csv("processed_data/sentiments.csv", index=False)

print("Done!")
