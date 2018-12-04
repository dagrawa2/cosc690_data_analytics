import numpy as np
import pandas as pd
import utils

# set number of tweets to be randomly sampled out of 1.6 million; None means all 1.6 million tweets
n_tweets = 1000000

# load headers for the columns of the sentiment-140 data set
headers = utils.load_json("headers.json")

# make dictionary of months and their numerical label
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
months = {mon: i+1 for i, mon in enumerate(months)}

# load path where the sentiment-140 training data is located
with open("DATA_PATH.txt", "r") as file:
	DATA_PATH = file.read()

# load sentiment-140 training data and sample a subset of tweets
print("Loading data . . . ")
data = pd.read_csv(DATA_PATH, encoding="latin1", header=None, names=headers, usecols=["target", "date", "text"])
if n_tweets is not None:
	print("Sampling ", n_tweets, " tweets . . . ")
	data = data.sample(n=n_tweets, random_state=123)

# transform the sentiment score to -1 (negative), 0 (neutral), and 1 (positive)
print("Processing . . . ")
data["target"] /= 2
data["target"] -= 1
data["target"] = pd.to_numeric(data["target"], downcast="integer")

# parse date into month and day
data["month"] = pd.to_numeric(data["date"].apply(lambda date: months[date.split(" ")[1]]), downcast="integer")
data["day"] = pd.to_numeric(data["date"].apply(lambda date: int(date.split(" ")[2])), downcast="integer")

# add day index-- the nunber of days after April 6, 2009-- the first date in the data
days = {4: 0, 5: 30, 6: 61}
data["day_index"] = pd.to_numeric(data["month"].apply(lambda month: days[month]) + data["day"] - 6, downcast="integer")

data = data[["day_index", "month", "day", "text", "target"]]

# save processed data
print("Saving processed data . . . ")
data.to_csv("processed_data/data.csv", index=False)


# prepare to gather sentiment distribution (over negative, neutral, and positive) for each day
print("Getting daily sentiments . . . ")
data.drop(axis="columns", columns=["text"], inplace=True)

data = data.values
data[:,-1] += 1

# for each day index, get the sentiment distribution
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

# save daily sentiment distributions
print("Saving daily sentiments . . . ")
data_new.to_csv("processed_data/sentiments.csv", index=False)

print("Done!")
