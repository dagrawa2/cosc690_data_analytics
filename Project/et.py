import numpy as np
import pandas as pd
import utils

np.random.seed(123)

indices = np.load("processed_corpus/indices.npy")
tweets = pd.read_csv("processed_data/data.csv", usecols=["text"]).values.reshape((-1))[indices]
topics = utils.load_json("results/topic_names.json")
doc_topic = np.load("results/doc_topic.npy")

indices = np.arange(tweets.shape[0])
np.random.shuffle(indices)
tweets = tweets[indices]
doc_topic = doc_topic[indices]

with open("results/example_tweets.txt", "w") as fp:
	for i in range(100):
		fp.write(tweets[i]+"\n")
		j = np.argmax(doc_topic[i])
		fp.write(topics[j]+", "+str(np.round(doc_topic[i, j], 5))+"\n\n")

print("Done!")