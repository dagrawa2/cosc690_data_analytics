import numpy as np
import pandas as pd
import utils

np.random.seed(123)

# load indices corresponding to nonempty tweets in the preprocessed corpus
# then load unprocessed tweets corresponding to these indices
# load the topic names as well as the document-topic matrixx
indices = np.load("processed_corpus/indices.npy")
tweets = pd.read_csv("processed_data/data.csv", usecols=["text"]).values.reshape((-1))[indices]
topics = utils.load_json("results/topic_names.json")
doc_topic = np.load("results/doc_topic.npy")

# shuffle the tweets and the corresponding rows of the document-topic matrix
indices = np.arange(tweets.shape[0])
np.random.shuffle(indices)
tweets = tweets[indices]
doc_topic = doc_topic[indices]

# take random sample of 100 tweets (randomness comes from the above shuffle) and save tweet, predicted topic, and topic probability
with open("results/example_tweets.txt", "w") as fp:
	for i in range(100):
		fp.write(tweets[i]+"\n")
		j = np.argmax(doc_topic[i])
		fp.write(topics[j]+", "+str(np.round(doc_topic[i, j], 5))+"\n\n")

print("Done!")