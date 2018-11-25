import numpy as np
import pandas as pd

indices = np.load("processed_corpus/indices.npy")

topics = np.load("results/doc_topic.npy")

topic_dist = np.mean(topics, axis=0)
np.save("results/topic_dist.npy", topic_dist)

sent = pd.read_csv("processed_data/data.csv", usecols=["target"]).values
sent = sent[indices]

#sent_dist = np.mean(topics*sent, axis=0)
sent_dist = np.sum(topics*sent, axis=0)/np.sum(topics, axis=0)
np.save("results/sent_dist.npy", sent_dist)

print("Done!")
