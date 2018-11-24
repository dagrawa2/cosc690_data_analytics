import numpy as np
import pandas as pd

indices = np.load("processed_corpus/indices.npy")

topics = np.load("results/doc_topic.npy")

day = pd.read_csv("processed_data/data.csv", usecols=["day_index"]).values.reshape((-1))
day = day[indices]
sent = pd.read_csv("processed_data/data.csv", usecols=["target"]).values
sent = sent[indices]

sent_evol = []
for i in np.unique(day):
	sent_dist = np.mean(topics[day==i]*sent[day==i], axis=0) # /np.sum(topics[day==i], axis=0)
	sent_evol.append( np.concatenate((np.array([i]), sent_dist), axis=0) )

sent_evol = np.stack(sent_evol, axis=0)
np.save("results/sent_evol.npy", sent_evol)

print("Done!")
