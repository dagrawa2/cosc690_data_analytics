import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import utils

topic_names = np.array(utils.load_json("results/topic_names.json"))
sent_dist = np.load("results/sent_dist.npy")

indices = np.argsort(sent_dist)[::-1]
topic_names = topic_names[indices]
sent_dist = sent_dist[indices]

print("Plotting . . . ")
plt.figure()
plt.bar(topic_names, sent_dist, color="black")
plt.title("Sentiment of each topic")
plt.xlabel("Topic")
plt.ylabel("Sentiment")
plt.savefig("plots/sent_dist.png", bbox_inches='tight')

print("Done!")
