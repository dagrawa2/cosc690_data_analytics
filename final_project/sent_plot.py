import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import utils

# load topic names and sentiment breakdown by topic
topic_names = np.array(utils.load_json("results/topic_names.json"))
sent_dist = np.load("results/sent_dist.npy")

# sort topics by decreasing sentiment
indices = np.argsort(sent_dist)[::-1]
topic_names = topic_names[indices]
sent_dist = sent_dist[indices]

# plot bar graph of sentiments for the topics
print("Plotting . . . ")
plt.figure()
plt.bar(topic_names, sent_dist, color="black")
plt.xticks(rotation=45)
plt.title("Sentiment of each topic")
plt.xlabel("Topic")
plt.ylabel("Sentiment")
plt.savefig("plots/sent_dist.png", bbox_inches='tight')

print("Done!")
