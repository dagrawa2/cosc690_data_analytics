import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import utils

# load daily sentiment breakdown by topic and day index
sent_evol = np.load("results/sent_evol.npy")
day = sent_evol[:,0]
sent_evol = sent_evol[:,1:]
n_topics = sent_evol.shape[1]

# load same colormap that was used in the TSNE plot; load topic names
colormap = np.load("results/colormap.npy")
topic_names = utils.load_json("results/topic_names.json")

# plot curves of sentiment evolution for each topic
print("Plotting . . . ")
plt.figure()
for i in range(n_topics):
	plt.plot(day, sent_evol[:,i], color=colormap[i], label=topic_names[i])
plt.legend(title="Topic")
plt.title("Sentiment of each topic over time")
plt.xlabel("Day (since April 6, 2009)")
plt.ylabel("Sentiment")
plt.savefig("plots/sent_evol.png", bbox_inches='tight')

print("Done!")
print("Total time: ", np.round((time.time()-t_0)/60, 5), " min")
