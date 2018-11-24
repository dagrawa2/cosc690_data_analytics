import random
import time
import numpy as np
from sklearn.manifold import TSNE
import utils

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

n_points = 5000
threshold = 0


t_0 = time.time()

X = np.load("results/doc_topic.npy")
n_topics = X.shape[1]

idx = np.amax(X, axis=1) > threshold
X = X[idx]
X = X[:n_points]
topics = np.argmax(X, axis=1)

print("Performing TSNE . . . ")
t_1 = time.time()
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
X = tsne_model.fit_transform(X)
print(". . . done; took ", np.round((time.time()-t_1)/60, 5), " min")

# generate random hex color
colormap = []
for i in range(n_topics):
	r = lambda: random.randint(0, 255)
	colormap += ('#%02X%02X%02X' % (r(), r(), r())),
colormap = np.array(colormap)
np.save("results/colormap.npy", colormap)

topic_names = utils.load_json("results/topic_names.json")

print("Plotting . . . ")
plt.figure()
for i in range(n_topics):
	pnts = X[topics==i]
	plt.scatter(pnts[:,0], pnts[:,1], color=colormap[i], label=topic_names[i])
plt.legend(title="Topic")
plt.title("TSNE projection of tweets based on topics")
plt.savefig("plots/tsne.png", bbox_inches='tight')

print("Done!")
print("Total time: ", np.round((time.time()-t_0)/60, 5), " min")
