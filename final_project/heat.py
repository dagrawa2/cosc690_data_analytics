import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

# class to help normalize colors in heatmap
class MidpointNormalize(Normalize):

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


# function to plot heatmap
def plotHeatMap(data, x_title='X Axis', y_title='Y Axis', title='', x_ticks=[], y_ticks=[], filename="heatmap.png"):
	vmin, midpoint = np.min(data), np.mean(data)
	plt.imshow(data[::-1], interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=vmin, midpoint=midpoint))
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.colorbar()
	plt.xticks(np.arange(len(x_ticks)), x_ticks)
	plt.yticks(np.arange(len(y_ticks)), y_ticks[::-1])
	plt.title(title)
	plt.savefig(filename, bbox_inches='tight')

# produce heatmap of log-likelihoods produced from model_ho.py
data = pd.read_csv("results/lls.csv", usecols=["ll"]).values.reshape((4, 3)).T
xticks = [5, 10, 15, 20]
yticks = [0.01, 0.1, 1.0]

plotHeatMap(data, x_title="Number of topics", y_title="alpha", title="Log-likelihood for various runs of LDA", x_ticks=xticks, y_ticks=yticks, filename="plots/heat.png")

print("Done!")