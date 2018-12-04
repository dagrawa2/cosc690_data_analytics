import numpy as np
import utils

# load topic names
topic_names = np.array(utils.load_json("results/topic_names.json"))

# load topic summaries
with open("results/topic_summaries.txt", "r") as fp:
	topic_summaries = fp.readlines()

topic_summaries = [line.split(":")[-1] for line in topic_summaries]

# save topic summaries but where "Topic i" is replaced with the name of the ith topic
with open("results/topic_summaries_with_names.txt", "w") as fp:
	for name, summary in zip(topic_names, topic_summaries):
		fp.write(name+":"+summary)

print("Done!")