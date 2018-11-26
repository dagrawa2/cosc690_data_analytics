import os
import time
import lda
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import utils

# set LDA hyperparameters
n_topics = 20
n_iter = 500
alpha = 0.01

# set number of words to print for each topic
n_top_words = 10

# start timer
t_0 = time.time()

# load preprocessed corpus of tweetss
with open("processed_corpus/tokens.txt", "r") as fp:
	tweets = [line.strip("\n") for line in fp.readlines()]

# discard empty tweets (that resulted from preprocessing) and record number of nonempty tweets
tweets = [t for t in tweets if len(t) > 0]
n_tweets = len(tweets)

# ignore terms that have a document frequency strictly lower than 5
cvectorizer = CountVectorizer(min_df=5)
cvz = cvectorizer.fit_transform(tweets)

# train LDA model
print("Training LDA model . . . ")
t_1 = time.time()
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter, alpha=alpha, random_state=123)
X_topics = lda_model.fit_transform(cvz)
print(". . . Done; took ", np.round((time.time()-t_1)/60, 5), " min")

# Get document-topic and topic-word arrays
print("Saving doc-topic and topic-word arrays . . . ")
np.save("results/doc_topic.npy", X_topics)
np.save("results/topic_word.npy", lda_model.topic_word_)

# Get topic summaries; list of top ten words for each topic
print("Getting topic summaries . . . ")
topic_summaries = []
topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	topic_summaries.append(' '.join(topic_words))

print("Saving topic summaries . . . ")
with open("results/topic_summaries.txt", "w") as fp:
	for i, words in enumerate(topic_summaries):
		fp.write("Topic "+str(i+1)+": "+words+"\n")

# Define and save topic_names (ith topic is named "Topic i" by default)
if not os.path.isfile("results/topic_names.json"):
	topic_names = ["Topic "+str(i+1) for i in range(n_topics)]
	utils.save_json(topic_names, "results/topic_names.json")

print("Done!")
print("Total time: ", np.round((time.time()-t_0)/60, 5), " min")
