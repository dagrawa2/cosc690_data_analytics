import time
import lda
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

n_topics = 20
n_iter = 500
alpha = 0.01

n_top_words = 10


t_0 = time.time()

with open("processed_corpus/tokens.txt", "r") as fp:
	tweets = [line.strip("\n") for line in fp.readlines()]

tweets = [t for t in tweets if len(t) > 0]
n_tweets = len(tweets)

# ignore terms that have a document frequency strictly lower than 5
cvectorizer = CountVectorizer(min_df=5)
cvz = cvectorizer.fit_transform(tweets)

print("Training LDA model . . . ")
t_1 = time.time()
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter, alpha=alpha, random_state=123)
X_topics = lda_model.fit_transform(cvz)
print(". . . Done; took ", np.round((time.time()-t_1)/60, 5), " min")

print("Saving doc-topic and topic-word arrays . . . ")
np.save("results/doc_topic.npy", X_topics)
np.save("results/topic_word.npy", lda_model.topic_word_)

print("Getting topic summaries . . . ")
topic_summaries = []
topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	topic_summaries.append(' '.join(topic_words))

print("Saving topic summaries . . . ")
with open("results/topics.txt", "w") as fp:
	for i, words in enumerate(topic_summaries):
		fp.write("Topic "+str(i+1)+": "+words+"\n")

print("Done!")
print("Total time: ", np.round((time_time()-t_0)/60, 5), " min")
