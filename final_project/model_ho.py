import time
import lda
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import utils

# Set LDA hyperparameters to be tested; all combinations of n_topics and alpha will be run
n_topics = [5, 10, 15, 20]
n_iter = 500
alpha = [0.01, 0.1, 1.0]

# Set number of words to print for each topic
n_top_words = 10

# start timer
t_0 = time.time()

# load preprocessed corpus
with open("processed_corpus/tokens.txt", "r") as fp:
	tweets = [line.strip("\n") for line in fp.readlines()]

# discard empty tweets that resulted from preprocessing and record number of nonempty tweets
tweets = [t for t in tweets if len(t) > 0]
n_tweets = len(tweets)

# ignore terms that have a document frequency strictly lower than 5
cvectorizer = CountVectorizer(min_df=5)
cvz = cvectorizer.fit_transform(tweets)

# Initialize list to record log-likelihoods
lls = []

# train LDA model for each combination of hyperparameters and record log-likelihood
for n_topics_value in n_topics:
	for alpha_value in alpha:
		print("Training LDA model (n_topics="+str(n_topics_value)+", alpha="+str(alpha_value)+") . . . ")
		t_1 = time.time()
		lda_model = lda.LDA(n_topics=n_topics_value, n_iter=n_iter, alpha=alpha_value, random_state=123)
		lda_model.fit(cvz)
		lls.append([n_topics_value, alpha_value, lda_model.loglikelihood()])
		print(". . . Done; took ", np.round((time.time()-t_1)/60, 5), " min")

		# still for each hyperparameter combination, get topic summaries-- ttop ten words for each topic
		print("Getting topic summaries . . . ")
		topic_summaries = []
		topic_word = lda_model.topic_word_  # get the topic words
		vocab = cvectorizer.get_feature_names()
		for i, topic_dist in enumerate(topic_word):
			topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
			topic_summaries.append(' '.join(topic_words))

		print("Saving topic summaries . . . ")
		with open("results/topic_summaries_"+str(n_topics_value)+"_"+str(alpha_value)+".txt", "w") as fp:
			for i, words in enumerate(topic_summaries):
				fp.write("Topic "+str(i+1)+": "+words+"\n")

print("Saving log-likelihoods . . . ")
pd.DataFrame(np.array(lls), columns=["n_topics", "alpha", "ll"]).to_csv("results/lls.csv", index=False)

print("Done!")
print("Total time: ", np.round((time.time()-t_0)/60, 5), " min")
