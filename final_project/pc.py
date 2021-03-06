import nltk
import pandas as pd
import re

import findspark
findspark.init()
from pyspark import SparkContext

num_of_stop_words = 50	  # Number of most common words to remove, trying to eliminate stop words


sc = SparkContext('local', 'PySPARK LDA Example')

# load common stop words and append additional stop words chosen by hand
print("Loading stop words . . . ")
stop_words = nltk.corpus.stopwords.words('english')
with open("more_stop_words.txt", "r") as fp:
	stop_words = stop_words + [w.strip("\n") for w in fp.readlines()]

# load corpus (produced by pd.py) into spark RDD
print("Loading corpus . . . ")
data = pd.read_csv("processed_data/data.csv", usecols=["text"])
data = list(data.values.reshape((-1)))
data = sc.parallelize(data)

# for each tweet in corpus: remove URLs, remove all non-alphabetic characters, make all letters lower case, tokenize, remove words 3 characters or shorter, and remove stop words
print("Tokenizing . . . ")
tokens = data												   \
	.map( lambda document: document.strip())			\
	.map(lambda document: re.sub(r"(?:\@|http?\://)\S+", "", document))  \
	.map(lambda document: re.sub("[^A-Za-z\s]+", " ", document))  \
	.map(lambda document: document.lower())  \
	.map( lambda document: re.split("[\s;,#]", document))	   \
	.map( lambda document: [x for x in document if len(x) > 3] )  \
	.map(lambda document: [x for x in document if x not in stop_words])

# get RDD of words and their frequencies sorted in descending order by frequency
print("Getting term counts . . . ")
termCounts = tokens							 \
	.flatMap(lambda document: document)		 \
	.map(lambda word: (word, 1))				\
	.reduceByKey( lambda x,y: x + y)			\
	.map(lambda tuple: (tuple[1], tuple[0]))	\
	.sortByKey(False)

# Identify a threshold to remove the top words, in an effort to remove stop words
threshold_value = termCounts.take(num_of_stop_words)[num_of_stop_words - 1][0]

# collect tokens RDD into list
print("Collecting tokens . . . ")
tokens = tokens.collect()

# collect vocabulary list after removing most common words and words appearing only once
print("Collecting vocabulary . . . ")
vocabulary = termCounts						 \
	.filter(lambda x : x[0] < threshold_value)  \
	.filter(lambda x : x[0] > 1)  \
	.map(lambda x: x[1])						\
	.collect()

# remove tokens not in vocabulary
print("Filtering out out-of-vocabulary tokens . . . ")
#tokens = [" ".join([w for w in document if w in vocabulary]) for document in tokens]
tokens = sc.parallelize(tokens)  \
	.map(lambda document: [w for w in document if w in vocabulary])  \
	.map(lambda document: " ".join(document))  \
	.collect()

# save preprocessed tweets
print("Saving tokens . . . ")
with open("processed_corpus/tokens.txt", "w") as fp:
	for document in tokens:
		fp.write(document+"\n")

# save vocabulary
print("Saving vocabulary . . . ")
with open("processed_corpus/vocabulary.txt", "w") as fp:
	for word in vocabulary:
		fp.write(word+"\n")

# save indices of nonempty tweets (some are empty due to preprocessing, but we keep these as empty lines to maintain line-wise correspondence with the unprocessed tweets)
print("Saving indices . . . ")
tokens = np.array(tokens)
indices = np.arange(tokens.shape[0])
indices = indices[tokens!=""]
np.save("Processed_corpus/indices.npy", indices)

print("Done!")