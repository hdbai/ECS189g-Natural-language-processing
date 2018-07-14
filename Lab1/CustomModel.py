# coding=utf-8
import math, collections

d = 0.75

class CustomModel:
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.preceding = collections.defaultdict(lambda: 0)
    self.following = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:
    #         word = datum.word

    # P(w_i| w_i-1) = count(w_i-1, w_i) / count(w_i−1)
    for sentence in corpus.corpus:
      datums = sentence.data
      for i in xrange(0,len(datums)-1):
        self.total += 1
        w_i = datums[i].word
        self.unigramCounts[w_i] += 1
        w_i_1 = datums[i+1].word
        self.bigramCounts[(w_i,w_i_1)] += 1

  # add 1 smoothing for unigram
    self.unigramCounts['UNK'] = 0
    for word in self.unigramCounts:
      self.unigramCounts[word] += 1
    for word in self.bigramCounts:
        self.bigramCounts[word] += 1

  #calculate preceding and following word in bigram
    for word in self.unigramCounts:
      for key in self.bigramCounts:
          if key[0] == word:
              self.following[word] += 1
          if key[1] == word:
              self.preceding[word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in xrange(0, len(sentence)-1):
        w_i = sentence[i]
        w_i_1 = sentence[i+1]
        bigramCount = self.bigramCounts[((w_i, w_i_1))]
        if self.unigramCounts[w_i] > 0:
            unigramCount = self.unigramCounts[w_i]
        else:
            unigramCount = 1
        follow_word = self.following[w_i]
        preced_word = self.preceding[w_i_1]

        discounted_bigram = max(bigramCount - d,0.0000001) / float(unigramCount)

        lambda_w_i = (d / float(unigramCount)) * (follow_word)

        p_continuation = (preced_word ) / (float(len(self.bigramCounts)))

        p_kn = discounted_bigram + lambda_w_i * p_continuation
        score += math.log(p_kn)
    return score

