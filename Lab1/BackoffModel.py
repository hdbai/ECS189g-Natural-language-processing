# coding=utf-8
import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
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
      for i in xrange(0,len(datums)):
        self.total += 1
        w_i = datums[i].word
        self.unigramCounts[w_i] += 1
      for i in xrange(0,len(datums) - 1):
        self.bigramCounts[(datums[i].word,datums[i+1].word)] += 1
  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in xrange(0, len(sentence) - 1):
        w1 = sentence[i]
        w2 = sentence[i+1]
        bigramCount = self.bigramCounts[((w1,w2))] | 0
        unigramCount = self.unigramCounts[w1]
        # PAdd−1(wi | wi−1) = c(wi−1,wi)+1 / c(wi−1)+V , V is number of times we add one which is len(self.bigramCounts))
        if bigramCount > 0:
          score += math.log(bigramCount)
          score -= math.log(unigramCount)
        else:
          count = self.unigramCounts[w2]
          score += math.log(0.4*(count + 1))
          score -= math.log(self.total + len(self.unigramCounts))
    return score
