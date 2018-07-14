# coding=utf-8
import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
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
        w_i = datums[i].word
        self.unigramCounts[w_i] += 1
        if i >= 1:
          w_i_1 = datums[i-1].word
          self.bigramCounts[(w_i_1,w_i)] += 1
  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in xrange(0, len(sentence)):
      if i >= 1:
        w_i = sentence[i]
        w_i_1 = sentence[i-1]
        bigramCount = self.bigramCounts[((w_i_1,w_i))]
        unigramCount = self.unigramCounts[w_i_1]
        # PAdd−1(wi | wi−1) = c(wi−1,wi)+1 / c(wi−1)+V , V is number of times we add one which is len(self.bigramCounts))
        score += math.log(bigramCount + 1)
        score -= math.log(unigramCount+len(self.bigramCounts))
    return score
