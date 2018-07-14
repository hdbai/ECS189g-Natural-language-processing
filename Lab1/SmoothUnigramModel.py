import math, collections

class SmoothUnigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothUnigramCounts = collections.defaultdict(lambda: 0)
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
    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.smoothUnigramCounts[token] = self.smoothUnigramCounts[token] + 1
        self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for token in sentence:
      if self.smoothUnigramCounts[token]:
        count = self.smoothUnigramCounts[token]
      else:
        count = 0
      score += math.log(count+1)
      score -= math.log(self.total+len(self.smoothUnigramCounts))
    return score
