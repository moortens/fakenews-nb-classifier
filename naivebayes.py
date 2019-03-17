import pandas as pd
import math 
import string
import re
import nltk
import json
import argparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# using train_test_split as the training dataset originally given did
# not contain labels, and therefore did not allow for calculating the
# accuracy.
from sklearn.model_selection import train_test_split
nltk.download(['stopwords', 'punkt'], quiet=True)

class NaiveBayesClassifier:
  def __init__(self, file: str = None, vocabulary: str = 'data/vocabulary.csv', train: str = 'data/train.csv'):
    '''
    initializes the naive bayes classifier
    '''

    # load the english stopwords
    self.stopwords = stopwords.words('english')

    # initialize the porter stemmer from the nltk package, used to find the root of a word
    self.porter = PorterStemmer()

    # a simple regular expression to remove punctuation (!,.?, etc).
    self.punctuation_pattern = re.compile('[{}]'.format(string.punctuation))

    self.vocabulary = pd.read_csv(vocabulary,  header=None, names=['words', 'frequency'])
    self.dataset, self.testset = train_test_split(pd.read_csv(train))

    if file is not None:
      self.load_data(file)

  def load_data(self, file):
    '''
    loads a pretrained model from disk
    '''
    if file is not None:
      try:
        with open(file, 'r') as db:
          data = json.load(db)

          self.reliable = data.get('reliable')
          self.unreliable = data.get('unreliable')
          self.prior_reliable_probability = data.get('prior_reliable_probability')
          self.prior_unreliable_probability = data.get('prior_unreliable_probability')
        return True
      except FileNotFoundError:
        return False


  def _prior_probabilities(self):
    '''
    calculates the prior probabilities to use in classify() 
    '''
    self.label_count = self.dataset.groupby('label').size()
    self.total_count = self.dataset['label'].size

    self.prior_reliable_probability = self.label_count[0]/self.total_count
    self.prior_unreliable_probability = self.label_count[1]/self.total_count

  def train(self, vocabulary_threshold: int = 0):
    '''
    trains the naive bayes classifier, building two vocabularies from
    the dataset.

    allows to set the vocabulary_threshold to test whether more occurances
    of a word increases or decreases accuracy
    '''
    self._prior_probabilities()

    real = self.build_vocabulary(self.dataset[self.dataset.label == 0])
    fake = self.build_vocabulary(self.dataset[self.dataset.label == 1])
    
    self.reliable = {}
    self.unreliable = {}
    
    for w in self.vocabulary.itertuples():
      if w.frequency > vocabulary_threshold:
        if (w.words in real):
          self.reliable[w.words] = real[w.words]
        if (w.words in fake):
          self.unreliable[w.words] = fake[w.words]

  def save(self, file: str = 'model.json'):
    '''
    saves the currently trained model to disk, for faster 
    classification in succeeding executions of the program.
    '''
    v = {
      "reliable": self.reliable,
      "unreliable": self.unreliable,
      "prior_reliable_probability": self.prior_reliable_probability,
      "prior_unreliable_probability": self.prior_unreliable_probability
    }

    with open(file, 'w') as db:
      json.dump(v, db)

  def classify(self, text, alpha: float = 1.0):
    '''
    classifies the text and returns 0 if fake and 1 if real using 
    naive bayes with logarithms. 
    '''

    vocab_size = len(self.reliable) + len(self.unreliable)

    real = math.log(self.prior_reliable_probability)
    fake = math.log(self.prior_unreliable_probability)
    
    vocab_reliable = sum(self.reliable.values())
    vocab_unreliable = sum(self.unreliable.values())

    words = self.__tokenize_string(text)
    for w in words:
      real += math.log((self.reliable.get(w, 0.0) + alpha) / (vocab_reliable + vocab_size))
      fake += math.log((self.unreliable.get(w, 0.0) + alpha) / (vocab_unreliable + vocab_size))

    if fake > real:
      return 1
    return 0

  def test(self, alpha: float = 1.0):
    '''
    test goes through the classification of the training and compares it to the actual answer
    it prints the number of counted real and fake articles and the real and fake classified articles

    it returns the percentage of accuracy in decimal
    '''
    z = {}
    z['correct'] = 0
    z['incorrect'] = 0
    for y in self.testset.itertuples():
      i = self.classify(str(y.text), alpha=alpha)
  
      if i is y.label:
        z['correct'] += 1
      else: 
        z['incorrect'] += 1    
    lcount = len(self.testset)
    print("Classified:", lcount)
    print("# Actual real", len(self.testset[self.testset.label == 0]))
    print("# Actual fake", len(self.testset[self.testset.label == 1]))
    print("# Classified correctly:", z['correct'])
    print("# Classified incorrectly:", z['incorrect'])

    return z['correct'] / (z['correct'] + z['incorrect'])

  def __tokenize_string(self, text):
    '''
    tokenizes a string, removing punctuation and stopwords
    '''
    text = str(text)
    text = re.sub(self.punctuation_pattern, '', text)
    return [self.porter.stem(w) for w in text.split() if w.isalpha() and w not in self.stopwords]

  def build_vocabulary(self, series):
    '''
    build_vocabulary takes a series (an entire column) from a Panda DataFrame and 
    assuming its a list of words, it counts the occurances of each word. 

    It then returns a new DataFrame from the resulting vocabulary.
    '''
  
    vocab = {}
    # turn the series into a list, for easier iteration
    
    for item in series.itertuples():
      ##print(item.text
      words = self.__tokenize_string(item.text)
      for w in words:
        if not w in vocab: # assuming a word is not in the dictionary, set it to one occurance
          vocab[w] = 1
        else:
          vocab[w] += 1

    # return a new vocabulary
    return vocab
