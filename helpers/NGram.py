import pandas as pd
from nltk.util import ngrams
import itertools
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import math
import time

class NGramModel:
  def __init__(self,train_corpus):
    abstracts = []
    for abstract in train_corpus:
      tokens = word_tokenize(abstract)
      words = [word for word in tokens if word.isalpha()]
      abstracts.append(' '.join(words))

    #calculate ngrams
    bigrams = []
    unigrams = []
    trigrams = []
    for text in abstracts:
      _1gram = text.split(" ")
      _2gram = [' '.join(e) for e in ngrams(_1gram, 2)]
      _3gram = [' '.join(e) for e in ngrams(_1gram, 3)]
      bigrams.append(_2gram)
      unigrams.append(_1gram)
      trigrams.append(_3gram)

    flat_trigrams = list(itertools.chain.from_iterable(trigrams))
    flat_bigrams = list(itertools.chain.from_iterable(bigrams))
    flat_unigrams = list(itertools.chain.from_iterable(unigrams))

    self.dict_trigrams_counts = dict(Counter(flat_trigrams))
    self.dict_bigrams_counts = dict(Counter(flat_bigrams))
    self.dict_unigrams_counts = dict(Counter(flat_unigrams))

    #All bigram probabilities
    self.dict_bigrams_prob = {}
    for bigram in list(self.dict_bigrams_counts.keys()):
      bigram_words = bigram.split(' ')
      prob = self.dict_bigrams_counts[bigram]/self.dict_unigrams_counts[bigram_words[0]]
      self.dict_bigrams_prob[bigram] = prob

    #all trigram probabilities
    self.dict_trigrams_prob = {}
    for trigram in list(self.dict_trigrams_counts.keys()):
      trigram_words = trigram.split(' ')
      bigram = ' '.join(trigram_words[:2])
      prob = self.dict_trigrams_counts[trigram]/self.dict_bigrams_counts[bigram]
      self.dict_trigrams_prob[trigram] = prob
    

  def sentence_log_probability_unigram(self,sentence):
    sent_tokens = re.split(r'\W+', sentence.lower())

    all_unigrams = list(self.dict_unigrams_counts.keys())
    sent_prob = 1
    for token in sent_tokens:
      if token in all_unigrams:
        sent_prob *= self.dict_unigrams_counts[token]/len(all_unigrams)
      else:
        sent_prob *= 0
      
    log_prob = 0
    if sent_prob != 0:
      log_prob = math.log(sent_prob)
    perplexity = sent_prob ** (1.0/len(sent_tokens))
    return log_prob,perplexity


  def sentence_log_probability_bigram(self,sentence):
    
    sent_tokens = re.split(r'\W+', sentence.lower())
    sent_bigram = [' '.join(e) for e in ngrams(sent_tokens, 2)]
    
    all_unigrams = list(self.dict_unigrams_counts.keys())
    all_bigrams = list(self.dict_bigrams_counts.keys())
    
    token_1_prob = 0
    if sent_tokens[0] in all_unigrams:
        token_1_prob = self.dict_unigrams_counts[sent_tokens[0]]/len(all_unigrams)
    sent_prob = token_1_prob
    
    for i in sent_bigram:
      if i in all_bigrams:
        sent_prob *= self.dict_bigrams_prob[i]
      else:
        sent_prob *= 0
    
    log_prob = 0
    if sent_prob != 0:
      log_prob = math.log(sent_prob)
    perplexity = sent_prob ** (1.0/len(sent_tokens))
    return log_prob,perplexity

  def sentence_log_probability_trigram(self,sentence):

    sentence = sentence.lower()
    sent_tokens = re.split(r'\W+', sentence)
    sent_trigram = [' '.join(e) for e in ngrams(sent_tokens, 3)]
    
    all_unigrams = list(self.dict_unigrams_counts.keys())
    all_bigrams = list(self.dict_bigrams_counts.keys())
    all_trigrams = list(self.dict_trigrams_prob.keys())
    
    if sent_tokens[0] in all_unigrams:
      token_1_prob = self.dict_unigrams_counts[sent_tokens[0]]/len(all_unigrams)
    else:
      token_1_prob = 0

    token12_joined = ' '.join(sent_tokens[:2])

    if token12_joined in all_bigrams and sent_tokens[0] in all_unigrams:
      token12_prob = self.dict_bigrams_counts[token12_joined]/self.dict_unigrams_counts[sent_tokens[0]]
    else:
      token12_prob = 0

    sent_prob = token_1_prob * token12_prob
    
    for i in sent_trigram:
      if i in all_trigrams:
        sent_prob *= self.dict_trigrams_prob[i]
      else:
        sent_prob *= 0
    
    log_prob = 0
    if sent_prob != 0:
      log_prob = math.log(sent_prob)
    perplexity = sent_prob ** (1.0/len(sent_tokens))
    return log_prob,perplexity
   