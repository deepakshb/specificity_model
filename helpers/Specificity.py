import pandas as pd
import numpy as np
from nltk import tokenize
import pysentiment2 as ps
import spacy
import re
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from time import sleep
from tqdm import tqdm
from helpers.NGram import NGramModel
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import CountVectorizer

class SpecificityFeatureExtraction:
    def __init__(self,sentences):
        self.texts = sentences
        self.nn_syn = [syn.name() for syn in list(wn.all_synsets('n'))]
        self.vb_syn = [syn.name() for syn in list(wn.all_synsets('v'))]
    
    def count_words(self,sentence):
        return len(re.findall(r'\w+', sentence))
    
    def normalization_score(self,dict_scores):
        positive_counts = dict_scores['Positive']
        negative_counts = dict_scores['Negative']
        total_words = dict_scores['word_count']
        if (positive_counts - negative_counts) != 0:
            norm_score = (total_words-min(positive_counts,negative_counts))/(max(positive_counts,negative_counts)-min(positive_counts,negative_counts))
        else: 
            norm_score = 0
        return norm_score
    
    def get_phrases(self,doc,pos_type):
        "Function to get PPs from a parsed document."
        phs = []
        for token in doc:
            # Try this with other parts of speech for different subtrees.
            if token.pos_ == pos_type:
                pp = ' '.join([tok.orth_ for tok in token.subtree])
                phs.append(pp)
        return phs
        
    
    def sentence_length_features(self):
        dict_sent_length_features = {'num_words':[],'noun_count':[]}
        nlp = spacy.load('en_core_web_sm')
        
        
        for index,text in enumerate(tqdm(self.texts,desc='Collecting Sentence Length features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            num_words = self.count_words(text)
            dict_sent_length_features['num_words'].append(num_words)
            
            doc = nlp(text)

            noun_count = 0
            for token in doc:
                if token.pos_ == 'NOUN':
                    self.noun_count = noun_count + 1
            dict_sent_length_features['noun_count'].append(noun_count)
        return dict_sent_length_features
    
    def polarity_features(self,dict_mpqa):
        dict_polarity_features = {'tgi_pos':[],'tgi_neg':[],'tgi_polar':[],'tgi_word_count':[],'tgi_norm_score':[],
                                 'mpqa_pos':[],'mpqa_neg':[],'mpqa_polar':[],'mpqa_word_count':[],'mpqa_norm_score':[]}
        hiv4 = ps.HIV4()
       
        for index,text in enumerate(tqdm(self.texts,desc='Collecting Polarity features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            tokens = hiv4.tokenize(text)
            tgi_polarity_scores = hiv4.get_score(tokens)
            tgi_polarity_scores['word_count'] = self.count_words(text)
            dict_polarity_features['tgi_pos'].append(tgi_polarity_scores['Positive'])
            dict_polarity_features['tgi_neg'].append(tgi_polarity_scores['Negative'])
            dict_polarity_features['tgi_polar'].append(tgi_polarity_scores['Polarity'])
            dict_polarity_features['tgi_word_count'].append(tgi_polarity_scores['word_count'])
            dict_polarity_features['tgi_norm_score'].append(self.normalization_score(tgi_polarity_scores))

            words = re.findall(r'\w+', text)
            pos_count = 0
            neg_count = 0
            for word in words:
                if word in list(dict_mpqa.keys()):
                    if dict_mpqa[word]['priorpolarity'] == 'negative':
                        neg_count = neg_count + 1
                    else:
                        pos_count = pos_count + 1
            polarity_score = 0
            if pos_count + neg_count != 0:
                polarity_score = (pos_count - neg_count)/(pos_count + neg_count)
              
            mpqa_polarity_scores = {'Positive':pos_count,'Negative':neg_count,'Polarity':polarity_score,'word_count':self.count_words(text)}
            dict_polarity_features['mpqa_pos'].append(pos_count)
            dict_polarity_features['mpqa_neg'].append(neg_count)
            dict_polarity_features['mpqa_polar'].append(polarity_score)
            dict_polarity_features['mpqa_word_count'].append(self.count_words(text))
            dict_polarity_features['mpqa_norm_score'].append(self.normalization_score(mpqa_polarity_scores))
        return dict_polarity_features          
    
    def specificity_features(self,dict_sent_counts,corpus_length):
        
        dict_specificity_features = {'avg_noun_dist':[],'min_noun_dist':[],'max_noun_dist':[],'avg_verb_dist':[],'min_verb_dist':[],
        'max_verb_dist':[],'avg_idfs':[],'min_idf':[],'max_idf':[]}
        
        nlp = spacy.load('en_core_web_sm')
   
        for index,text in enumerate(tqdm(self.texts,desc='Collecting Specificity features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            doc = nlp(text)
            noun_paths = []
            verb_paths = []
            #wn_lemmas = set(wn.all_lemma_names())

            for token in doc:
                if token.pos_ == 'NOUN' and token.text.isalpha():
                    syn = token.lemma_+'.n.01'
                    if syn in self.nn_syn:
                        ss = wn.synset(syn)
                        noun_paths.append(min([len(path) for path in ss.hypernym_paths()]))
                                        
                elif token.pos_ == 'VERB' and token.text.isalpha():
                    syn = token.lemma_+'.v.01'
                    if syn in self.vb_syn:
                        ss = wn.synset(syn)
                        verb_paths.append(min([len(path) for path in ss.hypernym_paths()]))
            
            if len(noun_paths) > 0:
                avg_noun_dist = sum(noun_paths)/len(noun_paths)
                min_noun_dist = min(noun_paths)
                max_noun_dist = max(noun_paths)
            else:
                avg_noun_dist = 0
                min_noun_dist = 0
                max_noun_dist = 0
            dict_specificity_features['avg_noun_dist'].append(avg_noun_dist)
            dict_specificity_features['min_noun_dist'].append(min_noun_dist)
            dict_specificity_features['max_noun_dist'].append(max_noun_dist)
            
            if len(verb_paths) > 0:
                avg_verb_dist = sum(verb_paths)/len(verb_paths)
                min_verb_dist = min(verb_paths)
                max_verb_dist = max(verb_paths)
            else:
                avg_verb_dist = 0
                min_verb_dist = 0
                max_verb_dist = 0
            
            dict_specificity_features['avg_verb_dist'].append(avg_verb_dist)
            dict_specificity_features['min_verb_dist'].append(min_verb_dist)
            dict_specificity_features['max_verb_dist'].append(max_verb_dist)    
                
            #idf of words in different corpus
            #find number of documents that contain the specific word
            # idf = number of documents having the word/total number of documents
            
            words = re.findall(r'\w+', text)
            idfs = []
            for word in words:
                dict_sent_counts
                if word not in list(dict_sent_counts.keys()):
                    desc_with_word_count = 1
                else:
                    if dict_sent_counts[word] == 0:
                        desc_with_word_count = 1
                    else:
                        desc_with_word_count = dict_sent_counts[word]
          
            #desc_with_word_count = 0
            #for desc in arr_descriptions:
            #    if word in desc:
            #        desc_with_word_count = desc_with_word_count + 1
            
                idf_val = desc_with_word_count/corpus_length
                idfs.append(idf_val)
            
            dict_specificity_features['avg_idfs'].append(sum(idfs)/len(idfs))
            dict_specificity_features['min_idf'].append(min(idfs))
            dict_specificity_features['max_idf'].append(max(idfs))    
        return dict_specificity_features
        
    def ne_cd_features(self):
        dict_necd_features = {'number_token_count':[],'plural_noun_count':[],'proper_noun_count':[]}

        nlp = spacy.load('en_core_web_sm')
        for index,text in enumerate(tqdm(self.texts,'Collecting NE+CD features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
        
            doc = nlp(text)
            number_token_count = 0
            plural_noun_count = 0
            proper_noun_count = 0
            for token in doc:
                if token.pos_ == 'NUM':
                    number_token_count = number_token_count + 1
                elif token.pos_ == 'NOUN' and token.tag_ == 'NNS':
                    plural_noun_count = plural_noun_count + 1
                elif token.pos_ == 'PROPN':
                    proper_noun_count = proper_noun_count + 1
            dict_necd_features['number_token_count'].append(number_token_count)
            dict_necd_features['plural_noun_count'].append(plural_noun_count)
            dict_necd_features['proper_noun_count'].append(proper_noun_count)
        return dict_necd_features
        
    def syntactic_features(self):
        nlp = spacy.load('en_core_web_sm')
        dict_syntactic_features={'adjective_count':[],'adverb_count':[],'adjective_phrases_count':[],'adverbial_phrases_count':[],
        'prepositional_phrases_count':[],'verb_phrase_count':[],'avg_length_of_verb_phrases':[]}
        
        for index,text in enumerate(tqdm(self.texts,desc='Collection syntactic features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            
            doc = nlp(text)
        
            adjective_count = 0
            adverb_count = 0
        
            for token in doc:
                if token.pos_ == 'ADJ':
                    adjective_count = adjective_count + 1
                elif token.pos == 'ADV':
                    adverb_count = adverb_count + 1
            
            dict_syntactic_features['adjective_count'].append(adjective_count)
            dict_syntactic_features['adverb_count'].append(adverb_count)
            
            dict_syntactic_features['adjective_phrases_count'].append(len(self.get_phrases(doc,pos_type='ADJ')))
            dict_syntactic_features['adverbial_phrases_count'].append(len(self.get_phrases(doc,pos_type='ADV')))
            dict_syntactic_features['prepositional_phrases_count'].append(len(self.get_phrases(doc,pos_type='ADP')))

        
            verb_phrases = self.get_phrases(doc,pos_type='VERB')
            dict_syntactic_features['verb_phrase_count'].append(len(verb_phrases))
            
            
            word_counts_vp = [self.count_words(phrase) for phrase in verb_phrases]
            if len(word_counts_vp) != 0:
                dict_syntactic_features['avg_length_of_verb_phrases'].append(sum(word_counts_vp)/len(word_counts_vp)) 
            else:
                dict_syntactic_features['avg_length_of_verb_phrases'].append(0)
        return dict_syntactic_features
    
    def language_model_features(self,patent_abstracts):
        dict_language_model_feat = {'unigram_log_prob':[],'unigram_pp':[],
                                    'bigram_log_prob':[],'bigram_pp':[],'trigram_log_prob':[],'trigram_pp':[]}
        obj_ngram = NGramModel(patent_abstracts)
        for index,text in enumerate(tqdm(self.texts,desc='Collection Language Model features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            
            prob_u,pp_u = obj_ngram.sentence_log_probability_unigram(text)
            prob_b,pp_b = obj_ngram.sentence_log_probability_bigram(text)
            prob_t,pp_t = obj_ngram.sentence_log_probability_trigram(text)
            
            dict_language_model_feat['unigram_log_prob'].append(prob_u)
            dict_language_model_feat['unigram_pp'].append(pp_u)
            dict_language_model_feat['bigram_log_prob'].append(prob_b)
            dict_language_model_feat['bigram_pp'].append(pp_b)
            dict_language_model_feat['trigram_log_prob'].append(prob_t)
            dict_language_model_feat['trigram_pp'].append(pp_t)
        
        return dict_language_model_feat
    
    def word_features(self,state='t',sent_type='i'):  #state can be 't' for training or 'p' for prediction  
        sentences = []
        print('Collecting word features.')
        for sent in self.texts:
            tokens = word_tokenize(sent)
            words = [word for word in tokens if word.isalpha()]
            sentences.append(' '.join(words))
        
        if sent_type == 'i':
            file_path = '/Users/deepak/Desktop/thesis_pqai/specificity_model/models/count_vec_in.pickle'
        elif sent_type == 's':
            file_path = '/Users/deepak/Desktop/thesis_pqai/specificity_model/models/count_vec_sp.pickle'
        
        
        dict_word_feat = {}
        if state == 't':
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(sentences)
            with open(file_path, 'wb') as fout:
                pickle.dump(vectorizer, fout)
        elif state == 'p':
            with open(file_path, 'rb') as f:
                vectorizer = pickle.load(f)
            X = vectorizer.transform(sentences)
            
        dict_word_feat['word_feat']=X
        return dict_word_feat
        
        
        
        
        
        
        
        
        