import pandas as pd
import numpy as np
from helpers.Specificity import SpecificityFeatureExtraction as fe
from nltk import tokenize
from itertools import chain
import re
import os
import pickle
from tqdm import tqdm

class ExtractFeatures:
    def __init__(self,df_pdtb,df_patent,state='tr'):
        self.df_patent = df_patent
        
        if state == 'tr':
            implicit_rel = df_pdtb[df_pdtb['Relation']=='Implicit']
            instantiation_rel = implicit_rel[(implicit_rel['ConnHeadSemClass1'] == 'Expansion.Instantiation') | 
                                             (implicit_rel['ConnHeadSemClass2'] == 'Expansion.Instantiation')]
            specification_rel = implicit_rel[(implicit_rel['ConnHeadSemClass1'] == 'Expansion.Restatement.Specification') | 
                                             (implicit_rel['ConnHeadSemClass2'] == 'Expansion.Restatement.Specification')]


            full_texts_in = instantiation_rel.FullRawText.values
            full_texts_sp = specification_rel.FullRawText.values

            splitted_sents_in = [tokenize.sent_tokenize(i) for i in full_texts_in]
            splitted_sents_sp = [tokenize.sent_tokenize(i) for i in full_texts_sp]

            indexes_in = []
            for i in range(len(splitted_sents_in)):
                if len(splitted_sents_in[i]) != 2:
                    indexes_in.append(i)

            indexes_sp = []
            for i in range(len(splitted_sents_sp)):
                if len(splitted_sents_sp[i]) != 2:
                    indexes_sp.append(i)

            print('Removing ',len(indexes_in),' sentences from Instantiation.')
            for index in sorted(indexes_in, reverse=True):
                del splitted_sents_in[index]

            print('Removing ',len(indexes_sp),' sentences from Specification.')
            for index in sorted(indexes_sp, reverse=True):
                del splitted_sents_sp[index]


            self.instantiation_sents = []
            self.instantiation_labels = []

            self.specification_sents = []
            self.specification_labels = []

            for i in splitted_sents_in:
                self.instantiation_sents.append(i[0])
                self.instantiation_labels.append(0)
                self.instantiation_sents.append(i[1])
                self.instantiation_labels.append(1)

            for i in splitted_sents_sp:
                self.specification_sents.append(i[0])
                self.specification_labels.append(0)
                self.specification_sents.append(i[1])
                self.specification_labels.append(1)
        elif state == 'test':
            self.test_sents = df_pdtb.text.values
            self.test_labels = df_pdtb.labels.values
            self.state = 'test'
            
    def mpqa_lexicon_file_to_dict(self,filepath):
        with open(filepath) as f:
            lines = f.readlines()
        lines[0].strip().split(' ')
        self.dict_mpqa = {}
        for line in lines:
            dict_props = {}
            arr_props = line.strip().split(' ')
            for i in arr_props:
                pro = i.split('=')
                if len(pro) > 1:
                    dict_props[pro[0]] = pro[1]
            self.dict_mpqa[line.strip().split(' ')[2].split('=')[1]] = dict_props
        return self.dict_mpqa
    
    def generate_or_read_token_count_in_patent_docs(self,sentences,sent_type):
        if sent_type == 'i':
            file_path = 'in_sent_counts.pickle'
        elif sent_type == 's':
            file_path = 'sp_sent_counts.pickle'
        else:
            file_path = 'test_sent_counts.pickle' 
        
        if os.path.exists(file_path) and (sent_type == 'i' or sent_type == 's'):
            print('Dictionary found!')
            with open(file_path, 'rb') as handle:
                self.dict_sent_count_desc = pickle.load(handle)
        else:      
            print('Dictionary not found! Creating one...')
            sent_tokens = [re.findall(r'\w+', sentence) for sentence in sentences]
            unique_words = list(set(list(chain.from_iterable(sent_tokens))))
            self.dict_sent_count_desc = {}
            
            for index,word in enumerate(tqdm(unique_words,desc='Counting number of documents with the word.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
                
                sent_count = len(list(filter(lambda x: word in x, self.df_patent.abstract.values)))
                self.dict_sent_count_desc[word] = sent_count
    
            with open(file_path, 'wb') as handle:
                pickle.dump(self.dict_sent_count_desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def extract_features(self,sent_type = 'i'):    # 'i' for instantiation, 's' for specifications
        path = 'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
        dict_a = self.mpqa_lexicon_file_to_dict(path)
        
        if sent_type == 'i':
            str_folder_name = 'instantiation'
            labels = self.instantiation_labels
            obj_fe = fe(self.instantiation_sents)
            self.generate_or_read_token_count_in_patent_docs(self.instantiation_sents,sent_type)
        elif sent_type == 's':
            str_folder_name = 'specification'
            labels = self.specification_labels
            obj_fe = fe(self.specification_sents)
            self.generate_or_read_token_count_in_patent_docs(self.specification_sents,sent_type)
        
        #pickle labels
        with open('features/'+str_folder_name+'/'+'labels.pickle', 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
        #polarity features
        self.dict_polarity_features = obj_fe.polarity_features(dict_a)
        with open('features/'+str_folder_name+'/'+'polarity_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_polarity_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #sentence length features
        self.dict_sentence_length_features = obj_fe.sentence_length_features()
        with open('features/'+str_folder_name+'/'+'sentence_length_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_sentence_length_features, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        #specificity features
        self.dict_specificity_features = obj_fe.specificity_features(self.dict_sent_count_desc,self.df_patent.abstract.values.shape[0])
        with open('features/'+str_folder_name+'/'+'specificity_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_specificity_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #ne+cd features        
        self.dict_nedcd = obj_fe.ne_cd_features()
        with open('features/'+str_folder_name+'/'+'necd_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_nedcd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #syntactic features
        self.dict_syntactic_feat = obj_fe.syntactic_features()
        with open('features/'+str_folder_name+'/'+'syntactic_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_syntactic_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #language model features
        self.dict_lm_feat = obj_fe.language_model_features(self.df_patent.abstract.values)
        with open('features/'+str_folder_name+'/'+'lm_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_lm_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
       
        #word features
        self.dict_word_feat = obj_fe.word_features(state='t',sent_type=sent_type)
        with open('features/'+str_folder_name+'/'+'word_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_word_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

    def extract_test_features(self,sent_type='i'):    #here sent_type is used whether to extract features according to ins or sp   
        path = '/Users/deepak/Desktop/thesis_pqai/specificity_model/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
        dict_a = self.mpqa_lexicon_file_to_dict(path)
        
        str_folder_name = 'test'
        labels = self.test_labels
        
        obj_fe = fe(self.test_sents)
        self.generate_or_read_token_count_in_patent_docs(self.test_sents,'t')
        
        
        #pickle labels
        with open('features/'+str_folder_name+'/'+'labels.pickle', 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
        #polarity features
        self.dict_polarity_features = obj_fe.polarity_features(dict_a)
        with open('features/'+str_folder_name+'/'+'polarity_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_polarity_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #sentence length features
        self.dict_sentence_length_features = obj_fe.sentence_length_features()
        with open('features/'+str_folder_name+'/'+'sentence_length_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_sentence_length_features, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        #specificity features
        self.dict_specificity_features = obj_fe.specificity_features(self.dict_sent_count_desc,self.df_patent.abstract.values.shape[0])
        with open('features/'+str_folder_name+'/'+'specificity_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_specificity_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #ne+cd features        
        self.dict_nedcd = obj_fe.ne_cd_features()
        with open('features/'+str_folder_name+'/'+'necd_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_nedcd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #syntactic features
        self.dict_syntactic_feat = obj_fe.syntactic_features()
        with open('features/'+str_folder_name+'/'+'syntactic_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_syntactic_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #language model features
        self.dict_lm_feat = obj_fe.language_model_features(self.df_patent.abstract.values)
        with open('features/'+str_folder_name+'/'+'lm_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_lm_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
       
        #word features
        self.dict_word_feat = obj_fe.word_features(state='p',sent_type=sent_type)
        with open('features/'+str_folder_name+'/'+'word_features.pickle', 'wb') as handle:
            pickle.dump(self.dict_word_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        all_feat_dicts = [self.dict_polarity_features,self.dict_sentence_length_features,self.dict_specificity_features,
                             self.dict_nedcd,self.dict_syntactic_feat,self.dict_lm_feat]
        
        feats = {}
        for i in all_feat_dicts:
            feats.update(i)
        df = pd.DataFrame(feats)
        X = np.concatenate((df.values,self.dict_word_feat['word_feat'].todense()),axis = 1)
        
        return X
        
class ReadFeatureFiles:
    def __init__(self):
        self.ins_folder_path = 'features/instantiation'
        self.sp_folder_path = 'features/specification'
    
    def check_if_features_available(self,sent_type='i'):
        if sent_type == 'i':
            folder_path = self.ins_folder_path
        elif sent_type == 's':
            folder_path = self.sp_folder_path

        if len(os.listdir(folder_path)) == 0:
            feat_available = False
        else:    
            feat_available = True
        return feat_available
    
    def read_files(self,sent_type):
        if sent_type == 'i':
            folder_path = self.ins_folder_path
        elif sent_type == 's':
            folder_path = self.sp_folder_path

        file_names = ['necd_features.pickle','polarity_features.pickle','sentence_length_features.pickle',
                      'specificity_features.pickle','syntactic_features.pickle','lm_features.pickle']
        
        feats = {}
        for name in file_names:
            file_path = folder_path+'/'+name
            print(file_path)
            with open(file_path, 'rb') as handle:
                dict_f = pickle.load(handle)
                feats.update(dict_f)
        
        #Read labels
        with open(folder_path+'/'+'labels.pickle', 'rb') as handle:
                labels = pickle.load(handle)
        feats.update({'labels':labels})
        df = pd.DataFrame(feats)
        
        #word features need to be read_sepeately
        with open(folder_path+'/'+'word_features.pickle', 'rb') as handle:
            dict_wf = pickle.load(handle)
        word_feat = dict_wf['word_feat'].todense()
        return df,word_feat
    
    def read_features(self):
        if self.check_if_features_available('i'):
            print('Instantiation Features found!')
            self.df_i,self.wf_i = self.read_files(sent_type='i')
        else:
            print('Instantiation Features not found! Generate them first.')

        #Specification
        if self.check_if_features_available('s'):
            print('Specification Features found!')
            self.df_s,self.wf_s = self.read_files(sent_type='s')
        else:
            print('Specification Features not found! Generate them first.')  