import pandas as pd
import os
import pickle
import numpy as np

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