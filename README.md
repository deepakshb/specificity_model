# Specificity Model

This is unofficial implementation of "Automatic identification of general and specific sentences by leveraging discourse annotations" by Louis, Annie  and
Nenkova, Ani - 2011.

To Use this implementation, you need Penn Discourse Treebank 2.0 dataset provided by Linguistic Data Consortium. Unfortunately, it should not be uploaded publically as it requires authorization from LDC. To use this implementation, download the dataset and convert it CSV format and put the file into the base folder.

At first, you need to generate following features:

1. Sentence length
2. Polarity
3. Specificity.
4. NE+CD
5. Language models
6. Syntax
7. Word

Among these 7 features, Specificity and Language model features requires an additional dataset to compute idf value and uni,bi and tri-gram models. In my case, I am using abstract sentences of BIGPATENT dataset as my project requires to work with patent dataset but the original implementaton uses Newyork times dataset. Please make sure you name the text column as 'abstract' and label column as 'labels'.

Word features are computed using Sklearn's CountVectorizer(). This model will be stores in the 'models' folder so that it can be use be used later to annotate your test sentences

To generate features, you can use the following code:

from helpers.ExtractFeatures import ExtractFeatures
fe = ExtractFeatures(df_pdtb,df_patent)
fe.extract_features('i')                #extracts features for instantiation sentences
fe.extract_features('s')                #extracts features for specification sentences

The above code will generate features and store it in the 'features' folder. Both instantiation and specification features will be stored in this folder as pickle files. 
As it could be time consuming to generate features, hence I am uploading my set of features. Please note that the Specificity and Language Model features will depend upon the additional dataset you use.  


To read features, use the following code:

from helpers.ExtractFeatures import ReadFeatureFiles
obj_read_feats = ReadFeatureFiles()
obj_read_feats.read_features()

You can access these features using following variables:

obj_read_feats.df_i                     #pandas dataframe of non-lexical features (1-7) with labels (Instantiation dataset)
obj_read_feats.wf_i                     #Numpy matrix of lexical (word) features (Instantiation dataset)
obj_read_feats.df_s                     #pandas dataframe of lexical features (1-7) with labels
obj_read_feats.wf_s                     #Numpy matrix of lexical (word) features (Specification dataset)


For reference, you can check specificity_model.ipynb file. 
    
      
