# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from os import listdir
from os.path import isfile, join
import time
import logging

# log_number = open("LDA_Log/Logging","r").read()
# update_log = open("LDA_Log/Logging","w")
# update_log.writelines(str(int(log_number) + 1))
# update_log.close()

# logging.basicConfig(filename='LDA_Log/lda_model_' + log_number + '.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
tokenizer = RegexpTokenizer(r'\w+')

    
# create sample documents
mypath = "data"

listFile = []
for file_name in listdir(mypath) :
    listFile.append(mypath + "/" + file_name)

# compile sample documents into a list
doc_set = []

start_time = time.time()
for inputFile in listFile :
    f = open(inputFile, "r")
    inputStr = f.read()
    doc_set.append(inputStr) 

end_time = time.time()   
print ("Time load : ", (end_time - start_time)*1000 , " ms")
print ("The number of documents", len(doc_set))
# loop through document list



texts = []
for doc in doc_set:
    
    texts.append(doc.split(" "))

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
dictionary.save("dictionary.dict")

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
start_time = time.time()
ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = 300, id2word = dictionary, passes = 10,eval_every=5, workers=5)
end_time = time.time()
print ("Time load : ", (end_time - start_time) , " s")

topics = ldamodel.print_topics(num_topics=-1)
for topic in topics :
    print ("Topic : ", topic)

ldamodel.save("ldamodel.model") 

# data = texts[1]
# a = ldamodel[data]
# print a
# for i in range(5) :
#     data = ldamodel.get_document_topics(corpus[i])
#     print (data)
