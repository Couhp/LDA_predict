# -*- coding: utf-8 -*-
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel
from os import listdir
from os.path import join
import math
import time
import matplotlib.pyplot as plt
import itertools

###############################################################################
####################          OVERALL CLASSIFY               ##################
###############################################################################


TOPIC_SIZE = 400
CAT_SIZE = 5
FOLDER_TRAIN = "train_data"
FOLDER_TEST = "test_data"
MODEL_PATH =  "/home/phuocluu/project_cat_content/python/data/model/model_xahoi/"
MODEL_FILE_PATH = MODEL_PATH + "ldamodel.model"
DICTIONARY_PATH = MODEL_PATH + "dictionary.dict"
LEARNING_RATE = 0.005


#Load Model and File :
ldamodel = LdaModel.load(MODEL_FILE_PATH)
# print (ldamodel.print_topics(num_topics=-1))
dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

# init vector of weight for mapping between "Cat_tag" and "Topic_of_LDA"
vector = np.zeros(shape=(TOPIC_SIZE,CAT_SIZE),  dtype=float)

#Docs contain all document for training, docs_test for testing
docs = []
categories = []
docs_test = []
categories_test = []
files_name = []
list_topic = {}


def cat_change(cat) :
    if cat == '2' : return 0
    elif cat == '3' : return 1
    elif cat == '4' : return 2
    elif cat == '5' : return 3
    elif cat == '33' : return 4
    elif cat == '136' : return 5
    elif cat == '226' : return 6 

def __init__ () :

    ##############
    #  Load data train and test
    for file_name in listdir(FOLDER_TRAIN) :
        f_train = open(join(FOLDER_TRAIN,file_name),'r')
        
        categories.append((f_train.readline().rstrip()))
        docs.append( f_train.readline().rstrip())

    for file_name in listdir(FOLDER_TEST) :
        files_name.append(file_name)
        f_test = open(join(FOLDER_TEST,file_name),'r')
        
        categories_test.append((f_test.readline().rstrip()))
        docs_test.append( f_test.readline().rstrip())
    ##############

def get_Lda_Topics (doc, index=-1) :
    # Return LDA Topics for doc
    if index in list_topic :
        return list_topic[index]
    new_vec = dictionary.doc2bow(doc.split(" "))
    topics = ldamodel.get_document_topics(new_vec)
    return topics


def init_Vec () :
    # Initilization weight vector
    for _ in range(len(docs)) :
        topics = get_Lda_Topics (docs[_])
        list_topic[_] = topics
        tags = [categories[_]]

        for topic in topics :
            topic_name = topic[0]
            topic_value = topic[1]
            for tag in tags :
                if tag == None : continue
                vector[int(topic_name)][int(tag)] += float(topic_value)        
    return 

def normalize () :
    #Normalize vector by 1
    for i in range(TOPIC_SIZE) :
        raw = 0
        for j in range(CAT_SIZE) :
            raw += vector[i][j]
        if raw != 0 :
            for j in range(CAT_SIZE) :
                vector[i][j] /= raw   
    return

def predict (doc, index=-1) :
    # return prediction category of a Doc
    topics = get_Lda_Topics (doc=doc, index = index)
    result = np.zeros(shape=(CAT_SIZE))
    
    for topic in topics :
        topic_name = topic[0]
        topic_value = topic[1]
        
        result += topic_value * vector[topic_name]
    
    # sorted_index = result.argsort()
    return np.argmax(result)

def accurancy (dataset) :
    if dataset == "train" :
        counter = 0
        for _ in range(len(docs)) :            
            if str(predict(doc=docs[_],index=_)) == str(categories[_]) :
                counter += 1
        return counter / len(docs)
    if dataset == "test" :
        counter = 0
        for _ in range(len(docs_test)) :
            if str(predict(docs_test[_])) == str(categories_test[_]) :
                counter += 1
        return counter / len(docs_test)

def retrain() :
    for _ in range(len(docs)) :
        learning_rate = LEARNING_RATE
        topics = get_Lda_Topics(doc=docs[_], index=_)
        tag_true = categories[_]
        if tag_true == None : continue
        tag_false = predict(doc=docs[_],index=_)
        value = 0
        if str(tag_false) != str(tag_true) :
            # Optimize learning rate for Topic
            if int(tag_true) == 1 : learning_rate /= 2

            for topic in topics :
                topic_name = topic[0]
                topic_value = topic[1] 
                summary = np.sum(vector[int(topic_name)])
                if summary < 0.01 :
                    summary = 0.1
                
                value = summary * learning_rate * topic_value 
                #Increase True value
                vector[int(topic_name)][int(tag_true)] += value
                #Decrease false value
                if vector[int(topic_name)][int(tag_false)] > value :
                    vector[int(topic_name)][int(tag_false)] -= value
                else :
                    vector[int(topic_name)][int(tag_false)] = 0
            normalize()
    return

#==================================================================

def train () :
    #Training 
    init_Vec()
    normalize()
    while True :
        precision = accurancy("train") 
        print ("accurancy :" + str(precision))
        if precision < 0.94 :
            retrain()
        else :
            np.savetxt('array.out', vector, delimiter=',', fmt='%.5e')
            break
    

def test () :
    #Test
    print("test : ", accurancy("test"))
    


def main () :

    __init__()
    train()
    test()


if __name__ == '__main__':
    main()





