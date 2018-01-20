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


####################################################################################
###############     ANALYSIS FOR SIMILARITY OF "GAMEK" AND "GENK"    ###############
####################################################################################


TOPIC_SIZE = 300
CAT_SIZE = 3
FOLDER_TRAIN = "train_data"
DATA_FOLDER = FOLDER_TRAIN
FOLDER_TEST = "test_data"
MODEL_PATH =  "/home/phuocluu/project_cat_content/python/data/model/model_xahoi/"
MODEL_FILE_PATH = MODEL_PATH + "ldamodel.model"
DICTIONARY_PATH = MODEL_PATH + "dictionary.dict"
LEARNING_RATE = 0.05

#Load Model and File :
ldamodel = LdaModel.load(MODEL_FILE_PATH)
# print (ldamodel.print_topics(num_topics=-1))
dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

# init vector of weight for mapping between "Cat_tag" and "Topic_of_LDA"
vector = np.loadtxt('array.out', delimiter=',')

#Docs contain all document for training, docs_test for testing
docs = []
categories = []
kt_vec = np.zeros(shape=(300), dtype=float)
xh_vec = np.zeros(shape=(300), dtype=float)
xe_vec = np.zeros(shape=(300), dtype=float)

def get_Lda_Topics (doc) :
    # Return LDA Topics for doc
    new_vec = dictionary.doc2bow(doc.split(" "))
    topics = ldamodel.get_document_topics(new_vec)
    return topics

def entropy(input) :
    sum = 0
    for _ in input :
        sum += _
    result = 0
    for _ in input :
        result += - (_/sum) * (np.log([(_/sum)])[0])
    return result

def __init__ () :
    
    for file_name in listdir(DATA_FOLDER) :
        fin = open(join(DATA_FOLDER,file_name), "r")
        
        categories.append(fin.readline().rstrip())
        docs.append(fin.readline().rstrip())

    for _ in range(len(docs)) :
        topics = get_Lda_Topics(docs[_])
        
        for topic in topics :
            if categories[_] == "0" :
                topic_name = topic[0]
                topic_value = topic[1]
                if topic_value > 0.1 :
                    xh_vec[topic_name] += 1
            if categories[_] == "3" :    
                topic_name = topic[0]
                topic_value = topic[1]
                if topic_value > 0.1 :
                    kt_vec[topic_name] += 1
            if categories[_] == "7" :    
                topic_name = topic[0]
                topic_value = topic[1]
                if topic_value > 0.1 :
                    xe_vec[topic_name] += 1

################    PRESENTATION    ############################
################################################################
def process () :
    belief_topic = []
    for _ in range(TOPIC_SIZE) :
        a = xh_vec[_] + 1
        b = kt_vec[_] + 1
        c = xe_vec[_] + 1
        entropy_val = entropy([a,b,c])
        # print (_, " " ,[a,b,c], " : ", entropy_val,"\n")
        if entropy_val < 0.8 :
            belief_topic.append(_)
        else :
            xh_vec[_] = 0
            kt_vec[_] = 0
            xe_vec[_] = 0
    print (belief_topic)

def presentation () :
    ind = np.arange(300)  # the x locations for the groups
    

    plt.subplots()
    
    rects1 = plt.bar(ind, xh_vec, color='r')
    rects2 = plt.bar(ind, kt_vec, color='y')
    rects3 = plt.bar(ind, xe_vec, color='b')

    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    # plt.xticks(ind + width / 2)
    
    plt.legend((rects1[0], rects2[0],  rects3[0]), ('xa hoi', 'kinh te','xe'))

    plt.show()

def main() :
    __init__()
    process()
    presentation()



if __name__ == '__main__':
    main()





    

