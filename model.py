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
VECTOR = "array.out"#MODEL_PATH + "array.out"

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
    global vector
    vector = np.loadtxt(fname=VECTOR,delimiter=',')
    print (np.shape(vector))
    for file_name in listdir(FOLDER_TRAIN) :
        f_train = open(join(FOLDER_TRAIN,file_name),'r')
        tag = f_train.readline().rstrip()
        value = f_train.readline().rstrip()

        # if cat_change(tag) == None :
        #     print (tag)
        categories.append((tag))
        docs.append( value)

    for file_name in listdir(FOLDER_TEST) :
        files_name.append(file_name)
        f_test = open(join(FOLDER_TEST,file_name),'r')
        
        categories_test.append( (f_test.readline().rstrip()))
        docs_test.append( f_test.readline().rstrip())
    ##############


def get_Lda_Topics (doc, index=-1) :
    # Return LDA Topics for doc
    if index in list_topic :
        return list_topic[index]
    new_vec = dictionary.doc2bow(doc.split(" "))
    topics = ldamodel.get_document_topics(new_vec)
    return topics

def predict (doc) :
    # return prediction category of a Doc
    topics = get_Lda_Topics (doc=doc)
    result = np.zeros(shape=(CAT_SIZE))
    
    for topic in topics :
        topic_name = topic[0]
        topic_value = topic[1]
        result += topic_value * vector[topic_name]
    # sorted_index = result.argsort()
    return np.argmax(result)


#=========================================================================
#=================   PRESENTATION   ======================================
#=========================================================================
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.show()


def distributed(index) :
    # Distributed presentation
    distributed_array = np.zeros(shape=(TOPIC_SIZE),dtype=int)
    for _ in range(len(docs_test)) :
        if predict(docs_test[_]) == index :
            topics = get_Lda_Topics(docs_test[_])
            for topic in topics :
                topic_name = topic[0]
                topic_value = topic[1]
                if topic_value > 0.1 :
                    distributed_array[topic_name] += 1 
    distributed_array = np.sort(distributed_array)
    presenter = plt.subplot(2,math.ceil(CAT_SIZE/2),index + 1) 
    # if index > CAT_SIZE/2 :
    #     presenter = plt.subplot(2,CAT_SIZE,index + 1)
    
    class_names = ['kinhte','thoisu','quansu','xe']
    plt.title(class_names[index])
    plt.ylabel('Frequency')
    # print(distributed_array)
    distributed_array = distributed_array[-201:-1]
    # print (distributed_array)
    ind = np.arange(200)
    presenter.bar(ind, distributed_array)
    if index == CAT_SIZE - 1 :
        plt.show()

def confusion_matrix (categories_test, categories_pred, classtype=0) :
    def my_confusion_matrix(y_true, y_pred):
        print (y_true)
        y_true = np.array([int(i) for i in y_true])
        y_pred = np.array(y_pred)

        cm = np.zeros((CAT_SIZE, CAT_SIZE))
        for n in range(y_true.shape[0]):
            cm[y_true[n], y_pred[n]] += 1
        return cm
    cnf_matrix = my_confusion_matrix(categories_test, categories_pred)
    class_names = []
    if classtype == 0 : 
        class_names = ['Kinh te','Xa hoi', 'quan su','xe co']
    else :
        class_names = ['Tech','game']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix for 570')
#============================================================================
#============================================================================

def test (dataset,classtype=0) :
    #Test
    if dataset == "test" :
        categories_pred = []
        for _ in range(len(docs_test)) :
                categories_pred.append(predict(docs_test[_]))
        
        print (categories, categories_pred)
        with open("checker.txt","w") as fout :
            fout.write("")
        for _ in range(len(categories_test)) :
            if categories_test[_] == 2 and categories_pred[_] == 5 :
                with open("checker.txt","a") as fout :
                    fout.write("\n" + docs_test[_])

        # Presentation
        confusion_matrix(categories_test, categories_pred, classtype=classtype)
    
    if dataset == "train" :
        categories_pred = []
        for _ in range(len(docs )) :
                categories_pred.append(predict(docs [_]))
        
        # with open("checker.txt","w") as fout :
        #     fout.write("")
        # for _ in range(len(categories )) :
        #     if categories [_] == 1 and categories_pred[_] == 0 :
        #         with open("checker.txt","a") as fout :
        #             fout.write("\n" + docs [_])

        # Presentation
        confusion_matrix(categories , categories_pred, classtype=classtype)




def main () :

    __init__()
    # train()
    test("test", classtype=0)


if __name__ == '__main__':
    main()


    