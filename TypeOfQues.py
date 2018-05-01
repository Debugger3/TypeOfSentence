
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Word2Vec


# In[2]:


data=pd.read_csv("LabelledData.txt",sep=",,,")


# In[3]:


data=shuffle(data, random_state=0)
print data.shape
X=data['sentence']
label=data['label']
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.20, random_state=42)


# In[4]:


countvect = CountVectorizer(max_features=100)
countVectorTrain=countvect.fit_transform(X_train)
countVectorTest=countvect.transform(X_test)
countfeatureTrain=pd.DataFrame(countVectorTrain.toarray(), columns=countvect.get_feature_names()) 
countfeatureTest=pd.DataFrame(countVectorTest.toarray(), columns=countvect.get_feature_names()) 

tfidfvect = TfidfVectorizer(max_features=100)
tfidfVectorTrain = tfidfvect.fit_transform(X_train)
tfidfVectorTest = tfidfvect.transform(X_test)
tfidfeatureTrain=pd.DataFrame(tfidfVectorTrain.toarray(), columns=tfidfvect.get_feature_names()) 
tfidfeatureTest=pd.DataFrame(tfidfVectorTest.toarray(), columns=tfidfvect.get_feature_names()) 


# In[5]:


print tfidfeatureTrain.shape
print countfeatureTrain.shape

print tfidfeatureTest.shape
print countfeatureTest.shape


# In[6]:


featureCount=countvect.get_feature_names()
featureTfidf=tfidfvect.get_feature_names()


# In[7]:


def modelTrainAndAccuracy(model,trainingfeature,testingfeature):
    clf=None
    if(model=="logistic"):
        clf = LogisticRegression()
        clf.fit(trainingfeature, y_train)
        
    elif (model=="decisiontree"):
        clf = DecisionTreeClassifier().fit(trainingfeature, y_train)
        
    elif(model=="knn"):
        clf = KNeighborsClassifier()
        clf.fit(trainingfeature, y_train)
        
    elif(model=="naive"):
        clf = MultinomialNB().fit(trainingfeature, y_train)
    
    if(clf!=None):
        print 'Accuracy of ',model,' classifier on training set: {:.2f}'.format(clf.score(trainingfeature, y_train))
        print 'Accuracy of ',model,' classifier on test set: {:.2f}'.format(clf.score(testingfeature, y_test))
    


# In[8]:


modelList=["logistic","decisiontree","knn","naive"]
print "---------------------Result using count vector as features---------------"
for model in modelList:
    modelTrainAndAccuracy(model,countfeatureTrain,countfeatureTest)
print "----------------Result using TFIDF as features-------------------------"
for model in modelList:
    modelTrainAndAccuracy(model,tfidfeatureTrain,tfidfeatureTest)


# In[9]:


#Best algorithm seems to be using logisitc regression using count vector as feature

