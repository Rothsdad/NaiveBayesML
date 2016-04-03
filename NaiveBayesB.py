#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from nltk.classify import NaiveBayesClassifier as NBC
from nltk.tokenize import word_tokenize as WT
from nltk.probability import MLEProbDist
polabel_4=['positive','negative','neutral','conflict']
food_word=['delicious', 'spicy', 'salad', 'fried', 'dish', 'quality', 'seated',\
           'seafood', 'bland', 'entree', 'pasta', 'prepared', 'terrific','curry',\
           'food', 'minutes', 'reservation', 'selection', 'tasted','unique']
service_word=['service', 'friendly', 'staff', 'slow', 'rude', 'waiters', 'treated', \
              'asked', 'waitress', 'customer', 'minutes', 'lack', 'smile', 'offered', \
              'rushed', 'received', 'delivery', 'management', 'waiter', 'owner']
price_word=['prices', 'overpriced', 'cost', 'price', 'priced', 'considering',\
            'expensive', 'cheap', 'spend', 'beat', 'greatest', '$20', 'range',\
            'oysters', 'charge', 'value', 'bank', 'diner', 'free', 'compare']
ambience_word=['atmosphere', 'ambience', 'ambiance', 'romantic', 'cute', 'scene',\
               'loud', 'view', 'relax', 'modern', 'noisy', 'cool', 'beautiful',\
               'interior', 'paris', 'relaxing', 'clean', 'quiet', 'space', 'set']
anecdotes_word=['friendly', 'fresh', 'wine', 'trip', 'dish', 'portions', 'sauce',\
                'ambience', 'spicy', 'service', 'staff', 'pleasantly', 'deli', 'read',\
                'surprised', 'gem', 'place!', 'delicious', 'salad', 'park']
label_5=['food','service','price','anbience',"anecdotes/miscellaneous"]
stop_word=['i','me','my','myself','we','our','ours','ourselves','you','your',\
           'yours','yourself','yourselves','he','him', 'his','himself','she','her',\
           'hers','herself', 'it','its','itself','they','them','their','theirs','themselves',\
           'what','which','who','whom','this','that','these','those','am','is','are',\
           'was', 'were', 'be', 'been', 'being','have', 'has', 'had','having','do',\
           'does','did','doing','a','an','the','of','at','by','for','with',\
           'about', 'against', 'between', 'into', 'through', 'during','before','after',\
            'to', 'from','in', 'out', 'on','off','again','further','then','once','here',\
           'there', 'when', 'where', 'why', 'how', 'all', 'any','each', 'few', 'more',\
           'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so','too', 'very',
           's','can','will','just','should','now','(',')','if','','above','up']
#获得测试数据训练集合
def getTextB(B):
    path='Test_PhaseB/'
    feas=[]
    fp=open(path+B+'.txt','r')
    for line in fp.readlines():
        label,text=line.split('//')[1:]
        tokens=WT(text)
        words=[word.lower().replace('.','').replace(',','')\
               for word in tokens if word.lower() not in stop_word]
        labels=label.split()
        if(len(labels)>1):
            numwords=len(words)//len(labels)
            for i in range(len(labels)):
                 l1=labels[i].replace("'",'')[1]
                 fea_word=getRelativeWords(words,numwords,l1)
                 feas.append(fea_word)
                 pass
        else:
            feas.append(dict(("%s"%wo.lower(),labels[0].replace("'",'')[1])\
                             for wo in words))
    fp.close()
    return feas
#获得训练数据特征集合
def getTrainText(filename):
    path='Train/'+filename+'.txt'
    fp=open(path,'r')
    label_p=[]
    label_n=[]
    label_ne=[]
    label_c=[]
    for line in fp.readlines():
        label,text=line.split('//')[1:]
        tokens= WT(text)
        words=[word.lower().replace('.','').replace(',','')\
               for word in tokens if word.lower() not in stop_word]
        labels=[word.replace("'",'') for word in label.split()]
        if(len(labels)>1):
            numwords=len(words)//len(labels)    
            for i in range(len(labels)):
                mylabel=labels[i].split('#')[1][2]
                if(mylabel=='s'):
                    po_words=getRelativeWords(words,numwords,labels[i].split('#')[0][1])
                    label_p.append((po_words,'positive'))
                elif(mylabel=='g'):
                    po_words=getRelativeWords(words,numwords,labels[i].split('#')[0][1])
                    label_n.append((po_words,'negative'))
                elif(mylabel=='u'):
                    po_words=getRelativeWords(words,numwords,labels[i].split('#')[0][1])
                    label_ne.append((po_words,'neutral'))
                elif(mylabel=='n'):
                    po_words=getRelativeWords(words,numwords,labels[i].split('#')[0][1])
                    label_c.append((po_words,'conflict'))
        else:
            l1=labels[0].split('#')[1].replace("'",'')
            l2=labels[0].split('#')[0].replace("'",'')[1]
            dd=dict(('%s'%wo,l2)for wo in words)
            if(l1[2]=='s'):
                label_p.append((dd,'positive'))
            elif(l1[2]=='g'):
                label_n.append((dd,'negtive'))
            elif(l1[2]=='u'):
                label_ne.append((dd,'neutral'))
            elif(l1[2]=='n'):
                label_c.append((dd,'conflict'))            
    fp.close()
    return label_p,label_n,label_ne,label_c,label_p+label_n+label_ne+label_c
#在words中获得指定单词附近numwords个词
def getRelativeWords(words,numwords,label):
    #存储领域内单词组
    curr_words=[]
    if(label=='o'):
        relative=food_word
    elif(label=='e'):
        relative=service_word
    elif(label=='r'):
        relative=price_word
    elif(label=='m'):
        relative=ambience_word
    elif(label=='n'):
        relative=anecdotes_word
    #在words中查找单词不能存储在curr_words中
    for i in range(len(words)):
        if(words[i] in relative):
            if(i+numwords/2<len(words)and i>=numwords/2):
                ww=words[i-numwords//2:i+1+numwords//2]
            elif(i+numwords/2>=len(words)):
                ww=words[-numwords:]
            elif(i<numwords/2):
                ww=words[:numwords]
            curr_words.extend(ww)
    if(len(curr_words)>0):
        return dict(("%s" %word,label)for word in curr_words)
    else:return dict(("%s" %word,label)for word in words)

trainfile=raw_input("trainfile:")
testfile=raw_input('testfile:')
test_feas = getTextB(testfile)
p,n,ne,c,train_fea=getTrainText(trainfile)
cls=NBC.train(train_fea,MLEProbDist)
tests=cls.classify_many(test_feas)
fp=open('Test_PhaseB/MLE_END_'+testfile+'.txt','w+')
for lable in tests:
    fp.write(str(lable)+'\n')
fp.flush()
fp.close()

    
    
