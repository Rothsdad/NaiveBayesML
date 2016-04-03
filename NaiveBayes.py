#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from nltk.classify import NaiveBayesClassifier as NBC
from nltk.tokenize import word_tokenize as WT
from nltk.corpus import stopwords
from nltk.probability import MLEProbDist
label_5=["food","service","price","ambience","anecdotes/miscellaneous"]
punction=['.','?','、','!']
#eng_stopwords= stopwords.words('english')
'''
'从训练数据词组中提取特征
'param: sen, list of words
'returntype: dict_word , dict of featurename:featurevalue
'    label, list of labels the sen belinging to
,eng_stop=stopwords.words('english')
'''
def featureset(clss,sen,eng_stop=stopwords.words('english'),punction=['.','?','、','!']):    
    label=[]
    dic_word=dict((('%s'%word.lower().replace('.','').replace(',',''),\
                    word.lower().replace('.','').replace(',',''))for word in sen.split()\
                   if word.lower() not in eng_stop if word not in punction))
    #dic_word=dict((('%s'%word.lower(),True)for word in sen.split()))
    for labe in clss.split():
        lab = labe.split('#')[0].replace("'",'')
        label.append(lab)
    #print(label)
    return dic_word,label

'''
'从文件中提取特征集合
'param:filename, string 文件名
'returntype，5 params，每个都是基于每一类的的特征集合
'   每一类特征集合格式：[({fname:fval,...},label),...]
'''
def getTrainFeatureSets(filename):
    path='Train/'+filename+'.txt'
    fp=open(path,'r')
    label_o=[]
    label_e=[]
    label_r=[]
    label_m=[]
    label_n=[]
    for line in fp.readlines():
        sen=line.split('//')
        dict_word,labels=featureset(sen[1],sen[2])
        for label in label_5:
            if label in labels:
                if label[1]=='o':
                    label_o.append((dict_word,label))
                elif label[1]=='e':
                    label_e.append((dict_word,label))
                elif label[1]=='r':
                    label_r.append((dict_word,label))
                elif label[1]=='m':
                    label_m.append((dict_word,label))
                else:label_n.append((dict_word,label))
            else:
                if label[1]=='o':
                    label_o.append((dict_word,'null'))
                elif label[1]=='e':
                    label_e.append((dict_word,'null'))
                elif label[1]=='r':
                    label_r.append((dict_word,'null'))
                elif label[1]=='m':
                    label_m.append((dict_word,'null'))
                else:label_n.append((dict_word,'null'))
    fp.close()
    return label_o,label_e,label_r,label_m,label_n
'''
'解析测试文件，获得样本特征
'param：filename,测试文件名
'returntype：1 param，list of the test feature sets
    [{fname:fval,...},...]
'''
def getTestFeatureSets(filename):
    path = 'Test_PhaseA/'+filename+'.txt'
    fp = open(path,'r')
    eng_stop=stopwords.words('english')
    eng_stop.extend(['.',',','?','!','(',')',"'"])
    feas=[]
    for line in fp.readlines():
        sen = line.split('//')[1]
        sam_fea=dict(('%s'%word.lower().replace('.','').replace(',',''),\
                      word.lower().replace('.','').replace(',',''))\
                     for word in WT(sen) if word.lower() not in eng_stop)
        feas.append(sam_fea)
    fp.close()
    return feas

def binaryclass(trainfile,testfile):
    label_o,label_e,label_r,label_m,label_n=getTrainFeatureSets(trainfile)
    tests=getTestFeatureSets(testfile)
    
    label_o_cls=NBC.train(label_o,MLEProbDist)
    label_e_cls=NBC.train(label_e,MLEProbDist)
    label_r_cls=NBC.train(label_r,MLEProbDist)
    label_m_cls=NBC.train(label_m,MLEProbDist)
    label_n_cls=NBC.train(label_n,MLEProbDist)
    '''
    label_o_cls=NBC.train(label_o)
    label_e_cls=NBC.train(label_e)
    label_r_cls=NBC.train(label_r)
    label_m_cls=NBC.train(label_m)
    label_n_cls=NBC.train(label_n)
    '''
    tests_o=label_o_cls.classify_many(tests)
    tests_e=label_e_cls.classify_many(tests)
    tests_r=label_r_cls.classify_many(tests)
    tests_m=label_m_cls.classify_many(tests)
    tests_n=label_n_cls.classify_many(tests)

    print([w1 for w1,w2 in label_o_cls.most_informative_features(20)])
    print([w1 for w1,w2 in label_e_cls.most_informative_features(20)])
    print([w1 for w1,w2 in label_r_cls.most_informative_features(20)])
    print([w1 for w1,w2 in label_m_cls.most_informative_features(20)])
    print([w1 for w1,w2 in label_n_cls.most_informative_features(20)])
    
    tf=open('Test_PhaseA/testFea.txt','w+')
    for sen in tests:
        tf.write(str(sen)+'\n')
    tf.close()
    
    fpr = open('Test_PhaseA/'+testfile+'.txt','r')
    fpw = open('Test_PhaseA/'+testfile+'MLECate.txt','w+')
    i=0
    for sp in fpr.readlines():
        fpw.write('#'+str(tests_o[i])+' '+'#'+str(tests_e[i])+' '+'#'+str(tests_r[i])\
                  +' '+'#'+str(tests_m[i])+' '+'#'+str(tests_n[i])+' '+sp)
        i+=1
    fpw.flush()
    fpw.close()

trainfile=raw_input('TrainFileName:')
testfile=raw_input('TestFileName:')
binaryclass(trainfile,testfile)
