#!/usr/bin/python
# -*- coding: UTF-8 -*-

try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import sys
from nltk.corpus import stopwords as st
from nltk.tokenize import word_tokenize as wt

def filepath(filename):
    if(filename[-1]=='n'):
        return 'Train/'+filename
    elif(filename[-1]=='A'):
        return 'Test_PhaseA/'+filename
    else:
        return 'Test_PhaseB/'+filename

def parseTrain():
    filename=raw_input('filename:')
    path=filepath(filename)
    
    try:
        tree=ET.parse(path+'.xml')
        root=tree.getroot()
    except Exception:
        raise('%s failed'%filename)

    fp = open(path+'.txt','w+')
    for sen in root.findall('sentence'):
        uid=sen.get('id')
        text=sen.find('text').text
        categories = sen.find('aspectCategories')
        catepo=[(str(category.get('category'))+'#'+str(category.get('polarity')))\
                 for category in categories.findall('aspectCategory')]        
        fp.write(str(uid)+'//'+str(catepo).replace('[','').replace(']',' ').replace(',',' ')\
                 +'//'+str(text)+'\n')
    fp.flush()
    fp.close()
def parseTestA():
    filename=raw_input('TestAfilename:')
    path=filepath(filename)    
    try:
        tree=ET.parse(path+'.xml')
        root=tree.getroot()
    except Exception:
        raise('%s failed'%filename)

    fp = open(path+'.txt','w+')
    for sen in root.findall('sentence'):
        uid=sen.get('id')
        text=sen.find('text').text
        fp.write(str(uid)+'//'+str(text)+'\n')
    fp.flush()
    fp.close()
def parseTestB():
    filename=raw_input('TestBfilename:')
    path=filepath(filename)    
    try:
        tree=ET.parse(path+'.xml')
        root=tree.getroot()
    except Exception:
        raise('%s failed'%filename)

    fp = open(path+'.txt','w+')
    for sen in root.findall('sentence'):
        uid=sen.get('id')
        text=sen.find('text').text
        #text=parseStopWords(str(sen.find('text').text))
        categories = sen.find('aspectCategories')
        cate=[str(category.get('category')) for category in categories.findall('aspectCategory')]
        fp.write(str(uid)+'//'+str(cate).replace('[','').replace(']',' ').replace(',',' ')\
                 +'//'+text+'\n')
    fp.flush()
    fp.close()
   
def parseStopWords(sentence):
    words=wt(sentence)
    english=st.words(english)
    ss=''
    for word in words:
        if(word.lower() not in english):
            ss=word+' '
    return ss

parseTrain()
parseTestA()
parseTestB()
        

    
