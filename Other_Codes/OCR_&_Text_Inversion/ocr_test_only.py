import cv2
import easyocr
#import text2sentence as t2s
import re
import string
import math
import difflib
import json
import os
import pandas as pd

def load_cheese_text():
    cheese_text_directory = './cheese_text.json'
    f2 = open(cheese_text_directory, 'r')
    cheese_text = json.load(f2)
    f2.close()
    return cheese_text
    
def cal_sim(s1,s2):
    result = difflib.SequenceMatcher(None,s1,s2).quick_ratio()
    return result

def clean_text(sentence):
    sentence = re.sub("[%s]" % re.escape(string.punctuation), "", sentence.lower())    
    sentence = sentence.replace('\n', ' ').replace('.', ' ').replace(',', ' ').replace('?', ' ')\
    .replace('\r', ' ').replace('!', ' ').replace('"\r', ' ').replace('"', ' ')\
    .replace("'", ' ').replace("''", ' ').replace('(', ' ').replace(')', ' ').replace(']', ' ')\
    .replace('-', ' ').replace('/', ' ')
    
    while ('  ' in sentence):
        sentence = sentence.replace('  ', ' ')
    return sentence

def detect_text(path):
    img = cv2.imread(path)
    reader = easyocr.Reader(['en'], gpu=True)
    text_ = reader.readtext(img)
    text_array = []
    for t in text_:
        loc,text,score = t
        text_array.append(clean_text(text))
    if(len(text_array) == 0):
        return 'no-text'
    else : 
        return text_array
    
def split_sentences(full_text):
    sents = re.split(' ', full_text)
    sents = [sent for sent in sents if len(sent) > 0]  # 去除只包含\n或空白符的句子
    return sents

def load_cheese_name():
    cheese_list_directory = './list_of_cheese.txt'
    f = open(cheese_list_directory)
    cheese_list = []
    for name in f:
       cheese_list.append(name.replace('\n',''))
    return cheese_list

def text_classification(path):
    
    cheese_list = load_cheese_name()
    cheese_text = load_cheese_text()
    text_array_ = detect_text(path)
    if(text_array_ == 'no-text'):return 'no-text'
    text_array = []
    for text in text_array_:
        sent = split_sentences(text)
        for word in sent:
            if(len(word)>=4):
                text_array.append(word)
    if(len(text_array)==0):return 'no-text'
    print(text_array)
    dict_score = {}
    for name in cheese_list:
        name_text = cheese_text[name]
        score_one_cheese = [max([cal_sim(s1,clean_text(s2)) for s1 in text_array]) for s2 in name_text]
        dict_score[name] = max(score_one_cheese)
    print(dict_score)
    max_score = 0
    result = ''
    for name in dict_score:
        if(dict_score[name]>max_score):
            max_score = dict_score[name]
            result = name
    #for special case:
    if(dict_score["TÊTE DE MOINES"] >= 0.92):
        return'TÊTE DE MOINES'
    if(dict_score["B\u00dbCHETTE DE CH\u00c8VRE"] >= 0.90):
        result = "B\u00dbCHETTE DE CH\u00c8VRE"
        return result
    if(dict_score["MIMOLETTE"] >= 0.87):
        return'MIMOLETTE'
    if(dict_score["EMMENTAL"] >= 0.90):
        return'EMMENTAL'
    if(dict_score["MOTHAIS"] >= 0.90):
        return'MOTHAIS'
    if(dict_score["EMMENTAL"]>=0.90):
        return "EMMENTAL"
    if(dict_score["PARMESAN"]>=0.90):
        return "PARMESAN"
    if(dict_score["FETA"]>=0.90):
        return "FETA"   
    if(dict_score["CABECOU"]>=0.90):
        return "CABECOU"
    if(dict_score["CAMEMBERT"]>=0.90):
        return "CAMEMBERT"
    if(dict_score["STILTON"]>=0.90):
        return "STILTON"
    if(dict_score["CHABICHOU"]>=0.90):
        return "CHABICHOU"
    if(dict_score["FROMAGE FRAIS"]>=0.90):
        return "FROMAGE FRAIS"
    if(max_score>=0.90):
        return result
    else:
        return 'no-text'

def decision(result_classifier,result_text):
    if(result_text == 'no-text'):
        return result_classifier
    if(result_classifier == 'SAINT-NECTAIRE' or result_classifier == 'CHABICHOU'):
        return result_classifier
    return result_text

def main():
    df = pd.read_csv('./submission_mixed.csv')
    dict = {'id':[],'label':[]}
    for i in range(len(df)):
        print(i)
        path = './dataset/test/'+ df['id'][i]+'.jpg'
        result = text_classification(path)
        dict['id'].append(df['id'][i])
        dict['label'].append(result)
        print(result)
        df_dict = pd.DataFrame(dict)
        df_dict.to_csv('test_ocr_result_real.csv',index = False)
    df_dict = pd.DataFrame(dict)
    df_dict.to_csv('test_ocr_result_real.csv',index = False)
                   
#main()
print(text_classification('dataset/test/j1OFUhDpUOGeRLn.jpg'))