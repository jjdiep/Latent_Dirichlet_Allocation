# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:12:38 2018

@author: elakkis
"""
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords



def preprocess_document(text):
    """
    preprocesses the document
    returns a list of words in the document
    """
    #encode and decode to get rid of weird characters
    #text = text.encode('ascii', errors='ignore').decode('ascii')
    
    #remove dashes and escape sequences
    text = text.replace('-', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('.', ' ')   
    text = text.replace(',', ' ') 
    text = text.replace("´s", ' ')
    #remove numbers
    text = re.sub(r"\d+", r"", text)

    #to lowercase
    text = text.lower()
    
    #split in words
    words = text.split()
    #remove stopwords

    stop_words = set(stopwords.words("english"))
    result = [w for w in words if not w in stop_words]
    result = [w for w in result if "www" not in w]
    result = [w for w in result if "http" not in w]
    result = [w for w in result if "src=" not in w]
    result = [w for w in result if "style=" not in w]
    result = [w for w in result if "class=" not in w]
    
    #remove remaining punctuation
    regex = re.compile('[!#%$\"&)\'(+*-/.;:=<?>@\[\]_^`\{\}|~"\\\\"]')
    result = [regex.sub('', w) for w in result]
    return result


def preprocess(filepath):
    """
    Reads the file with news article located at filepath
    Returns: list of documents and a vocabulary for those words
    """
	#read in the headlines
    data = pd.read_csv(filepath, encoding = "ISO-8859-1")

    #reshuffle
    #data = data.sample(frac=1)
    #clean up a bit
    data["Article"] = data["Article"].str.replace("strong>"," ")
    data["Article"] = data["Article"].str.replace("</strong"," ")
    data["Article"] = data["Article"].str.replace("\n", " ")
    data["Article"] = data["Article"].str.replace("\x91"," ")
    data["Article"] = data["Article"].str.replace("\x92"," ")
    data["Article"] = data["Article"].str.replace("\x93"," ")
    data["Article"] = data["Article"].str.replace("\x94"," ")

    def cleaning(s):
        s = str(s)
        x = re.sub(r"<.*?>", " ", s)
        #x = re.sub(r"\"https://.*?\"", " ", x)
        return x

    data["Article"] = data["Article"].apply(cleaning)
    documents = data["Article"].values
        


    #download stopwords from nltk
    nltk.download('stopwords')

        
    #preprocess each document
    prepr_docs = [preprocess_document(d) for d in documents]
    M = len(prepr_docs)


    #now get a word count for each word

    word_counts = {}


    for i in range(M):
        d = prepr_docs[i]
        N = len(d)
        for j in range(N):
            word = d[j]
            if word not in word_counts.keys():
                word_counts[word] = 1
            else:
                word_counts[word] += 1

    #create a vocabulary:mapping from word --> index of that word
    df = pd.DataFrame.from_dict(word_counts, orient = 'index').reset_index()
    df.columns = ["word", "count"]
    #frequency of the word in all documents
    df["freq"] = df["count"]/M
    df["word_length"] = df["word"].apply(lambda x: len(x))


    #remove words that are shorter than 3 characters
    df = df[df["word_length"] >= 3]
    #keep only those that appear frequently, but not too much
    #that way we eliminate common words like "said"
    #those do not contribute to the meaning much
    vocab_df = df[(df["count"] > 2) & (df["freq"] <= 0.75)]
    vocab_df = vocab_df.reset_index().drop(["count", "index", "freq", "word_length"], axis = 1).reset_index().set_index("word")
    vocab = vocab_df.to_dict()['index']

    #finally, we construct a list of document
    #each document - an array of indexes of the words in that document
    docs = []
    for i in range(M):
        d = prepr_docs[i]
        N_d = len(d)
        result = []
        
        for j in range(N_d):
            word = d[j]
            if word in vocab.keys():
                result.append(vocab[word])
        docs.append(np.array(result))

    #return labels as well
    y = (data["NewsType"] == "business")*1
    return docs, y.values, vocab_df
