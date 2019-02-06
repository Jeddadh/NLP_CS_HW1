#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import pandas as pd
from collections import Counter

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['ebnou','lahlali','jeddad','maghraoui']
__emails__  = ['elmoustapha.ebnou@student.ecp.fr','kenza.lahlali@student.ecp.fr','hamza.jeddad@student.ecp.fr','moncef.maghraoui@student.ecp.fr']




def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
           # sentences.append([word.lower() for word in l.split()\
            #    if word.isalnum() and len(word)>1])
            sentences.append( l.lower().split() )
            
    return sentences

def build_vocab(sentences):
    counter = Counter()
    vocab = []
    
    for sentence in sentences :
        
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
        counter.update(sentence)

    
    vocabulary = {word: (index, counter[word]) for index, word in enumerate(vocab)}
    idx2word = {idx : w for (idx,w) in enumerate(vocab)}
    return vocabulary, counter, idx2word



def build_samples(vocab,idx2word, sentences, sample_file='./samples.csv', context_window_size=5,K=5):
    """
    vocab : vocabulary 
    counter : words frequency used to calculate sampling probability
    
    """
    tokens = list()
    samples = list()
    labels = list()
    
    for sentence in sentences:
        for word in sentence:
            """
            discard_prob = 1 - np.sqrt(threshold/counter[word]) #from the paper
            discard = np.random.binomial(1, discard_prob)
            
            if discard == 0 :
                tokens.append(vocab.get(word,(None,0))[0]) #get the index
            """
            tokens.append(vocab.get(word,(None,0))[0]) #get the index
        
    for idx, token in enumerate(tokens[:-1]):
        if token is None: continue
        
        s = max(0, idx - context_window_size)
        t = min(idx + context_window_size , len(tokens)-1)
        
        context = tokens[s:idx] + tokens[idx+1:t+1]
        
        while None in context:
            context.remove(None)
            
        if not context : continue
    
        #pair (context , target) positive 
        try:
            samples.append((idx2word[token],idx2word[tokens[idx+1]])) #next ord as target
            labels.append(1)
        except:
            print("idx :",idx,"len(context) = ",len(context))
        for i in range(K):
            idx_ = np.random.randint(0,len(tokens)-1)
            samples.append((idx2word[token],idx2word[tokens[idx_]]))
            labels.append(0)
            
            
    return samples , labels            
            


def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
                           
        #Mapping between words and indexes 

        
        self.vocab = build_vocab(self.sentences)[0]
        self.idx2word = build_vocab(self.sentences)[2]
        
    

    def train(self,stepsize, epochs):
        
        embedding_dim = self.nEmbed
        #initialize embedding matrix 
        # 2 embeddings for each word (as a context and as a word)
        
        E1 = np.random.randn(embedding_dim,vocab_size) 
        E2 = np.random.randn(embedding_dim,vocab_size) 
        
        samples,labels = build_samples(vocab,idx2word, self.sentences)
        
        for epoch in range(epochs):
            
            
        

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if (word1 in self.sentences) and (word2 in self.sentences):
            
            

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)
