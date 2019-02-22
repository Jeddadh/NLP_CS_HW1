from __future__ import division
import argparse
import pandas as pd
import re
from time import time
# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

import pickle

index_names = "word_index.pkl"
index_embedding_name = lambda  epoch : "index_embedding_name_{}.pkl".format(epoch)

__authors__ = ['Elmoustapha EBNOU','Hamza JEDDAD','Kenza LAHLALI','Moncef MAGHRAOUI']
__emails__  = ['elmoustapha.ebnou@student.ecp.fr','hamza.jeddad@student.ecp.fr','kenza.lahlali@student.ecp.fr','moncef.maghraoui@student.ecp.fr']

def preprocess_sentence(sentence):
    """
    """
    sentence = sentence.lower()
    sentence = re.sub("[!”#$%&’'\"()*+,-./:;<=>?@[\]^_`{|}~]"," ",sentence) # replace punctuation by spaces
    sentence = re.sub("[0-9]"," ",sentence) # replace numerical characters with spaces
    sentence = re.sub("\n"," ",sentence)
    sentence = re.sub(" +", " ",sentence) # replace consecutive spaces with one space
    sentence = sentence.strip() # remove space from the first and the last element of the string
    return sentence.split(" ")

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path,encoding="utf8") as f:
        for l in f:
            sentences.append(preprocess_sentence(l))
    return sentences

def sentences_to_indices_and_get_vocab(sentences):
    word_occurence = {}
    word_index = {}
    index_occurence = {}
    sentences_index = []
    for sent in sentences :
        sent_index = []
        for word in sent :
            word_index[word] = word_index.get(word, len(word_index))
            word_occurence[word] = word_occurence.get(word, 0) + 1
            index_occurence[word_index[word]] = word_occurence[word]
            sent_index.append(word_index[word])
        sentences_index.append(sent_index)
    word_proba = {}
    total_occurence_pow = sum([word_occurence[word]**0.75 for word in word_occurence])
    for index in index_occurence :
        word_proba[index] = (index_occurence[index]**0.75) / total_occurence_pow

    return word_occurence, word_index, index_occurence, sentences_index, word_proba

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        y = np.exp(-x)
        return 1 / (1 + y)
    else:
        y = np.exp(x)
        return y / (1 + y)




def derivate_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def safe_softplus(x, limit=30):
    if x > limit:
        return x
    return np.log(1.0 + np.exp(x))

def safe_exp_exp(x,limit_inf = -40,limit_sup = 7):
    # if x < limit_inf :
    #     return 1e-20
    if x > limit_sup :
        return 1
    return np.exp(x)/(1+np.exp(x))

def save_emb(emb,epoch):
    file_name = index_embedding_name(epoch)
    with open(file_name,"wb") as f :
        pickle.dump(emb,f)

class SkipGram():
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.W_embedding = None
        self.C_embedding = None
        self.W_embedding_gradient = None
        self.C_embedding_gradients = None
        self.contexts_indices = list(range(-winSize//2,0))+list(range(1,int(winSize//2)))
        self.word_occurence, self.word_index, self.index_occurence, self.sentences_index, self.word_proba = \
            sentences_to_indices_and_get_vocab(self.sentences)
        self.list_proba = np.array(list(self.word_proba.values()))
        global index_names
        with open(index_names,"wb") as f:
            pickle.dump(self.word_index,f)
        self.list_words = np.array(list(self.word_proba.keys()))
        self.vocab_size = len(self.word_occurence)
        self.__initialize_embeddings()
        self.contexts = None
        self.__create_context()
        self.loss = 0



    def __initialize_embeddings(self):
        print("initializing embeddings")
        np.random.seed(10)
        self.W_embedding = 2*(np.random.normal(size = (self.vocab_size ,self.nEmbed))  - 0.5)
        print("embedding size", self.W_embedding.shape)
        self.C_embedding = 2*(np.random.normal(size = (self.vocab_size ,self.nEmbed))- 0.5)

    def __zero_grad(self):
        print("initializing gradients")
        self.W_embedding_gradient = np.zeros((self.vocab_size, self.nEmbed)).astype('float64')
        self.W_context_gradients = np.zeros((self.vocab_size, self.nEmbed)).astype('float64')

    def __create_context(self):
        self.contexts = []
        for sent in self.sentences_index :
            for i, word in enumerate(sent) :
                i_contexts = []
                for j in self.contexts_indices :
                    try :
                        i_contexts.append(sent[i+j])
                    except IndexError :
                        continue
                self.contexts.append([i,i_contexts])

    def __update_params(self, stepsize,random=True,batch_percentage=0.5):
        self.__zero_grad()
        n_contexts = len(self.contexts)
        for i in range(n_contexts):
            w, w_context = self.contexts[i]
            print("{i} / {n}".format(i=i,n=n_contexts),end="\r")
            context_size = len(w_context)
            W_emb = np.tile(self.W_embedding[w],context_size).reshape((context_size,self.nEmbed))
            C_emb = self.C_embedding[w_context]
            dots = np.dot(C_emb, self.W_embedding[w])
            sig_dots = -sigmoid(-dots).reshape((context_size,1))
            self.W_embedding[w] = self.W_embedding[w] - stepsize * \
                np.multiply(sig_dots,C_emb).sum(axis=0)
            # print(sig_dots)
            self.C_embedding[w_context] = self.C_embedding[w_context] - stepsize * \
                np.multiply(sig_dots,W_emb)
            self.loss -= np.log(sigmoid(dots)).sum()


            negative_context_size = self.negativeRate*context_size
            W_emb_negs = np.tile(self.W_embedding[w],negative_context_size).reshape((negative_context_size,self.nEmbed))

            negative_context = np.random.choice(self.list_words, negative_context_size, p=self.list_proba)
            C_emb_neg = self.C_embedding[negative_context]
            neg_dots = np.dot(C_emb_neg, self.W_embedding[w])
            sig_neg_dots = sigmoid(neg_dots).reshape((negative_context_size,1))
            self.W_embedding[w] = self.W_embedding[w] - stepsize * \
                np.multiply(sig_neg_dots,C_emb_neg).sum(axis=0)
            self.C_embedding[negative_context] = self.C_embedding[negative_context] - stepsize * \
                np.multiply(sig_neg_dots,C_emb_neg)
            self.loss -= np.log(sigmoid(-neg_dots)).sum()


    def train(self, stepsize, epochs):
        for i in range(epochs):
            time0 = time()
            self.loss = 0
            print("epoch ",i)
            self.__update_params(stepsize)
            if True :
                print("  loss : {loss}".format(loss=self.loss))
            print("  time {} s".format(time()-time0))
            save_emb(self.W_embedding,i)


    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
            We choose the null vector as the one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)

        """

        if word1 not in self.vocab_index :
            return 0

        if  word2 not in self.vocab_index:
            return 0

        emb1 = self.W_embedding[self.vocab_index[word1]]
        emb2 = self.W_embedding[self.vocab_index[word2]]
        return 0.5+0.5*np.dot(emb1,emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))



    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--text', help='path containing training data', required=True)
    # parser.add_argument('--model', help='path to store/read model (when training/testing)')
    # parser.add_argument('--test', help='enters test mode', action='store_true')
    #
    # opts = parser.parse_args()

    # if not opts.test:

    path = "C:/Users/Hamza/Centrale3A/MscAI_2018_2019/NLP/DM1/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
    # path =     "C:/Users/Hamza/Centrale3A/MscAI_2018_2019/NLP/DM1/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050"
    # path = "C:/Users/Hamza/Centrale3A/MscAI_2018_2019/NLP/DM1/my_data.txt"
    print("preparing sentences")
    sentences = text2sentences(path)
    print("creating model")
    sg = SkipGram(sentences,nEmbed=100, negativeRate=5, winSize = 5, minCount = 5)
    stepsize, epochs = 0.01, 20
    print("training model")
    sg.train(stepsize, epochs)
    print(sg.W_embedding)
        # sg.save(opts.model)
    #
    # else:
    #     pairs = loadPairs(opts.text)
    #
    #     sg = mSkipGram.load(opts.model)
    #     for a,b,_ in pairs:
    #         print sg.similarity(a,b)
