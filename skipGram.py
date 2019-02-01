from __future__ import division
import argparse
import pandas as pd
import re

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['Elmoustapha EBNOU','Hamza JEDDAD','Kenza LAHLALI','Moncef MAGHRAOUI']
__emails__  = ['stef','hamza.jeddad@student.ecp.fr','kenza','Moncef']

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
    with open(path) as f:
        for l in f:
            sentences.append(preprocess_sentence(l))
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivate_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


class SkipGram():
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.W_embedding = None
        self.W_context = [{}]*(self.winSize - 1 )
        self.contexts_indices = [i for i in range(-self.winSize//2,int(self.winSize/2 - 0.5) + 1) if i != 0]
        self.set_contexts = [set()]*(self.winSize - 1 )
        self.vocab = {}
        self.vocab_size = 0
        self.__create_vocabulary()


    def __create_vocabulary(self):
        self.vocab = {}
        self.set_contexts = [set()]*(self.winSize - 1 )
        for sentence in self.sentences :
            for word in sentence :
                self.vocab[word] = self.vocab.get(word,0) + 1

        # eliminte words with low frequency
        for word in self.vocab :
            if self.vocab[word] <= minCount :
                self.vocab.pop(word)

        # create contexts
        for sentence in self.sentences :
            for i,word in enumerate(sentence) :
                if word not in self.vocab :
                    continue
                sentence_size = len(sentence)
                for j in self.contexts_indices :
                    if 0 <= i + j < sentence_size :
                        if sentence[i+j] not in self.vocab :
                            continue
                        self.set_contexts[j].add((word,sentence[i+j]))
        return None

    def __initialize_embeddings(self):
        self.W_embedding = {}
        for word in self.vocab :
            self.W_embedding[word] = np.random.rand(self.nEmbed)
            for
            self.W


    def train(self, stepsize, epochs):
        W_context = np.zeros((self.nEmbed,))
        for i in range(epochs):
            pass

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    def get_vocabulary_and_contexts(self):


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
