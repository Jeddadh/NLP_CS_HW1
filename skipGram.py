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
        self.set_contexts = [set()]*(self.winSize - 1)
        self.set_negative_samples = [set()]*(self.winSize - 1)
        self.vocab = {}
        self.vocab_size = 0
        self.__create_vocabulary()
        self.__initialize_embeddings()


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
        self.vocab_size = len(self.vocab)

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
            for i in range(len(self.set_contexts)) :
                self.W_context[i][word] = np.random.rand(self.nEmbed)

    def __create_negative_samples(self):
        self.set_negative_samples = [set()]*(self.winSize - 1)
        for word in self.vocab :
            for j in range(self.winSize - 1):
                nb_negative_samples = 0
                while nb_negative_samples < self.negativeRate :
                    possible_words = list(self.vocab.keys())
                    possible_words.remove(word)
                    context_word = np.random.choice(possible_words)
                    if (word,context_word) in self.set_contexts[j] :
                        continue
                    self.set_negative_samples[j].add((word,context_word))
                    nb_negative_samples += 1
    def __loss(self):
        loss = 0
        for j in range(self.winSize -1) :
            for (word,j_context_word) in self.set_contexts[j]:
                loss -= np.log(sigmoid(np.dot(self.W_embedding[word],self.W_context[j][context_word])))
            for (word,j_context_word) in self.set_negative_samples[j]:
                loss -= np.log(sigmoid(-np.dot(self.W_embedding[word],self.W_context[j][context_word])))

    def __update_params(self,stepsize):
        for j in range(self.winSize-1):
            for (word,context_word) in self.set_contexts[j]:
                self.W_embedding[word] = self.W_embedding[word] - stepsize*(self.W_context[j][context_word]*sigmoid(-self.W_embedding[word]*self.W_context[j][context_word]))
            for (word,j_context_word) in self.set_negative_samples[j]:
                self.W_embedding[word] = self.W_embedding[word] + stepsize*(self.W_context[j][context_word]*sigmoid(self.W_embedding[word]*self.W_context[j][context_word]))

    def train(self, stepsize, epochs):
        for i in range(epochs):
            loss = self.__loss()
            print(loss)
            self.__update_params(stepsize)


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



    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)')
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        stepsize, epochs = 0.1, 100
        sg.train(stepsize, epochs)
        # sg.save(opts.model)
    # 
    # else:
    #     pairs = loadPairs(opts.text)
    #
    #     sg = mSkipGram.load(opts.model)
    #     for a,b,_ in pairs:
    #         print sg.similarity(a,b)
