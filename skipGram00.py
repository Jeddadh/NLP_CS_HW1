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
    with open(path,encoding="utf8") as f:
        for l in f:
            sentences.append(preprocess_sentence(l))
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

# def sigmoid(z,limit_sup=12):
#     if -limit_sup > z  :
#         return 0
#     if z > limit_sup  :
#         return 1
#     return 1/(1+np.exp(-z))

def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        y = exp(-x)
        return 1 / (1 + y)
    else:
        y = exp(x)
        return y / (1 + y)

def derivate_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def safe_softplus(x, limit=30):
#     if x > limit:
#         return x
#     return np.log(1.0 + np.exp(x))
    return log(1+exp(-abs(x))) + max(x,0)

def safe_exp_exp(x,limit_inf = -40,limit_sup = 7):
    # if x < limit_inf :
    #     return 1e-20
    if x > limit_sup :
        return 1
    return np.exp(x)/(1+np.exp(x))

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'a']
class SkipGram():
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.W_embedding = None
        self.W_context = {}
        self.W_embedding_gradient = {}
        self.W_context_gradients = {}
        self.contexts_indices = [i for i in range(-(self.winSize//2),int(self.winSize/2 - 0.5) + 1) if i != 0]
        self.set_contexts = {}
        self.set_negative_samples = {}
        self.vocab = {}
        self.vocab_size = 0
        self.nb_all_words = 0
        self.vocab_index = {}
        self.__create_vocabulary()
        self.__initialize_embeddings()
        # self.__create_negative_samples()

    def __create_vocabulary(self):
        print("creating vocab")
        self.vocab = {}
        self.vocab_index = {}
        for sentence in self.sentences :
            for word in sentence :
                if word in stopwords :
                    continue
                self.vocab[word] = self.vocab.get(word,0) + 1

        # eliminte words with low frequency
        print("eliminating words")
        word_to_be_eliminated = []
        index = 0
        self.nb_all_words = 0
        self.nnnnn = 0
        for word in self.vocab :
            if self.vocab[word] <= self.minCount :
                word_to_be_eliminated.append(word)
            else :
                self.nnnnn += self.vocab[word]
                self.vocab[word] = self.vocab[word]**0.75 #for the unigram distribution computation
                self.nb_all_words += self.vocab[word]

                self.vocab_index[word] = index
                index += 1
        for word in word_to_be_eliminated :
            self.vocab.pop(word)
        for word in self.vocab :
            self.vocab[word] /= self.nb_all_words
        self.vocab_size = len(self.vocab)
        self.set_contexts = {i:set() for i in self.vocab_index.values()}

        # create contexts
        print("creating context")
        proba_list = list(self.vocab.values())
        nnnnn = len(self.vocab)
        iiiii = 0
        for sentence in self.sentences :
            for i,word in enumerate(sentence) :
                if not i % 10 :
                    print("word {i}/{t}".format(i=int(100*(iiiii/self.nnnnn)),t=100),end="\r")

                if word not in self.vocab_index :
                    continue
                sentence_size = len(sentence)
                iiiii += 1
                for j in self.contexts_indices :
                    if 0 <= i + j < sentence_size :
                        try :
                            self.set_contexts[self.vocab_index[word]].add(self.vocab_index[sentence[i+j]])
                            self.set_negative_samples[self.vocab_index[word]] = self.set_negative_samples.get(self.vocab_index[word],set())
                            # print(np.random.choice(list(self.vocab),size=self.negativeRate,p=proba_list))
                            self.set_negative_samples[self.vocab_index[word]] = self.set_negative_samples[self.vocab_index[word]].union(set(np.random.choice(list(self.vocab_index.values()),size=self.negativeRate,p=proba_list)))
                            self.set_negative_samples[self.vocab_index[word]].remove(word)

                        except KeyError :
                            # the word is not in vocab_index
                            pass
        return None

    def __initialize_embeddings(self):
        print("initializing embeddings")
        np.random.seed(10)
        self.W_embedding = {}
        self.W_context = {}
        for word in self.vocab_index.values() :
            self.W_embedding[word] = np.random.rand(self.nEmbed).astype('float64') - 0.5
            self.W_context[word] = np.random.rand(self.nEmbed).astype('float64') - 0.5
    def __initialize_gradients(self):
        print("initializing gradients")
        self.W_embedding_gradient = {}
        self.W_context_gradients = {}
        for word in self.vocab_index.values() :
            self.W_embedding_gradient[word] = np.zeros(self.nEmbed).astype('float64')
            self.W_context_gradients[word] = np.zeros(self.nEmbed).astype('float64')


    # def __create_negative_samples(self):
    #     print("creating negative samples")
    #     self.set_negative_samples = {i:set() for i in self.vocab_index.values()}
    #     # return None
    #     for word in self.set_contexts:
    #         nb_negative_samples = 0
    #         possible_words = set(self.vocab_index.values()).difference(self.set_contexts[word])
    #         n_desired_neg_samples = self.negativeRate*len(self.set_contexts[word])
    #         nb_possible_words = len(possible_words)
    #         try :
    #             possible_words.remove(word)
    #         except KeyError:
    #             pass
    #         self.set_negative_samples[word] = set(np.random.choice(list(possible_words),size=min(n_desired_neg_samples,nb_possible_words)))

    def __loss(self):
        loss = 0
        for word in self.set_contexts:
            for context_word in self.set_contexts[word]:
                loss += safe_softplus(-np.dot(self.W_embedding[word],self.W_context[context_word]))
        for word in self.set_negative_samples:
            for context_word in self.set_negative_samples[word]:
                loss += safe_softplus(np.dot(self.W_embedding[word],self.W_context[context_word]))
        return loss

    def __update_params(self, stepsize,random=True,batch_percentage=0.5):
        self.__initialize_gradients()
        # calculate the gradient
        for word in self.set_contexts:
            if random :
                contexts_size = int(batch_percentage*len(self.set_contexts[word]))
                negative_contexts_size = contexts_size = int(batch_percentage*len(self.set_negative_samples[word]))
                contexts = np.random.choice(list(self.set_contexts[word]),contexts_size)
                # print(self.set_negative_samples[word])
                negative_contexts = np.random.choice(list(self.set_negative_samples[word]),negative_contexts_size)
            else :
                contexts = self.set_contexts[word]
                negative_contexts = self.set_negative_samples[word]
            for context_word in contexts:
                # Sum on D
                dot_product = np.dot(self.W_embedding[word],self.W_context[context_word])
                neg_exp_exp = sigmoid(-dot_product)
                self.W_embedding_gradient[word] = self.W_embedding_gradient[word] - neg_exp_exp*self.W_context[context_word]
                self.W_context_gradients[context_word] = self.W_context_gradients[context_word] - neg_exp_exp*self.W_embedding[word]
            for context_word in negative_contexts:
                # Sum on D'
                dot_product = np.dot(self.W_embedding[word],self.W_context[context_word])
                pos_exp_exp = sigmoid(dot_product)
                self.W_embedding_gradient[word] = self.W_embedding_gradient[word] + pos_exp_exp*self.W_context[context_word]
                self.W_context_gradients[context_word] = self.W_context_gradients[context_word] + pos_exp_exp*self.W_embedding[word]
        # update the params
        for word in self.vocab_index.values() :
            self.W_embedding[word] = self.W_embedding[word] - stepsize * self.W_embedding_gradient[word]
            self.W_context[word] = self.W_context[word] - stepsize*self.W_context_gradients[word]



    def train(self, stepsize, epochs):
        for i in range(epochs):
            print(i)
            if True :
                print("{i} : {loss}".format(i=i,loss=self.__loss()))
            self.__update_params(stepsize)


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
    path =     "C:/Users/Hamza/Centrale3A/MscAI_2018_2019/NLP/DM1/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050"
    # path = "C:/Users/Hamza/Centrale3A/MscAI_2018_2019/NLP/DM1/my_data.txt"
    print("preparing sentences")
    sentences = text2sentences(path)
    print("creating model")
    sg = SkipGram(sentences,nEmbed=100, negativeRate=1, winSize = 5, minCount = 5)
    stepsize, epochs = 0.01, 1000
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
