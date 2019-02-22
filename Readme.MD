@Authors: Ebnou Elmoustapha, Jeddad Hamza, Lahlali Kenza, Maghraoui Moncef

For our implementation we relied on Goldbery & Levy paper : **word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method**.

- **Loss and loss derivation** : See the attached report to check the loss function and its derivative computation

- **preprocessing steps** : 
  - replace punctuation by spaces
  - replace numerical characters with spaces
  - replace consecutive spaces with one space
  - remove space from the first and the last element of the string
  - split the sentence by space
  
- **getting vocabulary and positive contexts**
  - We explored all the file sentences, and for every word store its occurence in a dictionnary
  - We associated to each word an index
  - We associated to each word a proba (according to the unigram distribution)
  - **_Creating contexts_**
  - for every sentence :
    - for every word in the sentence :
      - w_context : list of word's contexts (according to the window size)
      - add the couple [word,w_context]

- **training**
  - _Initialization_ : We initialize the vectors with standard gaussian distribution
  - Parameters update :
    - for every sentence :
      - for every word :
        - sample N words from the vocabulary according to the unigram distribution (N = Negative Rate * # of words in the context) wi
        - update the word and  vector with v_w = v_w - stepsize*(dJ/dv_w)

Important choices of code:
- Network Architecture: 1 hidden layer
- No backpropagation
- Problem with overflow for sigmoid corrected.
- Problem with overflow with softplus corrected.
- stopwords: We kept the stopwords because we found that they are important and give better results. After using NLTK package and moving stopwords, we realized that it's not a good idea to keep these package as they delete some key information about the context. Therefore, we decided to keep all the stopwords.

**What could be done to improve our code**

- We could've vectorized the loss computation and the gradient which could have alowed us to avoid the "for" loops and gain time.
- We could have used an adaptatif stepsize.

**Save and load the model**
The model can be saved using pickle module
