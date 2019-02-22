@Authors: Ebnou Moustapha, Jedad Hamza, Lahlali Kenza, Maghraoui Moncef

**Loss Derivation**

Derivative_of_sigmoid(z) = sigmoid(-z)*sigmoid(z)

Loss_function = sum_{(w,c) \in D} log(sigmoid(v_c.v_w) + sum_{(w,c) \in D'} log(sigmoid(-v_c.v_w)

Derivative_of_loss w.r.t w' = sum_{(w,c) \in D, w=w'} v_c.sigmoid(-v_c*v_w) - sum_{(w,c) \in D', w=w'} v_c.sigmoid(v_c*v_w)


**Code Choices**

Steps:
- Generating one hot input matrix X
- Generating embedded word vectors for the context v_c
- Computation of loss derivative
- Computation of the probabilities.

Important choices of code:
- Network Architecture: 1 hidden layer
- No backpropagation
- We built a table to get rid of frequent words. Indeed, most frequent words provide less information that less frequent words in the computation of probability.
- Problem with overflow for sigmoid corrected.
- Problem with overflow with softplus corrected.
- stopwords: We kept the stopwords because we found that they are important and give better results. After using NLTK package and moving stopwords, we realized that it's not a good idea to keep these package as they delete some key information about the context. Therefore, we decided to keep all the stopwords.

**What could be done to improve our code**

- We could've vectorized the loss computation and the gradient which could have alowed us to avoid the "for" loops and gain time.
- We could have used an adaptatif stepsize.
