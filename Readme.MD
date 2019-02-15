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




**What didn't work**

- stopwords: We kept the stopwords because we found that they are important and give better results.

