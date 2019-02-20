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
- Problem with overflow for sigmoid:
In the beginning we used a sigmoid function that returns 0 if -limit_sup > z but we realized that this function is not optimal that's why we used the trick of using exp(z)/(1+exp(z)) if z<0.
- Problem with overflow with softplus: Whenever x gets too large the result is inf, we used the trick log(1+exp(x)) = log(1+exp(x)) - log(exp(x)) + x = log(1+exp(-x)) + x.






**What didn't work**

- stopwords: We kept the stopwords because we found that they are important and give better results.

