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
In the beginning we used the sigmoid code:
def sigmoid(z,limit_sup=12):
    if -limit_sup > z  :
        return 0
    if z > limit_sup  :
        return 1
    return 1/(1+np.exp(-z))
 But we realized that it is not optimal for us, because we don't distinguish z below the limit, that's why we implemented: 
 def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = exp(x)
        return z / (1 + z)






**What didn't work**

- stopwords: We kept the stopwords because we found that they are important and give better results.

