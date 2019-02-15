@Authors: Ebnou Moustapha, Jedad Hamza, Lahlali Kenza, Maghraoui Moncef

**Loss Derivation**

Derivative_of_sigmoid(z) = sigmoid(-z)*sigmoid(z)

Loss_function = sum_{(w,c) \in D} log(sigmoid(v_c*v_w) + sum_{(w,c) \in D'} log(sigmoid(-v_c*v_w)

Derivative_of_loss w.r.t w' = sum_{(w,c) \in D, w=w'} v_c*sigmoid(-v_c*v_w) - sum_{(w,c) \in D', w=w'} v_c*sigmoid(v_c*v_w)


**Code Choices**


- Network Architecture: 1 hidden layer
- No backpropagation





**What didn't work**

