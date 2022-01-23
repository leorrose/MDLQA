# Deep Learning

### ***Q1. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually reduce the bias***

- [x] True
- [ ] False

***Explanation*** - The more neurons in the hidden layer the more we can fit the network to our data and by that reduce the bias. The bias reduces as the model complexity grows.

### ***Q2. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually increase accuracy on the test group.***

- [ ] True
- [x] False

***Explanation*** - The more neurons in the hidden layer the more we can fit the network to our data and lead to overfitting  which will reduce test accuracy.

### ***Q3. True or False: each multilayer neural network can be described using a three-layer network.***

- [x] True
- [ ] False

***Explanation*** - Universal Approximation Theorem.

### ***Q4. True or False: in the Associative Auto neural network, the original information can always be restored without errors.***

- [ ] True
- [x] False

***Explanation*** - Associative Auto neural network cannot guarantee complete restoration always.

### ***Q5. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually increase accuracy on the train group.***

- [x] True
- [ ] False

***Explanation*** - The more neurons in the hidden layer the more we can fit the network to our data and lead to overfitting  which will increase train accuracy.

### ***Q6. True or False: in the Associative Auto neural network with 4 input nodes and two nodes in the hidden layer, the original information can always be restored without errors.***

- [ ] True
- [x] False

***Explanation*** - Associative Auto neural network cannot guarantee complete restoration always.

### ***Q7. True or False: the RNN model allows, among other things, to study a model that receives as a sequence input of values and emits a single value***

- [x] True
- [ ] False

***Explanation*** - Recurrent Neural Networks(RNN) are a type of Neural Network where the output from the previous step is fed as input to the current step. RNN's are mainly used for, Sequence Classification — Sentiment Classification & Video Classification. Output of an rnn can be a single value (single-output-rnn).

### ***Q8. True or False: the rule of thumb that the number of neurons in the middle layer is equal to the algebraic mean of the number of neurons in the input layer and output layer do not match the AutoEncoder model***

- [x] True
- [ ] False

***Explanation*** - In the AutoEncoder model number of neurons in the input and output layers are the same so an algebraic mean will be the same so all the layers will have the same dimensions which will not allow dimension reduction and learning of the input representation.

### ***Q9. True or False: in order to build a Stacked AutoEncoder model all observations must include the value of the target attribute.***

- [ ] True
- [x] False

***Explanation*** - The input and output of the a Stacked AutoEncoder is the observation itself so no need of the target attribute.

### ***Q10. True or False: the CNN model is for image classification only.***

- [ ] True
- [x] False

***Explanation*** - You can apply CNN on any kind of data if you define the dimension of the data in a way that CNN can understand.

### ***Q11. True or False: in the steepest descent method the search direction is always a descent direction.***

- [x] True
- [ ] False

***Explanation*** - Steepest descent involves looking at the steepness of the hill at their current position, then proceeding in the direction with the steepest descent(i.e., downhill).

### ***Q12. True or False: the more we increase the number of ANN neurons, the greater the chance of overfitting***

- [x] True
- [ ] False

***Explanation*** - The more neurons we have the more complex model with too many parameters we get. and this can lead to overfitting to the training set.

### ***Q13. True or False: the output of neuron in NN is minus 0.5 then the activation is certainly not sigmoid.***

- [x] True
- [ ] False

***Explanation*** - Sigmoid activation output is between 0 to 1.

### ***Q14. True or False: in a neural network, when SGD is used in a single epoch, it passes through random sections of data***

- [ ] True
- [x] False

***Explanation*** - In SGD an epoch would be the full presentation of the training data, and then there would be N weight updates per epoch (if there are N data examples in the training set).

### ***Q15. True or False: LeNet's CNN is usually more accurate than a standard deep network for digit recognition.***

- [x] True
- [ ] False

***Explanation*** - A plain deep NN on the MNIST dataset can get 98.40% test accuracy, Using CNN gets to 99.25% test accuracy.

### ***Q16. True or False: the Tanh activation function does not suffer from the Vanishing Gradient problem.***

- [ ] True
- [x] False

***Explanation*** - Tanh is a sigmoidal activation function that suffers from vanishing gradient problem.

### ***Q17. True or False: in a CNN, the number of weights to learn in the Max Polling layer depends on the size of the image Given a color image***

- [ ] True
- [x] False

***Explanation*** - There are no trainable parameters in a max-pooling layer. In the forward pass, it pass maximum value within each rectangle to the next layer. In the backward pass, it propagate error in the next layer to the place where the max value is taken, because that's where the error comes from.

### ***Q18. Given a color image (in RGB format) in size 8 × 8 we apply 5 2 × 2 filters with pad = 0 and stride = 1. what is the Number of parameters to be trained?***

***Explanation*** - (2\*2\*3 + 1) * 5 = 65.

### ***Q19. Given a color image (in RGB format) in size 8 × 8 we apply 5 2 × 2 filters with pad = 0 and stride = 1. what is the output size obtained?***

***Explanation*** - output shape is (8+0-2)/1 + 1  = 7 so 7\*7\*5 (7,7,5).

### ***Q20. Running Max Polling on output shape (7,7,5) with a filter of size 2 × 2 with stride = 1 what is the size of the new output?***

***Explanation*** - output shape is (7+0-2)/1 + 1  = 6 so 6*6*5 (6,6,5).

### ***Q21. True or False: for every x less than 0 the result of the RELU function will return a value smaller than sigmoid***

- [x] True
- [ ] False

***Explanation*** - If x<0 in RELU the output is 0 but for sigmoid the output is between (0,0.5] .

### ***Q22. True or False: the use of the chain rule is designed to allow the gradient to "roll back" even in deep networks.***

- [x] True
- [ ] False

***Explanation*** - The chain rule allows us to find the derivative of composite functions. It is computed extensively by the back propagation algorithm.

### ***Q23. True or False: TanH result has the same mark (minus/plus) as the input mark (after scheme)***

- [x] True
- [ ] False

***Explanation*** - TanH maps negative inputs to negative and positive to positive.

### ***Q24. True or False: by the “no free lunch” (NFL) theorem, no variation of the Gradient algorithm (e.g., Adam, SGD, AdaGrad etc) is always better than all the other variations***

- [x] True
- [ ] False

***Explanation*** - The “no free lunch” (NFL) theorem implies that no single machine learning algorithm is universally the best-performing algorithm for all problems.

### ***Q25. True or False: the drop out in a neural network allows for an ensemble approximation of neural networks Without the need to train each network for multiple iterations.***

- [x] True
- [ ] False

***Explanation*** - Dropout in a neural network can be considered as an ensemble technique, where multiple sub-networks are trained together by “dropping” out certain connections between neurons.

### ***Q26. True or False: increasing the number of neurons in the middle layer of a three-layer neural network will usually increase the variance error component.***

- [x] True
- [ ] False

***Explanation*** - The more neurons in the hidden layer the more we can fit the network to our data and lead to higher variance. In general, more complicated models will result in larger variance.

### ***Q27. True or False: the AdaBoost algorithm reduces both the bias component and the variance component.***

- [x] True
- [ ] False

***Explanation*** - Boosting is a meta-learning algorithm that reduces both bias and variance.

### ***Q28. True or False: the bagging algorithm reduces both the bias component and the variance component.***

- [ ] True
- [x] False

***Explanation*** - Bagging reduces the variance without reducing the bias.
