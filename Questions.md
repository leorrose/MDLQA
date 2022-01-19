# ***Questions***

### ***Q1. True or False: in the Bagging algorithm the size of the ensemble built is always equal to the number of iterations (models) determined by the user.***

- [x] True
- [ ] False

***Explanation:*** - In the Bagging algorithm the user inputs a base learner, number of iterations and n as input. In each iteration a bootstrap sample of size n is selected and a base learner is trained on this sample. At the end the models are combined.

### ***Q2. True or False: training time of LOOCV (Leave one-out Cross Validation) on a table with 100 examples is at least 10 times longer than the 10-CV Folds training time on this table.***

- [x] True
- [ ] False

***Explanation:*** - LOOCV is a special case of k-Fold Cross-Validation where k is equal to the size of data (n) so LOOCV = 100-cv Folds. In 100-cv Folds we train 100 model and in 10-CV Folds we train 10 models so its at least 10 times longer because in the 100-cv Folds each model is trained on more data than in 10-CV Folds.

### ***Q3. True or False: it is always possible to find an accurate solution for the classifier variables with analytical methods regardless of the type of classifier constructed.***

- [ ] True
- [x] False

***Explanation:*** - We prefer the analytical method in general because it is faster and because the solution is exact. Nevertheless, sometimes we must resort to a numerical method due to limitations of time or hardware capacity. For example when we cannot fit all the data into the memory of a single computer in order to perform the analytical calculation. Also, sometimes the analytical solution is unknown and all we have to work with is the numerical approach.

### ***Q4. True or False: one of the advantages of the Bagging algorithm over the AdaBoost algorithm is that Bagging can be performed in parallel without any change in code, so each classifier is built on a different processor/computer.***

- [x] True
- [ ] False

***Explanation:*** - Bagging is a parallel method that fits different, considered learners independently from each other, making it possible to train them simultaneously.

### ***Q5. True or False: the original SVM algorithm is intended for solving binary problems only.***

- [X] True
- [ ] False

***Explanation:*** - SVMs (linear or otherwise) inherently do binary classification. However, there are various procedures for extending them to multiclass problems.

### ***Q6. True or False: the original AdaBoost algorithm is intended for solving binary problems only.***

- [x] True
- [ ] False

***Explanation:*** - AdaBoost in its original form was designed for binary classification. It has been proved that this classifier is extremely successful in producing accurate classification when applied to two-class classification problems. AdaBoost was also proposed to be used as multi-class classifier by Freund and Schapire.

### ***Q7. True or False: Trying to solve a multiclass classification problem where clases are {a,b,c,d} using ECOC with 3 models. model 1 is intended to distinguish between the group {a,b} and {c,d}. model 2 is intended to distinguish between the group{a,c} and {b,d}. model 3 is intended to distinguish between the group {a,d} and {b,c}. If the models classified a new observation to groups 1,0,1 respectively then the observation should be assigned to group c.***

- [x] True
- [ ] False

***Explanation:*** - {c,d} & {a,c} & {n,c} = {c}.

### ***Q8. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually reduce the bias***

- [x] True
- [ ] False

***Explanation:*** - The more neurons in the hidden layer the more we can fit the network to our data and by that reduce the bias. The bias reduces as the model complexity grows.

### ***Q9. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually increase accuracy on the test group.***

- [ ] True
- [x] False

***Explanation:*** - The more neurons in the hidden layer the more we can fit the network to our data and lead to overfitting  which will reduce test accuracy.

### ***Q10. True or False: bias reduction will inevitably lead to an increase in variance***

- [ ] True
- [x] False

***Explanation:*** - Although there is a bias-variance trade off it is not guarantee that reducing the bias will inevitably lead to an increase in variance.

### ***Q11. True or False: given the function ![eq](1.png). If there is an optimum point then Newton's method will inevitably find the exact solution within one iteration regardless of the starting point Search***

- [ ] True
- [x] False

***Explanation:*** - newton's method has a quadratic convergence but it depends on the step size and starting point so it will not always converge in exact one iteration.

### ***Q12. True or False: the difference between Ensemble and Mixtures of Experts, is that in Ensemble classifiers are created from the same Basic algorithm and in MoE this is usually not the case.***

- [ ] True
- [x] False

***Explanation:*** - In ensemble we use a lot of algorithms and combine them but in MoE we use a small number of models where each model specializes in some aspect.

### ***Q13. True or False: each multilayer neural network can be described using a three-layer network.***

- [x] True
- [ ] False

***Explanation:*** - TODO (universality theorem?)

### ***Q14. True or False: in the Associative Auto neural network, the original information can always be restored without errors.***

- [ ] True
- [x] False

***Explanation:*** - Associative Auto neural network cannot guarantee complete restoration always.

### ***Q15. True or False: in the SVM algorithm, the dual representation is intended to allow the use of Kernel without the explicit knowledge of the conversion function.***

- [x] True
- [ ] False

***Explanation:*** - TODO (without the explicit knowledge of the conversion function? shouldn't we know th function but not apply it but apply on the multiplication only?)

### ***Q16. True or False: based on the "No Free Lunch" principle, no learning algorithm can be found that surpasses all the other algorithms in a given problem.***

- [ ] True
- [x] False

***Explanation:*** - The “no free lunch” (NFL) theorem implies that no single machine learning algorithm is universally the best-performing algorithm for all problems.

### ***Q17. True or False: according to the Occam razor, if two models have the same error on the training set, then the simpler model should be preferred.***

- [x] True
- [ ] False

***Explanation:*** - The Occam’s Razor in applied machine learning (Occam’s Two Razors) states that given two models with the same training-set error, the simpler one should be preferred because it is likely to have lower generalization error.

### ***Q18. True or False: existence of KKT conditions for a non-quadratic problem, represents the necessary condition for a global optimal solution.***

- [ ] True
- [x] False

***Explanation:*** - TODO (The Problem should be convex)

### ***Q19. True or False: increasing the number of neurons in the hidden layer of a three-layer neural network will usually increase accuracy on the train group.***

- [x] True
- [] False

***Explanation:*** - The more neurons in the hidden layer the more we can fit the network to our data and lead to overfitting  which will increase train accuracy.

### ***Q20. True or False: in the Associative Auto neural network with 4 input nodes and two nodes in the hidden layer, the original information can always be restored without errors.***

- [ ] True
- [x] False

***Explanation:*** - Associative Auto neural network cannot guarantee complete restoration always.

### ***Q21. True or False: existence of KKT conditions for a quadratic problem, represents the sufficient condition for a global optimal solution.***

- [x] True
- [ ] False

***Explanation:*** - TODO (The Problem is convex)

### ***Q22. The goal is to classify image x to one of the two classes A,B. The the priori probabilities are P(A) = 0.4, P(B) = 0.6. The probability of obtaining the above x given A is 0.7 Whereas the probability of obtaining the above x given B is 0.3. According to the Naïve Bayess method, the class that must be chosen to minimize the mistake expected value is B.***

- [ ] True
- [x] False

***Explanation:*** - P(A|x) = P(A) \* P(x|A) = 0.28, P(B|x) = P(B) \* P(x|B) = 0.18.

### ***Q23. True or False: in most cases the accuracy of the first classifier built by the AdaBoost Ensemble has a greater accuracy than the accuracy of first classifier built by the Bagging Ensemble.***

- [x] True
- [ ] False

***Explanation:*** - TODO

### ***Q24. The ML team received a file with new training data. The task is to create a new classification model based on this data. The input features are nominal only. Also the target attribute is binary. The team is debating between five possible classifiers: KNN, ID3, Naive Bayes, OneR, C4.5. To select the best classifier, it was decided to linearly weight the output of these classifiers (The probabilities for each CLASS). Which Ensemble method is suitable for this case?***

***Answer & Explanation:*** - 

### ***Q25. Given a training set that includes 100 attributes and 100 observations - which Ensemble method do you recommended to use (only one ensemble method). Explain why.***

***Answer & Explanation:*** - 

### ***Q26. Multi-class problems can be solved by converting to a number of binary classification problems (with binary classifications only). Explain how the conversion is recommended. Explain in which cases this method is required.***

***Answer & Explanation:*** - 

### ***Q27. True or False: in the AdaBoost algorithm the level of accuracy on the test set may continue to improve even when the level the accuracy of the train set reached 100%***

- [x] True
- [ ] False

***Explanation:*** -  This question is answered by Schapire and Freund, et. al. Even when the training error is zero, the margin (= sample distance to decision boundary) is still improved by further boosting iterations. This can lead to improving the test set accuracy even the train set reached 100%.

### ***Q28. True or False: if the AUC of model A is higher than that of model B, then the accuracy of model A is necessarily higher than that Of model B***

- [ ] True
- [x] False

***Explanation:*** - Accuracy is computed at the certain threshold (usually 0.5). While AUC is computed by adding all the "accuracies" computed for all the possible threshold values. ROC can be seen as an average (expected value) of those accuracies when are computed for all threshold values.

### ***Q29. True or False: the Error-Correcting Output Codes approach is designed to enable classification trees to address Multi-Class problems.***

- [x] True
- [ ] False

***Explanation:*** - The Error-Correcting Output Codes method is a technique that allows a multi-class classification problem to be reframed as multiple binary classification problems, allowing the use of native binary classification models to be used directly.

### ***Q30. True or False: for the purpose of thinning a forest of trees, it is advisable to dilute trees with a low correlation with the target feature and / or other trees.***

- [ ] True
- [ ] False

***Explanation:*** - 2017 says yes 2016 says no.

### ***Q31. True or False: the RNN model allows, among other things, to study a model that receives as a sequence input of values and emits a single value***

- [x] True
- [ ] False

***Explanation:*** - Recurrent Neural Networks(RNN) are a type of Neural Network where the output from the previous step is fed as input to the current step. RNN's are mainly used for, Sequence Classification — Sentiment Classification & Video Classification. Output of an rnn can be a single value (single-output-rnn).

### ***Q32. True or False: the rule of thumb that the number of neurons in the middle layer is equal to the algebraic mean of the number of neurons in the input layer and output layer do not match the AutoEncoder model***

- [x] True
- [ ] False

***Explanation:*** - In the AutoEncoder model number of neurons in the input and output layers are the same so an algebraic mean will be the same so all the layers will have the same dimensions which will not allow dimension reduction and learning of the input representation.

### ***Q33. True or False: in order to build a Stacked AutoEncoder model all observations must include the value of the target attribute.***

- [ ] True
- [x] False

***Explanation:*** - The input and output of the a Stacked AutoEncoder is the observation itself so no need of the target attribute.

### ***Q34. True or False: in the SVM model with the Kernel function It is necessary to know the explicit function of translating the observations from the original space to the new space.***

- [ ] True
- [x] False

***Explanation:*** - 

### ***Q35. True or False: the CNN model is for image classification only.***

- [ ] True
- [x] False

***Explanation:*** - You can apply CNN on any kind of data if you define the dimension of the data in a way that CNN can understand.

### ***Q36. True or False: the running time required to study a single tree according to C4.5 algorithm  is greater than the running time required to study a randomForest with a single tree.***

- [x] True
- [ ] False

***Explanation:*** - Each tree in a random forest is not trained on all sample\features meaning if we train only one tree it should take lees than training a single C4.5 which uses all sample\features to builds a tree.

### ***Q37. True or False: in the steepest descent method the search direction is always a descent direction.***

- [x] True
- [ ] False

***Explanation:*** - Steepest descent involves looking at the steepness of the hill at their current position, then proceeding in the direction with the steepest descent(i.e., downhill).

### ***Q38. True or False: Newton's method is good for converging to the minimum region even when the starting point is not close to the point the optimum.***

- [ ] True
- [x] False

***Explanation:*** - If the guess is not close enough it can cause the method to approximate the wrong root or the method will diverge entirely.

### ***Q39. True or False: logistic regression is suitable for solving binary classification problems and multi-class problems.***

- [ ] True
- [x] False

***Explanation:*** - Logistic regression in its original form is intended for binary classification it can be used for multi-class problems but this requires the classification problem first be transformed into multiple binary classification problems.

### ***Q40. True or False: the Shrinkage method makes it possible to avoid overfitting.***

- [x] True
- [ ] False

***Explanation:*** - Regularization (also sometimes called shrinkage) is a technique that prevents the parameters of a model from becoming too large and “shrinks” them toward 0. The result of regularization is models that, when making predictions on new data, have less variance.

### ***Q41. True or False: in the SVM model, the transition from the primary representation of the optimization problem to the dual representation is intended to reduce the amount the constraints.***

- [ ] True
- [x] False

***Explanation:*** - the transition from the primary representation of the optimization problem to the dual representation is intended for the use of the kernel trick.

### ***Q42. True or False: in the SVM approach removing non-vectors support observations from the study group will never affect the model received.***

- [ ] True
- [x] False

***Explanation:*** -

### ***Q43. True or False: for large training sets there is no significant difference between biased estimator and unbiased estimator***

- [x] True
- [ ] False

***Explanation:*** - The larger the training set the difference between a biased estimator and a unbiased estimator decreases.
