# Ensemble Learning

### ***Q1. True or False: in the Bagging algorithm the size of the ensemble built is always equal to the number of iterations (models) determined by the user.***

- [x] True
- [ ] False

***Explanation*** - In the Bagging algorithm the user inputs a base learner, number of iterations and n as input. In each iteration a bootstrap sample of size n is selected and a base learner is trained on this sample. At the end the models are combined.

### ***Q2. True or False: one of the advantages of the Bagging algorithm over the AdaBoost algorithm is that Bagging can be performed in parallel without any change in code, so each classifier is built on a different processor/computer.***

- [x] True
- [ ] False

***Explanation*** - Bagging is a parallel method that fits different, considered learners independently from each other, making it possible to train them simultaneously.

### ***Q3. True or False: the original AdaBoost algorithm is intended for solving binary problems only.***

- [x] True
- [ ] False

***Explanation*** - AdaBoost in its original form was designed for binary classification. It has been proved that this classifier is extremely successful in producing accurate classification when applied to two-class classification problems. AdaBoost was also proposed to be used as multi-class classifier by Freund and Schapire.

### ***Q4. True or False: the difference between Ensemble and Mixtures of Experts, is that in Ensemble classifiers are created from the same Basic algorithm and in MoE this is usually not the case.***

- [ ] True
- [x] False

***Explanation*** - In ensemble we use a lot of algorithms and combine them but in MoE we use a small number of models where each model specializes in some aspect. the models in MoE can be from the same basic algorithm.

### ***Q5. True or False: in most cases the accuracy of the first classifier built by the AdaBoost Ensemble has a greater accuracy than the accuracy of first classifier built by the Bagging Ensemble.***

- [x] True
- [ ] False

***Explanation*** - In Bagging each classifier is built on a sample of the data or features but in AdaBoost each classifier is trained on the full weighted data and features so the first classifier in AdaBoost has more data and features to learn on so its accuracy should be greater than the first classifier build with random forest.

### ***Q6. The ML team received a file with new training data. The task is to create a new classification model based on this data. The input features are nominal only. Also the target attribute is binary. The team is debating between five possible classifiers: KNN, ID3, Naive Bayes, OneR, C4.5. To select the best classifier, it was decided to linearly weight the output of these classifiers (The probabilities for each CLASS). Which Ensemble method is suitable for this case?***

***Answer & Explanation*** - Stacking, Stacking is a way to improve model predictions by combining the outputs of multiple models and running them through another machine learning model.

### ***Q7. Given a training set that includes 100 attributes and 100 observations - which Ensemble method do you recommended to use (only one ensemble method). Explain why.***

***Answer & Explanation*** - Bagging, because it uses bootstrapping.

### ***Q8. True or False: in the AdaBoost algorithm the level of accuracy on the test set may continue to improve even when the level the accuracy of the train set reached 100%***

- [x] True
- [ ] False

***Explanation*** -  This question is answered by Schapire and Freund, et. al. Even when the training error is zero, the margin (= sample distance to decision boundary) is still improved by further boosting iterations. This can lead to improving the test set accuracy even the train set reached 100%.

### ***Q9. True or False: for the purpose of thinning a forest of trees, it is advisable to dilute trees with a low correlation with the target feature and / or other trees.***

- [ ] True
- [ ] False

***Explanation*** - 

### ***Q10. True or False: the running time required to study a single tree according to C4.5 algorithm  is greater than the running time required to study a randomForest with a single tree.***

- [x] True
- [ ] False

***Explanation*** - Each tree in a random forest is not trained on all sample\features meaning if we train only one tree it should take lees than training a single C4.5 which uses all sample\features to builds a tree.

### ***Q11. True or False: if we increase the max depth in Random Forest then the chance of overfitting will increase***

- [x] True
- [ ] False

***Explanation*** - Increasing depth decreases bias at the expense of increasing variance. Random forests can combat this increase in variance by aggregating over multiple trees, but are not immune to overfitting.

### ***Q12. True or False: Boosting is designed to reduce the bias component in models with small variance.***

- [ ] True
- [x] False

***Explanation*** - Boosting is a meta-learning algorithm that reduces both bias and variance.

### ***Q13. True or False: both the bagging algorithm and the random forest algorithm can be executed in parallel without any change in the code, so each classifier is built in a different processor / computer***

- [x] True
- [ ] False

***Explanation*** - Bagging is a parallel method that fits different, considered learners independently from each other, making it possible to train them simultaneously. Random forest is a bagging algorithm.

### ***Q14. True or False: increasing the size of the ensemble will always reduce the error on the test set.***

- [ ] True
- [x] False

***Explanation*** - If this was true than we will just keep increasing the number of size of the ensemble until we get 0 error and we wouldn't need any other algorithm.

### ***Q15. True or False: the AdaBoost algorithm reduces both the bias component and the variance component.***

- [x] True
- [ ] False

***Explanation*** - Boosting is a meta-learning algorithm that reduces both bias and variance.

### ***Q16. True or False: the bagging algorithm reduces both the bias component and the variance component.***

- [ ] True
- [x] False

***Explanation*** - Bagging reduces the variance without reducing the bias.
