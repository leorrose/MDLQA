# Evaluation

### ***Q1. True or False: training time of LOOCV (Leave one-out Cross Validation) on a table with 100 examples is at least 10 times longer than the 10-CV Folds training time on this table.***

- [x] True
- [ ] False

***Explanation*** - LOOCV is a special case of k-Fold Cross-Validation where k is equal to the size of data (n) so LOOCV = 100-cv Folds. In 100-cv Folds we train 100 model and in 10-CV Folds we train 10 models so its at least 10 times longer because in the 100-cv Folds each model is trained on more data than in 10-CV Folds.

### ***Q2. True or False: bias reduction will inevitably lead to an increase in variance***

- [ ] True
- [x] False

***Explanation*** - Although there is a bias-variance trade off it is not guarantee that reducing the bias will inevitably lead to an increase in variance.

### ***Q3. True or False: based on the "No Free Lunch" principle, no learning algorithm can be found that surpasses all the other algorithms in a given problem.***

- [ ] True
- [x] False

***Explanation*** - The “no free lunch” (NFL) theorem implies that no single machine learning algorithm is universally the best-performing algorithm for all problems.

### ***Q4. True or False: according to the Occam razor, if two models have the same error on the training set, then the simpler model should be preferred.***

- [x] True
- [ ] False

***Explanation*** - The Occam’s Razor in applied machine learning (Occam’s Two Razors) states that given two models with the same training-set error, the simpler one should be preferred because it is likely to have lower generalization error.

### ***Q5. True or False: if the AUC of model A is higher than that of model B, then the accuracy of model A is necessarily higher than that Of model B***

- [ ] True
- [x] False

***Explanation*** - Accuracy is computed at the certain threshold (usually 0.5). While AUC is computed by adding all the "accuracies" computed for all the possible threshold values. ROC can be seen as an average (expected value) of those accuracies when are computed for all threshold values.

### ***Q6. True or False: based on the "No Free Lunch" principle, no learning algorithm can be found that surpasses all the other algorithms in a every problem.***

- [x] True
- [ ] False

***Explanation*** - The “no free lunch” (NFL) theorem implies that no single machine learning algorithm is universally the best-performing algorithm for all problems.
