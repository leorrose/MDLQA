# Support Vector Machine

### ***Q1. True or False: the original SVM algorithm is intended for solving binary problems only.***

- [X] True
- [ ] False

***Explanation*** - SVMs (linear or otherwise) inherently do binary classification. However, there are various procedures for extending them to multi-class problems.

### ***Q2. True or False: in the SVM algorithm, the dual representation is intended to allow the use of Kernel without the explicit knowledge of the conversion function.***

- [x] True
- [ ] False

***Explanation*** - The dual representation enables us to use the kernel trick. The ultimate benefit of the kernel trick is that the objective function we are optimizing to fit the higher dimensional decision boundary only includes the dot product of the transformed feature vectors. Therefore, we can just substitute these dot product terms with the kernel function, and we don’t even use ϕ(x).

### ***Q3. True or False: in the SVM model with the Kernel function It is necessary to know the explicit function of translating the observations from the original space to the new space.***

- [ ] True
- [x] False

***Explanation*** - The ultimate benefit of the kernel trick is that the objective function we are optimizing to fit the higher dimensional decision boundary only includes the dot product of the transformed feature vectors. Therefore, we can just substitute these dot product terms with the kernel function, and we don’t even use ϕ(x).


### ***Q4. True or False: in the SVM model, the transition from the primary representation of the optimization problem to the dual representation is intended to reduce the amount the constraints.***

- [ ] True
- [x] False

***Explanation*** - the transition from the primary representation of the optimization problem to the dual representation is intended for the use of the kernel trick.

### ***Q5. True or False: in the SVM approach removing non-vectors support observations from the study group will never affect the model received.***

- [ ] True
- [x] False

***Explanation*** -

### ***6. True or False: theres in no extension that allows using SVM algorithm for solving multi-class problems.***

- [ ] True
- [x] False

***Explanation*** - SVMs (linear or otherwise) inherently do binary classification. However, there are various procedures for extending them to multi-class problems.