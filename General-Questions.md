# ***General Questions***

### ***Q1. True or False: Newton's method is good for converging to the minimum region even when the starting point is not close to the point the optimum.***

- [ ] True
- [x] False

***Explanation*** - If the guess is not close enough it can cause the method to approximate the wrong root or the method will diverge entirely.

### ***Q2. True or False: it is always possible to find an accurate solution for the classifier variables with analytical methods regardless of the type of classifier constructed.***

- [ ] True
- [x] False

***Explanation*** - We prefer the analytical method in general because it is faster and because the solution is exact. Nevertheless, sometimes we must resort to a numerical method due to limitations of time or hardware capacity. For example when we cannot fit all the data into the memory of a single computer in order to perform the analytical calculation. Also, sometimes the analytical solution is unknown and all we have to work with is the numerical approach.

### ***Q3. True or False: given the function ![eq](1.png). If there is an optimum point then Newton's method will inevitably find the exact solution within one iteration regardless of the starting point Search***

- [ ] True
- [x] False

### ***Q4. True or False: existence of KKT conditions for a non-quadratic problem, represents the necessary condition for a global optimal solution.***

- [ ] True
- [x] False

***Explanation*** - 

### ***Q5. True or False: existence of KKT conditions for a quadratic problem, represents the sufficient condition for a global optimal solution.***

- [x] True
- [ ] False

***Explanation*** - 

***Explanation*** - Newton's method has a quadratic convergence but it depends on the step size and starting point so it will not always converge in exact one iteration.

### ***Q6. True or False: Trying to solve a multiclass classification problem where clases are {a,b,c,d} using ECOC with 3 models. model 1 is intended to distinguish between the group {a,b} and {c,d}. model 2 is intended to distinguish between the group{a,c} and {b,d}. model 3 is intended to distinguish between the group {a,d} and {b,c}. If the models classified a new observation to groups 1,0,1 respectively then the observation should be assigned to group c.***

- [x] True
- [ ] False

***Explanation*** - {c,d} & {a,c} & {n,c} = {c}.

### ***Q7. True or False: the Error-Correcting Output Codes approach is designed to enable classification trees to address Multi-Class problems.***

- [x] True
- [ ] False

***Explanation*** - The Error-Correcting Output Codes method is a technique that allows a multi-class classification problem to be reframed as multiple binary classification problems, allowing the use of native binary classification models to be used directly.

### ***Q8. True or False: one hot encoding converts a categorical variable to a binary variables***

- [x] True
- [ ] False

***Explanation*** - A one hot encoding is a representation of categorical variables as binary vectors.

### ***Q9. True or False: for large training sets there is no significant difference between biased estimator and unbiased estimator***

- [x] True
- [ ] False

***Explanation*** - The larger the training set the difference between a biased estimator and a unbiased estimator decreases.

### ***Q10. Multi-class problems can be solved by converting to a number of binary classification problems (with binary classifications only). Explain how the conversion is recommended. Explain in which cases this method is required.***

***Answer & Explanation*** - Many algorithms such as the Perceptron, Logistic Regression, and Support Vector Machines were designed for binary classification and do not natively support classification tasks with more than two classes. So we need to find a way to use these algorithms for multi-class classification. this can be done using One-vs-One or One-vs-All.