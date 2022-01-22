# Decision Tree

### ***Q1. True or False: if a split in a decision tree decreases the entropy then the error will decrease.***

- [ ] True
- [x] False

***Explanation*** - For example we have 0 and 1 classes. If we have a node with (40,80) classes and we split into two nodes (28,42) and (12,38). The entropy will decrease from 0.918 to 0.897 but the the misclassification error will increase from 0.33 to 0.64.

### ***Q2. True or False: at a tree node there are 80 instances with class 1 and 20 with class 0. then entropy = (20/100) log log(20/100) + (80/100) log log(80/100)***

- [ ] True
- [x] False

***Explanation*** - entropy = -(20/100) log(20/100) + -(80/100) log(80/100)

### ***Q3. True or False: in a decision tree with binary splits the number of leaves will not exceed the number of observations in the train.***

- [x] True
- [ ] False

***Explanation*** - Each node in the decision tree splits our observations. if we split our data until we get only leafs than the maximum number of leafs in the number of observations (each observations is a leaf).

### ***Q4. True or False: in the decision tree pruning the entropy value before pruning is always greater than the entropy value after pruning***

- [ ] True
- [x] False

***Explanation*** - 

### ***Q5. True or False: given two decision trees studied on the same database. In the first tree 20 leaves and in the second tree 200 leaves. The bias component of the tree with 200 leaves is larger than the bias component in a tree of 20 leaves.***

- [ ] True
- [x] False

***Explanation*** - Because building a tree is deterministic and the more leafs we have the more complicated our decision tree so a tree with 200 leafs can fit better to the data and lead to lower bias. In general, more complicated models will result in lower bias.
