# Perceptron-algorithm
An assignment from a Data Mining course, entailing a from-scratch implementation of the perceptron algorithm
Here is the assignment brief: 

Objectives

This assignment requires you to implement the Perceptron algorithm using the Python program-
ming language.

Assignment description

Download the CA1data.zip file. Inside, you will find two files: train.data and test.data, corre-
sponding respectively to the train and test data to be used in this assignment. Each line in the
file represents a different train/test instance. The first four values (separated by commas) are
feature values for four features. The last element is the class label (class-1, class-2 or class-3).

Questions/Tasks
1. (15 marks) Explain the Perceptron algorithm (both the training and the test procedures)
for the binary classification case. Provide the pseudo code of the algorithm. It should be
the most basic version of the Perceptron algorithm, i.e. the one that was discussed in the
lectures.

2. (30 marks) Implement a binary perceptron. The implementation should be consistent
with the pseudo code in the answer to Question 1.
3. (15 marks) Use the binary perceptron to train classifiers to discriminate between
• class 1 and class 2,
• class 2 and class 3, and
• class 1 and class 3.
Report the train and test classification accuracies for each of the three classifiers after
training for 20 iterations. Which pair of classes is most difficult to separate?

4. (30 marks) Explain in your own words what the 1-vs-rest approach consist of. Extend the
binary perceptron that you implemented in part 3 above to perform multi-class classification
using the 1-vs-rest approach. Report the train and test classification accuracies for the
multi-class classifier after training for 20 iterations.

5. (10 marks) Add an `2 regularisation term to your multi-class classifier implemented in
part 4. Set the regularisation coefficient to 0.01, 0.1, 1.0, 10.0, 100.0 and compare the train
and test classification accuracies. What can you conclude from the results?
