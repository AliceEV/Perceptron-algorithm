# Student ID: 201685457
"This code assumes that the test.data and train.data files are in the same folder as this file."

import random
import numpy as np
import pandas as pd
"Two classes are used to generate the perceptron algorithm. The class Binary_perceptron is extended as a superclass to the class Multiclass_perceptron."
class Binary_perceptron:
  """
  An implementation of the perceptron algorithm for the binary classification task. 
  Methods:
    __init__    A constructor method that takes a single argument num_epochs, a user-defined hyperparameter as the number of epochs to run 
                the perceptron algorithm.
    clean_data  A method that takes in three arguments; data, the input dataset, classA and classB, which are the names of the two classes that the dataset should be
                filtered to contain. classA is assigned as the positive class and classB is the negative class. The method returns a new Pandas DataFrame with a subset
                of the original dataset that contains objects belonging to only the two specified classes.
    predict     A method that takes in a single argument, object, which corresponds to an activation score. It returns the sign of that object (i.e. -1 or +1). If the
                input is 0, the sign returned is -1. 
    accuracy    A method that calculates the accuracy of the perceptron's predictions. It takes as arguments a Pandas DataFrame. The method assumes that the last column
                of the input contains the predicted label, and the penultimate column contains the true label of the object.
    train       A method that takes a single argument, a dataset. The method initialises the perceptron's weights and bias to 0 and runs a number of loops equivalent to
                self.num_epochs. Within each epoch, the method iterates through each object in the training dataset, calculates the activation score using the dot product
                of the weights and input object's features and applies the self.predict() method to obtain a label of -1 or +1. If the predicted label is incorrect, the
                method updates the perceptron's weights and bias using the perceptron update rule. The method returns the learned weights and bias.
    test        A method that takes a dataframe (which will have been cleaned to contain objects of just two classes). For each object, an activation score is calculated,
                which is used to provide a predicted class label. 
  """
  def __init__(self, num_epochs: int):
    "The constructor method for the class. Takes an integer argument, num_epochs, which is a hyperparameter determined by the user."
    # Initialise the instance variable
    self.num_epochs = num_epochs
  def clean_data(self, data, classA, classB):
    """
    A method used to extract only the objects belonging to the stated class.
    It reads the file using Pandas and converts it into a numpy array. It then selects the rows (objects) for the array that belong to the stated two classes, classA and
    classB, and stores them in another array. It assigns a binary value of -1 to classA and 1 to classB. It converts the array back to a pandas DataFrame and returns it. 
    Arguments: data, a file, classA, a string, classB, a string. 
    Returns: a Pandas DataFrame
    """
    # Read the csv dataset using Pandas
    f = pd.read_csv(data)
    # Convert it into a numpy array for computation
    arr = f.values
    # Make a dataset of just two classes
    just_two_mask = np.logical_or(arr[:,4] == classA, arr[:,4] == classB)
    just_two_data = arr[just_two_mask]
    # Assign binary values of -1 to class A and 1 to class B
    classA_mask = just_two_data[:,4] == classA
    classB_mask = just_two_data[:,4] == classB
    just_two_data[classA_mask,4] = -1
    just_two_data[classB_mask,4] = 1
    # Store the cleaned array as a DataFrame
    just_two_data = pd.DataFrame(just_two_data)
    return just_two_data
  def predict(self, object):
    """
    A method used to convert a number into the sign of that number, for use as predicted class labels:
    Arguments: object, a float. 
    Returns: -1 if the object is 0, else returns the sign of the object, i.e. +1 or -1.
    """
    # Handle the case where the activation score is 0
    if np.sign(object) == 0:
        # Treat 0 as the negative class; return -1
        return -1
    else:
        # Return the class label, +1 or -1 using numpy.sign()
        return np.sign(object)    
  def accuracy(self, data: pd.DataFrame):
    """
    A method which returns the overall accuracy of the perceptron by comparing the predicted class labels with the actual class labels. It assumes that the last column of
    the dataframe passed as an argument represents the predicted label and the penultimate column is the true label. It computes the true positive, false negative, true
    negative and false positive counts. It uses these to compute the accuracy as (true pos +true neg)/number of objects.
    Arguments: data, a pandas DataFrame.
    Returns: accuracy, a float.  
    """
    # Extract the true labels of all objects from the penultimate column of the DataFrame
    true_y = data.iloc[:,-2].values
    # Extract the predicted labels of all objects from the last column
    pred_y = data.iloc[:,-1].values
    # Create a mask for the positive class
    positive_mask = true_y == 1
    # Create a mask for the negative class
    negative_mask = true_y == -1
    # Count the number of objects that match the positive mask
    tp = np.count_nonzero(pred_y[positive_mask]==1)
    # Count the number of objects predicted negative that were actually positive
    fn = np.count_nonzero(pred_y[positive_mask]==-1)
    # Count the number of objects that match the negative mask
    tn = np.count_nonzero(pred_y[negative_mask]==-1)
    # Count the number of objects predicted positive that were actually negative
    fp = np.count_nonzero(pred_y[negative_mask]==1)
    # Compute the accuracy measure
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    # Handle the case that the denominator equals zero
    if tp+tn+fp+fn == 0:
      return 0
    else:
      return accuracy
  def train(self, d):
    """
    This function uses the perceptron update algorithm to iteratively update the perceptron's weights and biases so that the perceptron learns the optimal values for 
    both. The dataset is converted into an array and the weights vector of zeros is initialised. Fix a random seed so that subsequent calls to the random functions
    will be the same each time the code is run. The array of the dataset is shuffled before the epochs begin. In each epoch, the algorithm iterates through each
    object. For each object, the dot product of the weights vector and feature values is computed, and added to bias gives the activation score. The activation score 
    and true label of the object are multiplied to determine whether there has been a misclassification. If there is a misclassification, the weights and bias are
    updated. The function finally returns the weights and bias.   
    Arguments: d, a pandas DataFrame. 
    Returns: weights, an array representing the weight vector; bias, a float.
    """
    # Initialise the bias to 0
    bias = 0 
    # Convert the dataset to a numpy array
    dataset = np.array(d,dtype=np.float32)
    # Compute the number of features/weights by taking the number of columns from the dataset, excluding the column with the class labels
    num_weights = np.shape(d)[1] - 1
    # Create an array of zeros with the same number of values as the number of features. Make this into a 1d array.
    weights = (np.zeros(num_weights)).reshape(1,4)
    # Initialise a random number generator with a seed value 6
    random.seed(6)
    # Shuffle the dataset
    random.shuffle(dataset)
    # Begin the epochs
    for iter in range(self.num_epochs):
      # Iterate through each object in the dataset
      for object in dataset:
        # Create an array of the feature values
        x_vals = np.array(object[:-1]).reshape(4)
        # Shape the array into a 1d array
        feats = x_vals.reshape(4,1)
        # Compute the activation score as the dot product between the weight vector and feature values, plus the bias
        a = np.dot(weights,feats) + bias
        # Determine the true class label of the object (its last column)
        y = self.predict(object[-1])
        # Multiply the activation score by the true class label, if it is negative then the signs are different and the object has been misclassified
        if a*y <= 0:
            # Update the weights by adding the feature values multipled by the true class label
            weights += y*x_vals
            # Update the bias by adding the true class label
            bias += y
    # Return the weight vector and bias for the pair of classes
    return weights, bias
  def test(self, test_data: pd.DataFrame, weights, bias):
    """
    This method applies the trained perceptron on the given data. It iterates through each object in the data, computes the activation score and uses that to predict 
    the class label for the object. Each predicted label is appended to a list, which is assigned as a new column to the data. The data is returned. 
    Arguments: test_data, a Pandas DataFrame; weights, an array with the weights of the trained model; bias, a float
    Returns: the modified data
    """
    # Reshape the weights to match the expected format for the dot product
    weights = weights.reshape(1,4)
    # Initialise an empty list to store the predicted lables
    y_pred = []
    # Iterate through each object in the data
    for index,object in test_data.iterrows():
      # Extract the feature values into an array and reshape it to match the expected format for the dot product
      x_vals = np.array(object[0:4]).reshape(4)
      feats = x_vals.reshape(4,1)
      # Compute the linear combination of the input features and the weights and add the bias
      a = np.dot(weights,feats) + bias
      # Make the prediction of the class label based on the sign of the linear combination
      sign = self.predict(a)
      # Append the prediction to the list of labels
      y_pred.append(sign[0])
    # Add the predicted labels as a new column to the test data and return it
    test_data = test_data.assign(C=y_pred)
    return test_data
class Multiclass_perceptron(Binary_perceptron):
    """
    An extension of the binary classification perceptron to a multiclass classification problem. 
    Methods:
    __init__, train, test, accuracy, predict:    These methods are provided from the Binary_perceptron superclass.
    regularised_train:  A method that trains the perceptron applying a penalty from l2 regularisation. It takes a dataset and regularisation term (gamma) as inputs. It
                        is essentially identical to the other train method, except that the weights are updated for each object regardless of whether there is a 
                        misclassification, though there is a slightly different calculation depending on whether there was a misclassification or not.
                        It returns the trained weights and bias.
    clean_data:         A modified version of the binary class clean_data method. It cleans the data for all three classes, meaning that one class is selected as the 
                        positive class and the other two are negative. It takes the data to be cleaned and strings representing the three classes as arguments, and 
                        returns the cleaned dataset. 
    predict_class:      A method which allows the perceptron to provide multiclass classification. It takes as inputs the dataset to be tested and the learned weights
                        and bias for each class, and outputs two series: the true labels for the dataset and the predicted labels. 
    multi_accuracy:     A method that gives the accuracy of the multiclass classifier. It takes a dataset, series of true labels and series of predicted labels as 
                        inputs, calculates the accuracy and returns the accuracy.
    """
    def __init__(self, num_epochs: int):
      super().__init__(num_epochs)
    def train(self, d):
      return super().train(d)
    def test(self, test_data, weights, bias):
      return super().test(test_data, weights, bias)
    def accuracy(self, data):
      return super().accuracy(data)
    def predict(self, object):
      return super().predict(object)
    def regularised_train(self,d,gamma: float):
      """
      A method that trains the perceptron with an added l2 penalty. This means that, even if there is no misclassification, the weights are still updated.
      Arguments: d, the dataset; gamma, the l2 regularisation term.
      Returns: the trained weights and bias
      """
      # Initialise the bias to zero
      bias = 0 
      # Convert the dataset to an array
      dataset = np.array(d,dtype=np.float32)
      # Create an array of zeros with the same number of values as the number of features. Make this into a 1d array.
      weights = np.zeros(4).reshape(1,4)
      # Shuffle the array
      random.shuffle(dataset)
      # Begin the epochs
      for iter in range(self.num_epochs):
        # Iterate through each object in the dataset
        for object in dataset:
          # Extract the object's feature values into an array
          x_vals = np.array(object[0:4])
          # Shape the array into the correct shape for the dot product
          feats = x_vals.reshape(4,1)
          # Compute the activation score
          a = np.dot(weights,feats) + bias
          # Compute the true class label from the object's last column
          y = self.predict(object[-1])
          # Determine whether the object has been misclassified
          if a*y <= 0:
            # Update the weights using stochastic gradient descent
            weights = int((1-(2*gamma)))*weights + y*x_vals
            # Update the bias
            bias += y
          # In the case of no misclassification, the weights are still updated with the regularisation term
          else: 
            # Apply l2 regularisation to the weights vector
            weights = int((1-(2*gamma)))*weights
            # The bias remains the same
      return weights, bias   
    def clean_data(self, data, classA, classB, classC):
      """
      This method is a modified version of the binary class method, where it handles three classes rather than just two. It is used to clean the data to prepare it for 
      1-vs-rest multiclass classification, where one class is the positive class and the other two are negative. 
      Arguments: data, the dataset to be cleaned, and strings representing the three classes.
      Returns: the cleaned dataset with binary labels rather than strings for the class labels. 
      """
      # Read the file
      f = pd.read_csv(data)
      # Convert to a dataframe
      df = pd.DataFrame(f)
      # Convert to an array
      arr = df.values
      # Create a boolean array for further data preprocessing
      just_two_mask = np.logical_or.reduce((arr[:,4] == classA, arr[:,4] == classB, arr[:,4] == classC))
      # Mask the boolean onto the array of the dataset to match the class labels
      just_two_data = arr[just_two_mask]
      # Create specific masks for each class
      classA_mask = just_two_data[:,4] == classA
      classB_mask = just_two_data[:,4] == classB
      classC_mask = just_two_data[:,4] == classC
      # Replace the class names with binary values of 1 for classA and -1 for the other two
      just_two_data[classA_mask,4] = 1
      just_two_data[classB_mask,4] = -1
      just_two_data[classC_mask,4] = -1
      # Store the updated data with the binary values as labels and return it
      just_two_data = pd.DataFrame(just_two_data)
      return just_two_data
    def predict_class(self, data, class1_weights, class1_bias, class2_weights, class2_bias, class3_weights, class3_bias):
      """
      This method combines the three binary classifiers using a 1-vs-rest approach. For each object in the dataset, the scores using the weights and biases for each 
      class are calculated, with the highest of these becoming the predicted class. The classes are also mapped to numerical values to be able to compare the list of
      predicted labels and the list of true labels.
      Arguments: data, the dataset; weights for each class, arrays; bias for each class, a float. 

      """
      # Read the data file
      f = pd.read_csv(data)
      # Convert to a dataframe
      data = pd.DataFrame(f)
      # Initialise an empty list to store the predicted class labels
      prediction = []
      # Create a list of tuples, one for each class. Each tuple contains a string of the class in the same format as seen in the data, the weights for the class and the
      # bias. 
      classes = [("class-1",class1_weights,class1_bias),
                  ("class-2",class2_weights,class2_bias),
                  ("class-3",class3_weights,class3_bias)]
      # Create a map to assign an identifying numerical value to each class for comparison
      class_map = {"class-1":1,
                   "class-2":2,
                   "class-3":3}
      # Iterate through each object in the dataset
      for index,object in data.iterrows():
        # Initialise an empty list to store the activation scores for each class
        scores = []
        # Iterate through each of the three classes
        for option in classes:
          # Create a weight vector for the corresponding weights for that class with the correct dimensions for the dot product
          weights_1d = option[1].reshape(1,4)
          # Extract the feature values 
          x_vals  = np.array(object.iloc[:-1])
          # Convert the feature values to the correct dimensions
          feats = x_vals.reshape(4,1)
          # Compute the activation score
          a = np.dot(weights_1d,feats) + option[2]
          # Append the class name and activation score to the scores list
          scores.append((option[0],a.item()))
        # Calculate the class with the maximum score for that object
        max_score = max(scores, key=lambda x:x[1])
        # Select that class as the predicted class by appending the name of the class to the prediciton list 
        prediction.append(max_score[0])
      # Extract the strings from the last column of the dataframe and map them to numerical values
      y_true = data.iloc[:,-1].map(class_map)
      # Create a series from the prediction list and map the class labels to numerical avlues
      y_pred = pd.Series(prediction).map(class_map)
      # Return the two series 
      return y_true, y_pred
    def multi_accuracy(self, data, y_true, y_pred):
      """
      This method allows for the calculation of the accuracy of the multiclass classifier. It takes a series of true values and one of predicted values and counts
      how many of these are identical, by position in the series. 
      Arguments: data, the dataset; y_true, the series of true class labels; y_pred, the series of predicted class labels. 
      Returns: accuracy, a float. 
      """
      # Read the data file
      f = pd.read_csv(data)
      # Convert the data file to a dataframe
      data = pd.DataFrame(f)
      # Count the number of correctly predicted class labels
      true_vals = np.sum(y_pred == y_true)
      # Calculate the accuracy as (tn + tp)/ (total number of objects)
      accuracy = true_vals/len(data)
      # Return the float accuracy
      return accuracy

"Question 2: Implement a binary perceptron"

# Create a binary perceptron instance
perceptron = Binary_perceptron(20)

"Question 3: Use the binary perceptron to train classifiers to discriminate between each pair of classes. Train for 20 iterations and report the train and test accuracies." 

# Clean the training data for each class pair
class1_2_train_data = perceptron.clean_data("train.data","class-1","class-2")
class2_3_train_data = perceptron.clean_data("train.data","class-2","class-3")
class1_3_train_data = perceptron.clean_data("train.data","class-1","class-3")

# Train the perceptron to provide the weights and bias for each class pair
class1_2_weights, class1_2_bias = perceptron.train(class1_2_train_data)
class2_3_weights, class_2_3_bias = perceptron.train(class2_3_train_data)
class1_3_weights, class1_3_bias = perceptron.train(class1_3_train_data)

# Clean the testing data for each class pair
class1_2_test_data = perceptron.clean_data("test.data","class-1","class-2")
class2_3_test_data = perceptron.clean_data("test.data","class-2","class-3")
class1_3_test_data = perceptron.clean_data("test.data","class-1","class-3")

# Test the weights and bias for each class pair on both the cleaned train and test dataset
print("\nClass 1 and 2 training:")
class1_2_train = perceptron.test(class1_2_train_data,class1_2_weights,class1_2_bias)
print(f"Accuracy = {perceptron.accuracy(class1_2_train):.2%}")
print("\nClass 1 and 2 testing:")
class1_2_test = perceptron.test(class1_2_test_data,class1_2_weights,class1_2_bias)
print(f"Accuracy = {perceptron.accuracy(class1_2_test):.2%}")
print("\nClass 2 and 3 training:")
class2_3_train = perceptron.test(class2_3_train_data,class2_3_weights,class_2_3_bias)
print(f"Accuracy = {perceptron.accuracy(class2_3_train):.2%}")
print("\nClass 2 and 3 testing:")
class2_3_test = perceptron.test(class2_3_test_data,class2_3_weights,class_2_3_bias)
print(f"Accuracy = {perceptron.accuracy(class2_3_test):.2%}")
print("\nClass 1 and 3 training:")
class1_3_train = perceptron.test(class1_3_train_data,class1_3_weights,class1_3_bias)
print(f"Accuracy = {perceptron.accuracy(class1_3_train):.2%}")
print("\nClass 1 and 3 testing:")
class1_3_test = perceptron.test(class1_3_test_data,class1_3_weights,class1_3_bias)
print(f"Accuracy = {perceptron.accuracy(class1_3_test):.2%}")

"""
Question 4: Extend the binary perceptron implemented above to perform multi-class classification using the 1-vs-rest approach. Train for 20 iterations and compare the
train and test classification accuracies.
"""

# Create a multiclass perceptron instance
perceptron_multi = Multiclass_perceptron(20)

# Clean the data for each class to prepare it for the 1-vs-rest approach
class1_train_data = perceptron_multi.clean_data("train.data","class-1","class-2","class-3")
class2_train_data = perceptron_multi.clean_data("train.data","class-2","class-1","class-3")
class3_train_data = perceptron_multi.clean_data("train.data","class-3","class-1","class-2")

# Train the perceptron to give the weights and bias for each class
class1_weights, class1_bias = perceptron_multi.train(class1_train_data)
class2_weights, class2_bias = perceptron_multi.train(class2_train_data)
class3_weights, class3_bias = perceptron_multi.train(class3_train_data)

print("------------------------------ \nMulticlass classifier:")

# Predicts the classes using predict_class for both the train and test data, and outputs two series representing the true labels and predicted labels
# Calculates the accuracy of the classifier for both the train and test data
train_true, train_pred = perceptron_multi.predict_class("train.data",class1_weights, class1_bias, class2_weights, class2_bias, class3_weights, class3_bias)
train_acc = perceptron_multi.multi_accuracy("train.data",train_true,train_pred)
print(f"Training accuracy = {train_acc:.2%}")
test_true, test_pred = perceptron_multi.predict_class("test.data",class1_weights, class1_bias, class2_weights, class2_bias, class3_weights, class3_bias)
test_acc = perceptron_multi.multi_accuracy("test.data",test_true,test_pred)
print(f"Testing accuracy = {test_acc:.2%}")

"Question 5: add an l2 regularisation term to the multi-class classifier. Compare the train and test classification accuracies." 
print("------------------------------\nWith regularisation:")
# Create a list to iterate through for training
clean_data = [class1_train_data,class2_train_data,class3_train_data]
# Iterate through each regularisation coefficient
for term in [0.01,0.1,1.0,10.0,100.0]:
  print(f"\nRegularisation coefficient is {term}.")
  # Initialise a counter to be used as an index
  classCount = 0
  # Initialise an empty list to store the weights and biases for each class
  weights_biases = []
  # Iterate through each class
  for num in ["Class 1","Class 2","Class 3"]:
    # Select the respective train data for that class
    data = clean_data[classCount]
    # Train the perceptron to give the weights and bias for that class
    weights, bias = perceptron_multi.regularised_train(data,term)
    # Append the weights to the list of weights and biases
    weights_biases.append(weights[0])
    # Append the bias to the list of weights and biases
    weights_biases.append(bias)
    # Increment the class counter
    classCount += 1

  # Given the weights and biases for each class, perceptron predicts the classes for each object for both the train and test data. This is done for each 
  # regularisation coefficient as it is still in the loop. Outputs two series representing the true labels and predicted labels
  # Calculates the accuracy of the classifier for both the train and test data  
  print("Training:")
  train_true, train_pred = perceptron_multi.predict_class("train.data",weights_biases[0],weights_biases[1],weights_biases[2],weights_biases[3],weights_biases[4],weights_biases[5])
  train_acc = perceptron_multi.multi_accuracy("train.data",train_true,train_pred)
  print(f"Accuracy = {train_acc:.2%}")
  print("Testing:")
  test_true, test_pred = perceptron_multi.predict_class("test.data",weights_biases[0],weights_biases[1],weights_biases[2],weights_biases[3],weights_biases[4],weights_biases[5])
  test_acc = perceptron_multi.multi_accuracy("test.data",test_true,test_pred)
  print(f"Accuracy = {test_acc:.2%}")