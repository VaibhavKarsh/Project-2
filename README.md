# Rainfall Prediction with Machine Learning
**1. Import Libraries**
Imported essential libraries: pandas for data manipulation, matplotlib and seaborn for visualization, numpy for numerical operations, and tensorflow for building the neural network.

**2. Data Loading and Cleaning**
Loaded the dataset from Rainfall.csv into a DataFrame.
Cleaned column names and checked for missing values, dropping any rows with missing data.
Encoded the rainfall column into binary values (1 for 'yes', 0 for 'no') and removed the 'day' column.
Displayed the cleaned data and computed median values for each rainfall category.

**3. Feature Preparation**
Extracted features (x) and target labels (y).
Applied Min-Max Scaling to normalize feature values.
Split the dataset into training and test sets.

**4. Model Training and Evaluation**

K-Nearest Neighbors (KNN)
Trained a KNN classifier and evaluated its performance using accuracy score and confusion matrix.

Logistic Regression
Trained a Logistic Regression model and assessed its accuracy and confusion matrix.

Support Vector Machine (SVM)
Trained an SVM with a linear kernel and evaluated performance metrics.

Naive Bayes
Trained a Gaussian Naive Bayes model and reported accuracy and confusion matrix.

Random Forest
Trained a Random Forest Classifier with 100 trees and evaluated its performance.

**5. Artificial Neural Network (ANN)**
Built a Sequential ANN model with two hidden layers (each with 7 units) and a sigmoid output layer.
Compiled the model with the Adam optimizer and binary crossentropy loss function.
Trained the ANN for 30 epochs.
Evaluated the ANN on the test set, reporting test accuracy.
Made predictions with the ANN, converted probabilities to binary classes, and displayed accuracy and confusion matrix.
