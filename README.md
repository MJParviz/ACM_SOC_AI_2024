
# Summer of Code 2024

The "Summer of Code" event, organized by the ACM Branch Students of Computer Engineering at the University of Tehran, focuses on artificial intelligence, machine learning, and deep learning. This program aims to provide students and enthusiasts with hands-on experience and knowledge in these cutting-edge fields. Participants engage in workshops, lectures, and practical projects, fostering collaboration and innovation. The event not only enhances technical skills but also encourages networking among peers and professionals in the AI community. By the end of the program, attendees gain valuable insights and practical experience that can help propel their careers in technology. 

### Week one

Bonus Task : 
#### Monte Carlo Simulation

Monte Carlo Simulation is a powerful technique used to model and analyze complex systems through random sampling. This method helps in understanding the behavior of systems and processes that are difficult to solve analytically. We will explore this technique in two different scenarios: estimating the value of Pi and analyzing the Mensch game.

#### Pi Calculation

First, we will use Monte Carlo simulation to estimate the value of Pi (π). The approach involves generating random points within a square and determining how many of those points fall inside an inscribed circle.
#### Mensch Game

The Mensch game, known in Germany and popular in many other countries, is a classic board game. For our analysis, we will focus on a simplified version of this game where each player has only one piece, and the gameplay is purely based on rolling dice and moving the piece. The objective is to calculate the probability of winning for each player (1st, 2nd, 3rd, and 4th) using Monte Carlo Simulation. 



### Week two

Main Task :

#### Introduction
The objective of this exercise is to become familiar with machine learning methods for predicting housing prices in Boston, USA. This exercise consists of three main sections and one optional section:

- **Introduction to the Dataset:**
   In this section, we will familiarize ourselves with the data, including the distribution, types of data, and statistical information related to the dataset. This part is generally referred to as data analytics.

- **Data Preprocessing:**
   This section is the most crucial part of any machine learning project. Using the insights from the previous section, we will transform raw real-world data into a suitable format for a machine learning model. This process involves cleaning and summarizing the data.

- **Model Creation and Evaluation:**
   In this section, we will create and evaluate various machine learning models. It consists of six phases, starting with manually implementing a first-order Linear Regression model without using pre-built libraries. We will then implement gradient descent and polynomial regression methods, followed by using the scikit-learn library for house price prediction. Subsequent phases will introduce more advanced models, and finally, we will evaluate all models to draw final conclusions.

Bonus Task : 
#### Part 1: Functions’ Implementation

Implement the following functions from scratch:

- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of the average of the squared differences between actual and predicted values.
- **R² Score (Coefficient of Determination)**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

#### Part 2: Building and Training the Linear Regression Model

Construct a regression model and train it using the diabetes dataset.

#### Part 3: Polynomial Regression

Create a polynomial regression model and evaluate its accuracy against the linear regression model.

#### Part 3: Model Evaluation

-  **Scatter Plot**: Compare the predicted values with the actual progression measures using a scatter plot. The x-axis represents the actual values, and the y-axis represents the predicted values.
-  **Evaluate the Model**: Evaluate the regression model on the training and testing data using the following functions:
   - **MSE**
   - **MAE**
   - **RMSE**
   - **R² Score**
### Week three

Main Task :
In this comprehensive project, you will explore various machine learning algorithms and techniques through practical exercises. The project is designed to provide hands-on experience with key concepts such as Bayes' Theorem, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Logistic Regression, and Recommender Systems. By working through these exercises, you'll gain a deeper understanding of how to preprocess data, develop predictive models, evaluate their performance, and fine-tune them for optimal results

- **Text Processing and Bayes' Theorem :** In this project, you are given a dataset in CSV format named `books_train.csv` containing book descriptions and their corresponding categories. Another file named `books_test.csv` contains descriptions of books without categories. Your task is to determine the category of each book based on its description.
- **K-Nearest Neighbors (KNN) :** In this project, you will work with the `Metro_Interstate_Traffic_Volume` dataset to predict traffic volume based on various features. The project will involve data visualization, preprocessing, and the application of the K-Nearest Neighbors (KNN) algorithm to develop a predictive model.

- **Recommender Systems with KNN :** In this exercise, you will develop a simple recommender system using the K-Nearest Neighbors (KNN) algorithm. The project will involve understanding collaborative filtering techniques, implementing a user-based collaborative filtering model, and evaluating its performance.

- **SVM, Logistic Regression, and ROC Analysis :** In this project, you will work with the `titanic.csv` dataset to predict which passengers survived the Titanic disaster. The project will involve data exploration, preprocessing, model development using SVM and Logistic Regression, and performance evaluation using ROC curves.

Bonus Task : 
- **Implementation of Simple Linear Regression :** In this phase, you will implement a simple Linear Regression model from scratch, focusing on a first-degree (linear) regression. You are not allowed to use any pre-built machine learning libraries (except for numpy). The goal is to predict the number of purchases a customer makes (represented in the Price in Thousands column) using the provided dataset.
- **Implementation of Multivariate Regression from Scratch :** You will implement a multivariate linear regression model from the ground up without relying on machine learning libraries like scikit-learn. This model will predict multiple dependent variables (e.g., "Price in Thousands" and "Horsepower") based on one or more independent variables.
- **Implementation of Manual K-Fold Cross-Validation from Scratch :** You will implement K-Fold cross-validation manually and use it to validate the multivariate regression model. K-Fold cross-validation involves dividing the dataset into K subsets, using each subset as a validation set while training on the remaining K-1 subsets, then averaging the performance.
### Week four
Main Task : 
The objective is to analyze a set of images using clustering algorithms. The data, provided in the form of images and associated CSV files, must be grouped into clusters based on similarity.

Steps:

- Preprocess the images and extract features using a pre-trained Convolutional Neural Network (CNN) like VGG16.
- Implement clustering algorithms (K-Means and DBSCAN) on the feature vectors to group the images into clusters.
- Evaluate and compare the performance of the clustering methods.

Bonus Task : CNN Implementation for Image Classification

The goal of this project is to implement a convolutional neural network (CNN) for classifying images using a popular deep learning framework like PyTorch. You are expected to understand the process of building, training, and evaluating CNN models, with a focus on image data.


