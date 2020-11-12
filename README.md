# Machine Learning Homework - 2

Implementation of the:
* [One-vs-All Logistic Regression](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)
* [Softmax Classifier](https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/)
* [Gaussian Discriminant Analysis](https://towardsdatascience.com/gaussian-discriminant-analysis-an-example-of-generative-learning-algorithms-2e336ba7aa5c)
* [Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## Dataset 

Dataset consists out of 176 samples divided into three classes. Each sample has 5 values, where the last one represents the class label (0-2) and the other 4 are features.
Dataset was split at the beginning into train-test using 80-20 [%] split. Dataset was stratified during the split so it would maintain the same class-to-class ratio. 

## Hyperparameter tuning
Values for all hyperparameters were decided by using K-fold cross-validation with K=5. During the cross-validation the models were trained for 300 epochs.

## One-vs-All Logistic Regression

Three distinct Logistic Regression models were created for this cause, one for each class. Each of those models was trained to determine if sample belongs to the class that model was assigned to. During evaluation each of those three models outputs the probability that the sample belongs to their classes. Final decision is based on the fact which model have the highest probability.

## Softmax Classifier

This model is the generalization of the Logistic Regression model for variable number of classes (Logistic Regression can only support two). It has K output units, one for each possible class. At it's output it gives the probabilities that sample belongs to each of the possible classes. Those probabilities sum up to 1. Final decision is based on the unit which has the highest probability. 

## Gaussian Discriminant Analysis

It models the conditional feature distribution for each of the possible classes. So it learns multinomial distributions for features for each class. When new sample arrives it estimates the probability that it's features came from each of the possible classes. It later uses those probabilities in the Bayes rule to make a final decision. The main drawback of this model is the fact that it assumes that input features have a normal distribution.

## Gaussian Naive Bayes

It models the conditional distribution for each feature separately for each of the possible classes. So it learns 12 different distributions (3 classes * 4 features). When new sample arrives it estimates the probability that each of it's features came from each of the possible classes. It later uses those probabilities in the Bayes rule to make a final decision.

# Test Set Performance

| Model  | Test Set Acc. |
|:------:|:-------------:|
| One-vs-All | 100 % |
| Softmax | 100 % |
| GDA | 100 % |
| GNB | 97.22 % |


## Setup & instructions
1. Open Anaconda Prompt and navigate to the directory of this repo by using: ```cd PATH_TO_THIS_REPO ```
2. Execute ``` conda env create -f environment.yml ``` This will set up an environment with all necessary dependencies. 
3. Activate previously created environment by executing: ``` conda activate ml-homework-2 ```
4. a) Execute ``` jupyter notebook ```. This will open up jupyter notebook in your deafult browser. 
   b) Open [ML_2_main.ipynb](src/ML_2_main.ipynb) notebook.
   c) Experiment with the code
   
   **NOTE: This step isn't necessary. You can open this notebook in any editor/IDE which supports jupyter notebooks.**
