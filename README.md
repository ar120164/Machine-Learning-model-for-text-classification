# Machine-Learning-model-for-text-classification
NLP model to classify Paycheck Protection Program (PPP) borrowers

## **Objective:**
To build a natural language processing model to classify Paycheck Protection Program (PPP) borrowers into companies or individuals based on their names.

## **Context**
Paycheck Protection Program (PPP) loans are implemented by the U.S. Small Business Administration with support from the Department of the Treasury.  This program provides small businesses with funds to pay up to 8 weeks of payroll costs, including benefits. The funds can also be used to pay interest on mortgages, rent and utilities. The Paycheck Protection Program (PPP) prioritizes millions of Americans employed by small businesses by authorizing up to $659 billion for job maintenance and certain other expenses.

A list of PPP borrowers is available, however, the data does not clearly distinguish whether the borrowers are businesses or individuals. Therefore, the objective is to find a way to classify the list of PPP borrowers into businesses and individuals. On the other hand, two different datasets are available, one with a list of firms and one with a list of individuals. From these two datasets, a natural language processing model is proposed to classify which PPP loan borrowers are companies or individuals from their names. 

## **Development**

**Model selection**
To select the model, a search was conducted on what models are used for classifying text into categories. In this search, we found a Multinomial Naive Bayes model developed for an analogous case in which the objective was to find the country of origin of a person from his or her name. With this example in mind and what we consulted about its advantages (text mining friendly, faster convergence than other models such as logistic regression, highly scalable and easily handles large datasets), it was decided to apply Multinomial Naive Bayes. However, some of the options found as alternative models are Bernoulli Naive Bayes (another type of Naive Bayes), Logistic Regression, k-Nearest Neighbors, decision trees and support vector machine.
The Naive Bayes learning algorithm is widely used in text classification problems because it is computationally efficient and simple to implement. It employs the bag-of-words approach, in which individual words in the document serve as features, ignoring word order. There are two types of event models that are widely used: 
- Bernoulli multivariate event model .
- Bayes multivariate event model.

The multinomial Naive Bayes algorithm refers to a vector of features in which each term reflects the number of times it occurs or the frequency with which it appears. Bernoulli, on the other hand, is a binary algorithm that determines whether a feature is present or not. Finally, there is also the Gaussian model (not widely used for text classification), which is based on a continuous distribution.

The Naive Bayes algorithm is based on Bayes' theorem, which states that the features of a data set must be mutually independent. This assumption is not usually true, however after reading about the subject, it was found that "...in practice Naive Bayes models have performed surprisingly well, even on complex tasks where it is clear that strong independence assumptions are false..."

**Text preprocessing**

A machine learning model cannot interpret name data as text data. Names must first be converted into a numerical representation, so a "bag-of-words" technique must be used. The concept is to create a vocabulary that includes a collection of different words, each of which is associated with a count of how many times it appears. To do this I used CountVectorizer from the scikit-learn library. First, I set up CountVectorizer and fitted it with a list of voters and companies, each with their words separated by blanks. In the fitting process, the voter and company names are split into their component words and a word dictionary is created.

Then, the name data is converted into row vectors by .transform() , with word frequency information. The output is a scipy sparse matrix in which the columns are the unique words identified, the rows are the observations (voters and companies) and the value at position i, j represents the number of times word j (now labeled as an integer) appears in observation i.


**Data splitting**
Finally, the data will be randomly split into training and test sets with a split ratio of 30% for the training and test set respectively. This is done using the .train_test_split() method, in sklearn.model_selection. x is represented by word_mat which is the sparse matrix described above and y are the classes (0 for person and 1 for company) of each observation. 

**Model construction**
Now the input for training the model is ready. I used the Multinomial Naive Bayes class from scikit-learn which is one of the two classical variants of Naive Bayes used in text classification. This specific model is the one that works with data represented as word vector counts.

The first step is to create a new object of type MultinomyNB(). Next, the model is fed with the training data and their respective labels. With this, the model is ready to predict the labels for 30% of the unused data in the dataset. For this purpose, predict is used.

The output is a vector of size equal to the number of observations in x_test, containing the classifications of each observation in terms of whether it is a person (0) or whether it is a company (1). To classify the new information with this model, the vectorizer was used to transform the new input data into word count vectors. Subsequently, the predict function is used again on the transformed data to obtain the classification.

## **Results**

**Data characteristics.**
- Voter data set: In general, some letters are the most common in voters' names, this occurs because voters' names include the middle name which is sometimes represented only by initials in the dataset. Other features with high frequency correspond to first or middle names such as ANN, MARIE, LEE, LYNN, etc.

- Company dataset: The most important character here is "&", which is one of the most common company traits in the PPP borrower name dataset and with which the model can apparently identify that the observations containing it are companies. In contrast to names, individual letters are not as common here. Instead, words that are very characteristic of company names, such as SERVICES, GROUP, COMPANIES, INVESTMENTS, and MANAGEMENT, are important for the identification of these companies.

- PPP borrower name dataset: The most common words in the PPP borrower name dataset are usually related to companies (with the exception of letters such as A, M, S and J).

After training, testing and cross-validating a multinomial Naive Bayes using the names of all registered companies in Florida and the names of voters in the same state, we observed an overall model accuracy of about 98%, showing a confident ability of the model to classify PPP borrowers into companies or individuals using their names.

Although words such as SMALL, INC or LLC are not the most common words for companies in the data with which the model was built and validated, it is clear that they are important in the classification. This, given that after applying the model on the names of the PPP borrowers, they are very common in the observations classified as companies.

## **Conclusions**
It is expected that researchers can use the classification of PPP borrowers to develop other analyses to compare individuals and firms.
