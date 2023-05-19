import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# Reads data from local file 
path1 = "ppp_borrower_names.txt" # Path contains the location of the spyder's output dataset
path2 = "florida_voters.txt" # Path contains the location of the spyder's output dataset
path3 = "florida_firms.txt" # Path contains the location of the spyder's output dataset

PPP_names = pd.read_csv(path1, delimiter = "\t", header=0)
Voters = pd.read_csv(path2, delimiter = "\t", header=0)
Firms = pd.read_csv(path3, delimiter = "\t", header=0)

#######################################
#DATA PRE-PROCESSING 

#Creating voters dataset with Complete name of the voter and corresponding class
Voters= Voters[['name_first', 'name_middle', 'name_last']]
Voters['Name'] = Voters['name_first'] + str(' ')  + Voters['name_middle'] + str(' ') + Voters['name_last']
Voters['Class']= 0 #PERSON
Voters= Voters.drop(columns=['name_first', 'name_middle', 'name_last'])
Voters.dropna(subset = ["Name"], inplace=True)

#Extracting first 20% of voters for the construction of the ML model
#(Not all data was used due to memory reasons)
Voters0 = Voters[:int((len(Voters))*0.2)] 
print(Voters0.shape)

#Creating firms dataset with Name of the firm and corresponding class
Firms = Firms[['stn_name']]
Firms = Firms.rename(columns={'stn_name':'Name'})
Firms['Class']= 1 #FIRM
Firms.dropna(subset = ["Name"], inplace=True)

#Extracting first 40% of firms for the construction of the ML model
#(Not all data was used due to memory reasons)
Firms0 = Firms[:int((len(Firms))*0.4)] 
print(Firms0.shape)

#Creating complete dataset for training and testing the ML model
Data = Voters0.append(Firms0, ignore_index = True)
print(Data.shape)


##################################
#MODEL

# Initialize and fit CountVectorizer with given text documents
vectorizer = CountVectorizer().fit(Data['Name'].values.astype('U'))
# use the vectorizer to transform the document into word count vectors (Sparse)
word_mat = vectorizer.transform(Data['Name'].values.astype('U'))
print((word_mat.shape))
print(vectorizer.vocabulary_)

print(Firms['Name'].isna().sum())
print(Voters['Name'].isna().sum())

#Split into training and test set (with split ratio  70%:30% )
x_train, x_test, y_train, y_test = train_test_split(word_mat, Data['Class'], test_size=0.3, shuffle = True, random_state=11)

# instantiate the model as clf(classifier) and train it
clf = MultinomialNB()
clf.fit(x_train, y_train)
pred0 = clf.predict(x_test)

#Performance measures
target_names = ["Person","Firm"]
print(confusion_matrix(y_true = y_test, y_pred = pred0))
print(classification_report(y_test, pred0, digits = 4, target_names = target_names))

##################################

#VALIDATION (RE-TESTING)
#Additional datasets for re-testing the model to check for overfitting
#Three datasets are created from the data not used for the construction of the model

Voters1 = Voters[int((len(Voters))*0.2):int((len(Voters))*0.5)] #Data is obtained from 20% of the voters up to 50%. 
Firms1 = Firms[int((len(Firms))*0.4):int((len(Firms))*0.6)] #Data is obtained from 40% of the firms up to 60%. 
Test1 = Voters1.append(Firms1, ignore_index = True)
print(Test1.shape)

Voters2 = Voters[int((len(Voters))*0.5):int((len(Voters))*0.75)] #Data is obtained from 60% of the voters up to 75%. 
Firms2 = Firms[int((len(Firms))*0.6):int((len(Firms))*0.8)] #Data is obtained from 60% of the firms up to 80%.
Test2 = Voters2.append(Firms2, ignore_index = True)
print(Test2.shape)

Voters3 = Voters[int((len(Voters))*0.75):] #Data is obtained from the last 25% of the voters
Firms3 = Firms[int((len(Firms))*0.8):] #Data is obtained from the last 25% of the firms
Test3 = Voters3.append(Firms3, ignore_index = True)
print(Test3.shape)


#Re-test 1
#Vectorizing inputs
Input1 = vectorizer.transform(Test1['Name'].values.astype('U'))

#Prediction
pred1 = clf.predict(Input1)

##Performance measures
target_names = ["Person","Firm"]
print(confusion_matrix(y_true = Test1['Class'], y_pred = pred1))
print(classification_report(Test1['Class'], pred1, digits = 4, target_names = target_names))


#Re-test 2
#Vectorizing inputs
Input2 = vectorizer.transform(Test2['Name'].values.astype('U'))

#Prediction
pred2 = clf.predict(Input2)

##Performance measures
target_names = ["Person","Firm"]
print(confusion_matrix(y_true = Test2['Class'], y_pred = pred2))
print(classification_report(Test2['Class'], pred2, digits = 4, target_names = target_names))



#Re-test 3
#Vectorizing inputs
Input3 = vectorizer.transform(Test3['Name'].values.astype('U'))

#Prediction
pred3 = clf.predict(Input3)

##Performance measures
target_names = ["Person","Firm"]
print(confusion_matrix(y_true = Test3['Class'], y_pred = pred3))
print(classification_report(Test3['Class'], pred3, digits = 4, target_names = target_names))


#Average accuracy score
ac0 = accuracy_score( y_test, pred0)
ac1 = accuracy_score( Test1['Class'], pred1)
ac2 = accuracy_score( Test2['Class'], pred2)
ac3 = accuracy_score( Test3['Class'], pred3)
ac = np.array((ac0, ac1, ac2, ac3))
print(np.average(ac))

###############################################

#CATEGORIZATION OF PPP NAMES
# Vectorizer to transform the document into word count vectors (Sparse)
word_mat2 = vectorizer.transform(PPP_names['stn_name'].values.astype('U'))
print((word_mat2.shape))

PPP_class = clf.predict(word_mat2)
PPP_names_wcs = pd.DataFrame()
PPP_names_wcs["stn_name"] = PPP_names["stn_name"]
PPP_names_wcs["Class"] = PPP_class

PPP_names_wcs["Class"] = PPP_names_wcs["Class"].replace(0,'Person')
PPP_names_wcs["Class"] = PPP_names_wcs["Class"].replace(1,'Firm')

print(PPP_names_wcs)

PPP_names_wcs.to_csv('PPP_names_classified.csv', index = False)


##############################################
#Word cloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate a word cloud image
wordcloud = WordCloud(width = 1000, height = 500,background_color="white").generate(" ".join(Voters0["Name"]))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Generate a word cloud image
wordcloud = WordCloud(width = 1000, height = 500,background_color="white").generate(" ".join(Firms0["Name"]))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Generate a word cloud image
wordcloud = WordCloud(width = 1000, height = 500,background_color="white").generate(" ".join(PPP_names_wcs[PPP_names_wcs["Class"] == "Person"]["stn_name"]))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Generate a word cloud image
wordcloud = WordCloud(width = 1000, height = 500,background_color="white").generate(" ".join(PPP_names_wcs[PPP_names_wcs["Class"] == "Firm"]["stn_name"]))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Generate a word cloud image
wordcloud = WordCloud(width = 1000, height = 500,background_color="white").generate(" ".join(PPP_names_wcs["stn_name"]))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

ala = " ".join(Firms0["Name"]))


