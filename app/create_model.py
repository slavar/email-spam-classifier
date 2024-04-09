import math

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

#read sample dataset
df_tmp1 = pd.read_csv('./training_data/Spam Email raw text for NLP.csv')
df_tmp1.drop('FILE_NAME', axis=1, inplace=True)

df_tmp2 = pd.read_csv('./training_data/combined_data.csv')
df_tmp2 = df_tmp2.rename(columns={'label':'CATEGORY','text':'MESSAGE'})
df = pd.concat([df_tmp1, df_tmp2])
df.tail()
print(df.CATEGORY.value_counts())
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
nltk.download('omw-1.4')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

corpus=[]
for i in range(len(df.values)):
    # removing all non-alphanumeric characters
    message = re.sub('[^a-zA-Z0-9]', ' ', str(df.values[i][1]))

    # converting the message to lowercase
    message = message.lower()

    # splitting the sentence into words for lemmatization
    message = message.split()

    # removing stopwords and lemmatizing
    message = [lemmatizer.lemmatize(word) for word in message
               if word not in set(stopwords.words('english'))]

    # Converting the words back into sentences
    message = ' '.join(message)

    # Adding the preprocessed message to the corpus list
    corpus.append(message)

# Take top 2500 features
cv = CountVectorizer(max_features=2500, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
y = df['CATEGORY']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1, stratify=y)

tf = TfidfVectorizer(ngram_range=(1,3), max_features=2500)
X = tf.fit_transform(corpus).toarray()

model = MultinomialNB()
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

print(classification_report(train_pred, y_train))
print(classification_report(test_pred, y_test))

with open('./app/tf.pkl','wb') as f_tf:
    pickle.dump(tf,f_tf)

with open('./app/model.pkl','wb') as f_m:
    pickle.dump(model,f_m)

print('Predicting...')

message = ["You won 10000 dollars, please provide your account details,So that we can transfer the money"]

message_vector = tf.transform(message)
category = model.predict(message_vector)
print("The message is", "spam" if category == 1 else "not spam")




