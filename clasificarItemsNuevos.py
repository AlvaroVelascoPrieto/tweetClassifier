# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import datetime
import pickle
import unicodedata
import inflect
import pandas as pd
import re
from imblearn.over_sampling import RandomOverSampler
from nltk import WordNetLemmatizer, LancasterStemmer
from nltk.corpus import stopwords
import sklearn.tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import emoji
from sklearn.naive_bayes import CategoricalNB, MultinomialNB


def tfidf(processed_features):
    vectorizer = TfidfVectorizer(lowercase=False, max_features=1600, min_df=10, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()
    return processed_features, vectorizer


def remove_non_ascii(word):
    """Se eliminan todas las palabras que no este en formato ascii"""
    new_words = []
    new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Todas las letras se transforman en minuscula"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Se borra la puntuacion de las palabras"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Se convierten los numeros en su representacion con palabras"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Se eliminan las stopwords"""
    stop_words = set(stopwords.words('english'))
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(df_train):
    features = df_train['text'].values.tolist()
    labels = df_train['airline_sentiment'].values
    processed_features = []

    for words in range(0, len(features)):
        words = str(features[words])
        words = emoji.demojize((words), delimiters=("", ""))
        # words = words.split(" ")[1:-1]
        # words = ' '.join([str(elem) for elem in words])
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        words = remove_stopwords(words)
        words = lemmatize_verbs(words)
        # words = stem_words(words)
        words = ' '.join([str(elem) for elem in words])
        processed_features.append(words)
    print(processed_features)
    return labels, processed_features

model="mejorModelo.sav"
p="./"
NaiveBayes=False

# Press the green button in the gutter to run the script.
if __name__ == '__main__': 
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'pmfh:n',['path=','model=','testFile=','help','algorithm='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)
    #LLAMADA ORIGINAL: python clasificarItemsNuevos.py -m mejorModelo.sav -f TweetsTrainDev.csv
    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)
        elif opt in ('-n'):
            NaiveBayes = True
    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)
        

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    ml_dataset = pd.read_csv(iFile)
    labels, processed_features = normalize(ml_dataset)
    processed_features, vectorizador = tfidf(processed_features)
    testX = pd.DataFrame(processed_features, index=ml_dataset.index,
                                      columns=vectorizador.get_feature_names_out())
    if(NaiveBayes):
        testX['retweet_count'] = ml_dataset['retweet_count']
        testX['tweet_created'] = ml_dataset['tweet_created']
        for i in testX['tweet_created'].index:
            try:
                testX['tweet_created'][i] = datetime.datetime.strptime(testX['tweet_created'][i], '%Y-%m-%d %H:%M:%S %z').timestamp()
            except ValueError:
                processed_features.drop(i)
    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')

    print(results_test['predicted_value'])
    

    
