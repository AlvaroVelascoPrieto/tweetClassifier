import datetime
import sys
from time import sleep
import getopt
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

def guardar_modelo(clf):
    nombreModel = f"mejorModelo{algorithm}.sav"
    pickle.dump(clf, open(nombreModel, 'wb'))

if __name__ == '__main__':
    p = ''
    optNeg = False
    overSamplingBool = False
    algorithm = 'NaiveBayes'
    iFile = "TweetsTrainDev.csv"
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'hpfa:no',['help', 'path=','testFile=','algorithm='])
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
        elif opt in ('-a', '--algorithm'):
            algorithm = arg
        elif opt in ('-n'):
            optNeg = True
        elif opt in ('-o'):
            overSamplingBool = True
        elif opt in ('-h','--help'):
            print('')
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName')
            print(' -o use oversampling')
            print(' --algorithm algrithmToUse (NaiveBayes, DecisionTree)')
            print('')
            exit(1)
    
    if p == './':
        iFile = p+ str(f)
    else:
        #iFile = p+"/" + str(f) #No es usable en windows
        pass
    print("optNeg:", optNeg)
    print("overSamplingBool", overSamplingBool)
    print("algorithm:", algorithm)
    sleep(1)
    ######AQUI METER EL GETOPT Y ASIGNAR VALORES A LAS VARIABLES DE ENCIMA EN FUNCION DE LA LLAMADA
    ml_dataset = pd.read_csv(iFile)
    labels, processed_features = normalize(ml_dataset)
    processed_features, vectorizador = tfidf(processed_features)
    processed_features = pd.DataFrame(processed_features, index=ml_dataset.index,
                                      columns=vectorizador.get_feature_names_out())
    if (algorithm == 'NaiveBayes'):
        processed_features['retweet_count'] = ml_dataset['retweet_count']
        processed_features['tweet_created'] = ml_dataset['tweet_created']
        for i in processed_features['tweet_created'].index:
            try:
                processed_features['tweet_created'][i] = datetime.datetime.strptime(processed_features['tweet_created'][i],
                                                                                    '%Y-%m-%d %H:%M:%S %z').timestamp()
            except ValueError:
                processed_features.drop(i)
        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
        ros = RandomOverSampler()
        X_trainOver, y_trainOver = ros.fit_resample(X_train, y_train)
        X_testOver, y_testOver = ros.fit_resample(X_test, y_test)
        if(optNeg):
            clf = MultinomialNB(alpha=True)
        else:
            clf = MultinomialNB(alpha=True)
        if(overSamplingBool):
            clf.fit(X_trainOver, y_trainOver)
            predictions = clf.predict(X_testOver)
            guardar_modelo(clf)
            print(confusion_matrix(y_testOver, predictions))
            print(classification_report(y_testOver, predictions))
            print("Using oversampling\n")
            print("F1-Score:\t\t", f1_score(y_testOver, predictions, average=None)[0])
            print("Precision score:\t", precision_score(y_testOver, predictions, average=None)[0])
            print("Recall score:\t\t",recall_score(y_testOver, predictions, average=None)[0])
        else:
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            guardar_modelo(clf)
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            print("Without using oversampling\n")
            print("F1-Score:\t\t", f1_score(y_test, predictions, average=None)[0])
            print("Precision score:\t", precision_score(y_test, predictions, average=None)[0])
            print("Recall score:\t\t",recall_score(y_test, predictions, average=None)[0])
    elif(algorithm == 'DecisionTree'):
        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
        undersample = RandomOverSampler()
        trainXUnder, trainYUnder = undersample.fit_resample(X_train, y_train)
        testXUnder, testYUnder = undersample.fit_resample(X_test, y_test)
        if(optNeg):
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=18, min_samples_leaf=1,min_samples_split=2)
        else:
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=13, min_samples_leaf=1, min_samples_split=2)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        guardar_modelo(clf)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print(f1_score(y_test, predictions, average=None)[0])
        print(precision_score(y_test, predictions, average=None)[0])
        print(recall_score(y_test, predictions, average=None)[0])