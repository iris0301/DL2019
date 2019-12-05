
import numpy as np
import re
import json
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download("words")
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
import csv



def split(word): 
    return list(word) 

def clean(text,emoji,stop_word):
    text = text.lower()
    
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()) 
    text = text.split()

    newtext = [x for x in text if not x.startswith('@') and not x.startswith('https')] ##remove @ and link
    newtext = [x for x in text if x not in stop_word] ## remove stopwords
    newtext = [WordNetLemmatizer().lemmatize(x,'v') for x in newtext] ##convert verb
    newtext = ' '.join(newtext)
    
    tokenizer = RegexpTokenizer(r'\w+')
    newtext = tokenizer.tokenize(newtext)
    
    stemmer = SnowballStemmer('english')
    newtext = list(map(stemmer.stem, newtext))
    emojis = split(emoji)
    newtext = newtext + emojis
    label = 0
    for ele in newtext:
        if ele =='amp':
            label = 1

    newtext = ' '.join(newtext)
    if label == 1:
        newtext = newtext.replace('amp','and')
   
    #print(newtext)
    return newtext

def padding(X, max_window):
    new_X = []
    for lis in X:
        new_X.append( lis + [0 for i in range(max_window-len(lis))] )
    return new_X

def get_data(file='smalldata.csv'):
    stop_word = [line.rstrip('\r\n') for line in open("stop_words.txt")]
    
    with open(file,encoding="utf8") as words_file:
        csv_reader = csv.DictReader(words_file, delimiter = ',')
        data = []
        for row in csv_reader:
            cleaned_row = []
            clean_text = clean(row['Text'],row['Emoji'],stop_word)
            
            if len(clean_text) == 0:
                continue
                
            if float(row['Sentiment']) > 0 and int(row['Sentiment140']) == 4:
                sentiment = 1 #'positive'
            elif float(row['Sentiment']) < 0 and int(row['Sentiment140']) == 0:
                sentiment = -1 #'negative'
            elif float(row['Sentiment']) == 0 and int(row['Sentiment140']) == 2:
                if float(row['Emoji_sentiment']) > 0:
                    sentiment = 1 #'positive'
                elif float(row['Emoji_sentiment']) < 0:
                    sentiment = -1 #'negative'
                else:
                    sentiment = 0 #'neutral'
            elif float(row['Sentiment']) == 0:
                if int(row['Sentiment140']) == 0:
                    sentiment = -1 #'negative'
                if int(row['Sentiment140']) == 4:
                    sentiment = 1 #'positive'
            elif int(row['Sentiment140']) == 2:
                if float(row['Sentiment']) < 0:
                    sentiment = -1 #'negative'
                if float(row['Sentiment']) > 0:
                    sentiment = 1 #'positive'
            else:
                continue
                
            
            cleaned_row.append(clean_text)
            cleaned_row.append(sentiment)
            data.append(np.array(cleaned_row))
        data = np.array(data)
    
    X = data[:,0]
    Y = data[:,1]
    Y = [int(label)+1 for label in Y]   # 0, 1, 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)
    
    word_dict = dict()
    word_id = 0
    for ele in X:
        ele = ele.split()
        for word in ele:
            if word not in word_dict:
                word_id += 1    # 0 for padding
                word_dict[word] = word_id

    max_window = 0  # maximum length of tweet      
    
    X_train_id = []
    
    i = 0
    for ele in X_train:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            i=i+1
            temp.append(word_dict[word])
        X_train_id.append(temp)
    X_test_id = []
    
    for ele in X_test:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            temp.append(word_dict[word])
        X_test_id.append(temp)
    #X_test_id = np.array(X_test_id)
    #X_train_id = np.array(X_train_id)
    
    '''
    X_train: training tweets
    X_test: testing tweets
    y_train: labels/sentiments of X_train
    y_test: labels/sentiments of X_test
    '''
    X_train_id = padding(X_train_id, max_window)
    X_test_id = padding(X_test_id, max_window)

    return X_train_id, X_test_id, y_train, y_test, word_dict



def data_noemoji(file='smalldata.csv'):
    stop_word = [line.rstrip('\r\n') for line in open("stop_words.txt")]
    
    with open(file,encoding="utf8") as words_file:
        csv_reader = csv.DictReader(words_file, delimiter = ',')
        data = []
        for row in csv_reader:
            cleaned_row = []
            clean_text = clean(row['Text'],'',stop_word)
            
            if len(clean_text) == 0:
                continue
                
            if float(row['Sentiment']) > 0 and int(row['Sentiment140']) == 4:
                sentiment = 1 #'positive'
            elif float(row['Sentiment']) < 0 and int(row['Sentiment140']) == 0:
                sentiment = -1 #'negative'
            elif float(row['Sentiment']) == 0 and int(row['Sentiment140']) == 2:
                sentiment = 0 #'neutral'
            elif float(row['Sentiment']) == 0:
                if int(row['Sentiment140']) == 0:
                    sentiment = -1 #'negative'
                if int(row['Sentiment140']) == 4:
                    sentiment = 1 #'positive'
            elif int(row['Sentiment140']) == 2:
                if float(row['Sentiment']) < 0:
                    sentiment = -1 #'negative'
                if float(row['Sentiment']) > 0:
                    sentiment = 1 #'positive'
            else:
                continue
                
            
            cleaned_row.append(clean_text)
            cleaned_row.append(sentiment)
            data.append(np.array(cleaned_row))
        data = np.array(data)
    
    X = data[:,0]
    Y = data[:,1]
    Y = [int(label)+1 for label in Y]   # 0, 1, 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)
    
    word_dict = dict()
    word_id = 0
    for ele in X:
        ele = ele.split()
        for word in ele:
            if word not in word_dict:
                word_id += 1    # 0 for padding
                word_dict[word] = word_id

    max_window = 0  # maximum length of tweet      
    
    X_train_id = []
    
    i = 0
    for ele in X_train:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            i=i+1
            temp.append(word_dict[word])
        X_train_id.append(temp)
    X_test_id = []
    
    for ele in X_test:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            temp.append(word_dict[word])
        X_test_id.append(temp)
    #X_test_id = np.array(X_test_id)
    #X_train_id = np.array(X_train_id)
    
    '''
    X_train: training tweets
    X_test: testing tweets
    y_train: labels/sentiments of X_train
    y_test: labels/sentiments of X_test
    '''
    X_train_id = padding(X_train_id, max_window)
    X_test_id = padding(X_test_id, max_window)

    return X_train_id, X_test_id, y_train, y_test, word_dict





def data_100k(file='tweets_100k.csv'):
    stop_word = [line.rstrip('\r\n') for line in open("stop_words.txt")]
    
    with open(file,encoding="utf8") as words_file:
        csv_reader = csv.DictReader(words_file, delimiter = ',')
        data = []
        for row in csv_reader:
            cleaned_row = []
            clean_text = clean(row['Text'],'',stop_word)
            
            if len(clean_text) == 0:
                continue
            
            cleaned_row.append(clean_text)
            cleaned_row.append(row['Sentiment'])
            data.append(np.array(cleaned_row))
        data = np.array(data)
    
    X = data[:,0]
    Y = data[:,1]
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)
    
    word_dict = dict()
    word_id = 0
    for ele in X:
        ele = ele.split()
        for word in ele:
            if word not in word_dict:
                word_id += 1    # 0 for padding
                word_dict[word] = word_id

    max_window = 0  # maximum length of tweet      
    
    X_train_id = []
    
    i = 0
    for ele in X_train:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            i=i+1
            temp.append(word_dict[word])
        X_train_id.append(temp)
    X_test_id = []
    
    for ele in X_test:
        ele = ele.split()
        max_window = max(max_window, len(ele))
        temp = []
        for word in ele:
            temp.append(word_dict[word])
        X_test_id.append(temp)
    #X_test_id = np.array(X_test_id)
    #X_train_id = np.array(X_train_id)
    
    '''
    X_train: training tweets
    X_test: testing tweets
    y_train: labels/sentiments of X_train
    y_test: labels/sentiments of X_test
    '''
    X_train_id = padding(X_train_id, max_window)
    X_test_id = padding(X_test_id, max_window)

    return X_train_id, X_test_id, y_train, y_test, word_dict


    