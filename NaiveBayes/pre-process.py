import nltk
import pickle
import re
import os
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn import preprocessing


def get_data(root = './data'):
    X_train = []
    y_train = []
    labels = os.listdir(root)
    # print(labels)
    for label in labels:
        list_file = os.listdir(os.path.join(root, label))
        for file in list_file:
            X = ""
            # print(file, label)
            with open(os.path.join(root, label, file), 'r', encoding= 'unicode_escape') as f_r:
                data = f_r.read().splitlines()
            
            for content in data:
                if content == '':
                    continue
                X += content + " "
            
            X_train.append(X)
            y_train.append(label)

    return X_train, y_train

def get_data_csv(root = './data_crawled.csv'):
    data = pd.read_csv(root)
    X_test = []
    y_test = []
    label = {1:'bussiness', 2 : 'entertainment', 3:'politics', 4:'sport', 5:'tech'}
    contents = data['content']
    category = data['category']
    for i in range(len(contents)):
        X_test.append(contents[i])
        y_test.append(label[int(category[i])])
    
    return X_test, y_test


class Pre_process:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.stopwords = set(stopwords.words('english'))

    def _lowercase(self, input_str):
        return input_str.lower()

    def _remove_numbers(self, input_str):
        return re.sub(r'\d+', '', input_str)
    
    def _remove_punctuation(self, input_str):
        return input_str.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_whitespaces(self, input_str):
        return input_str.strip()

    def _stemming(self, input_str):
        output_str = ''
        stemmer = PorterStemmer()
        input_str = word_tokenize(input_str)

        for word in input_str:
            output_str += stemmer.stem(word) + ' '
        
        return output_str 

    def _lemmatization(self, input_str):
        output_str = ''
        lemmatizer = WordNetLemmatizer()
        input_str = word_tokenize(input_str)

        for word in input_str:
            output_str += lemmatizer.lemmatize(word) + ' '
        return output_str
        
    def _remove_stopwords(self, input_str):
        output_str = ''
        input_str = word_tokenize(input_str)
        for word in input_str:
            if word not in self.stopwords:
                output_str += word + ' '
        return output_str
    
    def pre_process_data(self, train = True):
        if train == True:
            for i in tqdm(range(len(self.X_train))):
                self.X_train[i] = self._remove_punctuation(self.X_train[i])
                self.X_train[i] = self._lowercase(self.X_train[i])
                self.X_train[i] = self._remove_numbers(self.X_train[i])
                self.X_train[i] = self._remove_whitespaces(self.X_train[i])
                self.X_train[i] = self._stemming(self.X_train[i])
                self.X_train[i] = self._lemmatization(self.X_train[i])
                self.X_train[i] = self._remove_stopwords(self.X_train[i])
            pickle.dump(self.X_train, open('./X_train.pkl', 'wb'))
        else:
            for i in tqdm(range(len(self.X_test))):
                self.X_test[i] = self._remove_punctuation(self.X_test[i])
                self.X_test[i] = self._lowercase(self.X_test[i])
                self.X_test[i] = self._remove_numbers(self.X_test[i])
                self.X_test[i] = self._remove_whitespaces(self.X_test[i])
                self.X_test[i] = self._stemming(self.X_test[i])
                self.X_test[i] = self._lemmatization(self.X_test[i])
                self.X_test[i] = self._remove_stopwords(self.X_test[i])
            pickle.dump(self.X_test, open('./X_test.pkl', 'wb'))

    def _label_encoder(self):
        encoder = preprocessing.LabelEncoder()
        self.y_train = encoder.fit_transform(self.y_train)  
        self.y_test = encoder.fit_transform(self.y_test)
        print(encoder.classes_)
        pickle.dump(self.y_train, open('./y_train.pkl', 'wb'))
        pickle.dump(self.y_test, open('./y_test.pkl', 'wb'))

    def run(self):
        print('Pre-processing train data ...')
        self.pre_process_data()
        print('Pre-processing test data ...')
        self.pre_process_data(train=False)
        self._label_encoder()


if __name__ == '__main__':
    X_train, y_train = get_data()
    X_test, y_test = get_data_csv()
    pre_process = Pre_process(X_train, y_train, X_test, y_test)
    pre_process.run()
    