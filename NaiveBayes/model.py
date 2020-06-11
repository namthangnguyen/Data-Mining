import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing


X_train = pickle.load(open('./X_test.pkl', 'rb'))
y_train = pickle.load(open('./y_test.pkl', 'rb'))
X_test = pickle.load(open('./X_train.pkl', 'rb'))
y_test = pickle.load(open('./y_train.pkl', 'rb'))


print("Word embedding ...")
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(1, 3))
tfidf_vect_ngram.fit(X_train)
X_train = tfidf_vect_ngram.transform(X_train)
X_test = tfidf_vect_ngram.transform(X_test)

print('Training ...')
model = MultinomialNB()
model.fit(X_train, y_train)
predictions_test = model.predict(X_test)

print('Results validation ===========================')
print('Accuracy score:  ', accuracy_score(y_test, predictions_test))
print('Precision score: ', precision_score(y_test, predictions_test, average = None))
print('Recall score:    ', recall_score(y_test, predictions_test, average = None))


# Precision score:  [0.78868552 0.9640884  0.9009009  0.96353167 0.92647059]
# Recall score:     [0.92941176 0.90414508 0.71942446 0.98238748 0.94264339]