from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import RussianStemmer
import re


def sanitize(text):
    """
    filter text for all non [0-9a-zA-Z] characters and stem words
    """
    stem = RussianStemmer()
    return ' '.join(stem.stem(word) for word in re.split(r'\s', re.sub(r'\W', ' ', text.lower())) if len(word) > 0)


def load_data(input_file):
    with open(input_file, encoding='utf8') as file:
        # train_file = file.read()
        print('Reading complete!')
        # separate data into topics and articles
        topics = []
        articles = []
        for article in file.readlines():
            topic, article = article.split('\t', maxsplit=1)
            topics.append(topic)
            articles.append(sanitize(article))
        print('Input complete!')
    return topics, articles


train_topics, train_articles = load_data('news/news_train.txt')

# vectorize data
vectorizer = TfidfVectorizer(max_features=30000, norm='l1').fit(train_articles)
vectorized_train = vectorizer.transform(train_articles)
print('Vectorising complete!')

# create a linear Support Vector classificator
# C=45 was found separately using gridsearch from sklearn.model_selection.GridSearchCV
clf = svm.LinearSVC(C=45, dual=False, verbose=True)
print(clf)
# train
clf.fit(vectorized_train, train_topics)
print('Training complete!')

# get data for prediction
with open("news/news_test.txt", encoding='utf8') as file:
    test_articles = [sanitize(article) for article in file.readlines()]
    vectorized_test = vectorizer.transform(test_articles)
# predict
prediction = clf.predict(vectorized_test)

# write results
with open('task1_output.txt', 'w+') as output:
    output.write('\n'.join(prediction))
