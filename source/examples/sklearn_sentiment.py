from nltk.corpus import movie_reviews, stopwords
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC


def load_corpus():
    sentences = []
    classes = []

    for fd in movie_reviews.fileids():
        if fd.startswith('neg/'):
            cls = 'neg'
        else:
            cls = 'pos'

        for line in movie_reviews.open(fd):
            sentences.append(line)
            classes.append(cls)

    return sentences, classes


def fitness_function(solution):
    pass


def vectorizer(sentences, tf_idf=True, remove_stopwords=True, n_grams=1):
    if remove_stopwords:
        sw = stopwords.words('english')
    else:
        sw = None

    if tf_idf:
        Vect = TfidfVectorizer
    else:
        Vect = CountVectorizer

    return Vect(ngram_range=(1, n_grams), stop_words=sw)




class NoReductor:
    def fit_transform(self, X):
        return X


def reductor(selection):
    reductors = [
        NoReductor(),
        PCA(50),
        TruncatedSVD(50),
        FastICA(50),
    ]

    return reductors[selection]




def classifier(selection):
    classifiers = [
        LogisticRegression(penalty='l1'),
        LogisticRegression(penalty='l2'),
        GaussianNB(),
        MultinomialNB(),
        SVC(kernel='linear'),
        SVC(kernel='rbf'),
    ]

    return classifiers[selection]
