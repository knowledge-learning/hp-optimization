from nltk.corpus import movie_reviews, stopwords
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from ..pbil import PBIL


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


def vectorizer(tf_idf=True, remove_stopwords=True, n_grams=0):
    # Scale to (1-n)
    n_grams = n_grams + 1

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


reductors = [
    NoReductor(),
    TruncatedSVD(50),
    # These don't allow for sparse matrices
    # PCA(50),
    # FastICA(50),
]

classifiers = [
    LogisticRegression(penalty='l1'),
    LogisticRegression(penalty='l2'),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    GaussianNB(),
]


def main():
    genes = [
        2, # True or False for Tf-IDF
        2, # True or False for stopwords
        2, # 0, 1, ==> 1-gram, 2-gram
        len(reductors), # dimensionality reduction
        len(classifiers), # classifier
    ]

    sentences, classes = load_corpus()

    def fitness(solution):
        tf_idf, sw, n_gram, reductor, classifier = solution

        # Map to actual pipeline
        vect = vectorizer(tf_idf==1, sw==1, n_gram)
        red = reductors[reductor]
        clas = classifiers[classifier]

        X = vect.fit_transform(sentences)
        X = red.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2)
        clas.fit(X_train, y_train)
        score = clas.score(X_test, y_test)

        print(vect.__class__.__name__, red.__class__.__name__, clas.__class__.__name__, score)
        return score

    pbil = PBIL(100, 20, 0.1, genes, fitness)
    pbil.run(100)


if __name__ == '__main__':
    main()
