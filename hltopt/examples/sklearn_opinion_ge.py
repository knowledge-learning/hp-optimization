# coding: utf-8

import pprint
import random
import yaml

from nltk.corpus import movie_reviews, stopwords
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from ..ge import Grammar, PGE, Individual


class MyGrammar(Grammar):
    def __init__(self, sentences, classes):
        super().__init__()

        self.sentences = sentences[:200]
        self.classes = classes[:200]

    def grammar(self):
        return {
            'Pipeline': 'Prep Vect Red Class',
            'Prep'    : 'none | stopW',
            'Vect'    : 'TF | CV',
            'TF'      : 'i(1,2)',
            'CV'      : 'i(1,2)',
            'Red'     : 'none | svd',
            'Class'   : 'nb | LR | SVM',
            'LR'      : 'l1 Reg | l2 Reg',
            'Reg'     : 'f(0.01,10)',
            'SVM'     : 'linear | rbf'
        }

    def evaluate(self, i:Individual):
        # preprocesamiento
        if i.choose('none', 'stopW') == 'none':
            sw = None
        else:
            sw = stopwords.words('english')

        # vectorizador
        vect_cls = i.choose(TfidfVectorizer, CountVectorizer)
        n_gram = i.nextint()
        vect = vect_cls(stop_words=sw, ngram_range=(1,n_gram))

        # reductor
        reductor = i.choose(NoReductor(), TruncatedSVD(50))

        # clasificador
        clas = self._classifier(i)

        # evaluar
        X = vect.fit_transform(self.sentences)
        X = reductor.fit_transform(X)

        if isinstance(clas, GaussianNB) and hasattr(X, 'toarray'):
            X = X.toarray()

        score = 0
        n = 1
        for _ in range(n):
            X_train, X_test, y_train, y_test = train_test_split(X, self.classes, test_size=0.33)
            clas.fit(X_train, y_train)
            score += clas.score(X_test, y_test)

        score /= n
        print(score)

        return score

    def _classifier(self, i:Individual):
        # escoger entre SVM, NB y LR
        return i.choose(self._nb, self._lr, self._svm)(i)

    def _svm(self, i:Individual):
        return SVC(kernel=i.choose('linear', 'rbf'), gamma='scale')

    def _nb(self, i:Individual):
        return GaussianNB()

    def _lr(self, i:Individual):
        return LogisticRegression(penalty=i.choose('l1', 'l2'), C=i.nextfloat())


def load_corpus():
    sentences = []
    classes = []

    ids = list(movie_reviews.fileids())
    random.shuffle(ids)

    for fd in ids:
        if fd.startswith('neg/'):
            cls = 'neg'
        else:
            cls = 'pos'

        fp = movie_reviews.open(fd)
        sentences.append(fp.read())
        classes.append(cls)

    print("Sentences:", len(sentences))

    return sentences, classes


class NoReductor:
    def fit_transform(self, X):
        return X


def main():
    print("Loading corpus")
    grammar = MyGrammar(*load_corpus())

    print("Running heuristic")
    ge = PGE(grammar, popsize=100, selected=10, learning=0.25)
    ge.run(100)


if __name__ == '__main__':
    main()
