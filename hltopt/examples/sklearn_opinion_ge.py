# coding: utf-8

import yaml
import pprint
import random

from nltk.corpus import movie_reviews, stopwords
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from ..ge import GrammarGE, GE, Individual


class MyGrammar(GrammarGE):
    def __init__(self, sentences, classes):
        self.sentences = sentences#[:200]
        self.classes = classes#[:200]

    def grammar(self):
        return {
            'Pipeline': 'Prep Vect Red Class',
            'Prep': 'none | stopW',
            'Vect': 'Tf | CV',
            'Tf': 'i(1,1)',
            'CV': 'i(1,1)',
            'Red': 'none | svd',
            'Class': 'nb | LR | SVM',
            'LR': 'l1 C | l2 C',
            'C': 'f(0.001,10)',
            'SVM': 'linear | rbf'
        }

    def evaluate(self, i:Individual):
        # preprocesamiento
        if i.nextbool():
            sw = stopwords.words('english')
        else:
            sw = None

        # vectorizador
        vect_cls = i.choose(TfidfVectorizer, CountVectorizer)
        n_gram = i.nextint(1) + 1
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
            X_train, X_test, y_train, y_test = train_test_split(X, self.classes, test_size=0.2)
            clas.fit(X_train, y_train)
            score += clas.score(X_test, y_test)

        score /= n
        print(score)

        return score

    def _classifier(self, i:Individual):
        # escoger entre SVM, NB y LR
        return i.choose(self._svm, self._lr, self._nb)(i)

    def _svm(self, i:Individual):
        return SVC(kernel=i.choose('linear', 'rbf'))

    def _nb(self, i:Individual):
        return GaussianNB()

    def _lr(self, i:Individual):
        return LogisticRegression(penalty=i.choose('l1', 'l2'), C=i.nextfloat(0.001, 10))


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

        for line in movie_reviews.open(fd):
            sentences.append(line)
            classes.append(cls)

    return sentences, classes


class NoReductor:
    def fit_transform(self, X):
        return X


def main():
    grammar = MyGrammar(*load_corpus())

    print(yaml.dump(grammar.parse()))
    print(grammar.complexity())

    ge = GE(grammar, popsize=10, selected=5)
    ge.run(100)


if __name__ == '__main__':
    main()
