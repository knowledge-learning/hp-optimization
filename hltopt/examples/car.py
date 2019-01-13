# coding: utf-8

import random
from hltopt.datasets.uci.car import load_corpus
from ..sklearn import SklearnClassifier


def main():
    X, y = load_corpus(representation='onehot')

    print(X)

    random.seed(0)
    classifier = SklearnClassifier(popsize=100, select=20, iters=100, timeout=10)
    classifier.fit(X, y)


if __name__ == "__main__":
    main()
