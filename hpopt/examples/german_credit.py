# coding: utf-8

import random
from hpopt.datasets.uci.german_credit import load_corpus
from ..sklearn import SklearnClassifier


def main():
    X, y = load_corpus()

    random.seed(0)
    classifier = SklearnClassifier(popsize=100, select=20, iters=100, fitness_evaluations=10, timeout=300, global_timeout=3600, verbose=True)
    classifier.fit(X, y)


if __name__ == "__main__":
    main()
