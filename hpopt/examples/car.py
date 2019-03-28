# coding: utf-8

import sys
import random
from hpopt.datasets.uci.car import load_corpus
from ..sklearn import SklearnClassifier, SklearnGrammar


def main():
    print("Loading corpus...", end="")
    X, y = load_corpus(representation='onehot')
    print(" done")

    random.seed(0)
    classifier = SklearnClassifier(popsize=100, select=20, iters=100, timeout=60, fitness_evaluations=10, verbose=True)
    classifier.fit(X, y)


if __name__ == "__main__":
    main()
