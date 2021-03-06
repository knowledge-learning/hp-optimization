# coding: utf-8

import numpy as np
import os


def load_corpus(red=False, white=False):
    if red == False and white == False:
        raise ValueError("Select either red or white or both.")

    path = os.path.dirname(os.path.abspath(__file__))

    f_white = open(os.path.join(path, "winequality-white.csv"), "r")
    f_red = open(os.path.join(path, "winequality-red.csv"), "r")

    X = []
    y = []

    title_line = True

    if white:
        for i in f_white.readlines():

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    title_line = True

    if red:
        for i in f_red.readlines():

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    return np.asarray(X), np.asarray(y)
