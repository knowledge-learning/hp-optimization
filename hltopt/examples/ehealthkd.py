# coding: utf-8

import functools
import pprint
import bisect
import os
import pprint
import random
import string
import nltk
import spacy
import unicodedata
import gensim
import yaml
import numpy as np

from scipy import sparse as sp
from pathlib import Path

from sklearn_crfsuite.estimator import CRF
from seqlearn.hmm import MultinomialHMM
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from gensim.models import Word2Vec
import gensim.downloader as api

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Input, concatenate

from ..ge import Grammar, PGE, Individual, InvalidPipeline
from ..datasets.ehealthkd import TassDataset

from ..utils import szip, sdiv


class Token:
    def __init__(self, text: str, init: int, norm: str = None, pos: str = None, tag: str = None, dep: str = None, vector=None):
        self.text = text
        self.init = init
        self.end = init + len(text)
        self.norm = norm or text
        self.pos = pos
        self.tag = tag
        self.dep = dep
        self.vector = vector

    def __and__(self, other):
        return max(0, min(self.end, other.pos_end) - max(self.init, other.pos_init)) > 0

    def __repr__(self):
        return repr(self.__dict__)


class TassGrammar(Grammar):
    def __init__(self):
        super().__init__()

        self.stemmer = SnowballStemmer("spanish")
        self.spacy_nlp = spacy.load('es')

    def grammar(self):
        return {
            'Pipeline' : 'Repr A B C | Repr AB C | Repr A BC | Repr ABC',
            'ABC'      : 'Class',
            'BC'       : 'Class',
            'AB'       : 'Class | Seq',
            'A'        : 'Class | Seq',
            'B'        : 'Class',
            'C'        : 'Class',

            # Sequence algorithms
            'Seq'      : 'HMM', # | crf',
            'HMM'      : 'HMMdec HMMalp',
            'HMMdec'   : 'viterbi | bestfirst',
            'HMMalp'   : 'f(0.01, 10)',

            # Classifiers
            'Class'    : 'LR | nb | SVM | dt | NN',

            # Classic classifiers
            'LR'       : 'Reg Penalty',
            'Reg'      : 'f(0.01,100)',
            'Penalty'  : 'l1 | l2',
            'SVM'      : 'Kernel',
            'Kernel'   : 'linear | rbf | poly',

            # Generic neural networks
            'NN'       : 'Drop CVLayers DLayers FLayer | Drop RLayers DLayers FLayer | Drop DLayers FLayer',
            'Drop'     : 'f(0.1,0.5)',
            # Convolutional layers
            'CVLayers' : 'Count MinFilter MaxFilter FormatCon',
            'Count'    : 'i(1,5)',
            # Con el objetivo de eliminar la recursividad de la gramática y controlar el tamaño de las capas
            # se define un espacio genérico donde no se define cada capa de la red, sino que se define una
            # arquitectura de alto nivel que restringe las posibles redes. No aparecen representadas todas
            # las posiblidades pero este subconjunto de ellas es semáticamente más interesante.
            'MinFilter': 'i(1,5)',
            'MaxFilter': 'i(1,5)',
            # Todos los filtros con el mismo tamaño, un filtro para cada tamaño y tamaños de filtros aleatorios
            'FormatCon': 'same | all | rand',
            # Recursive layers
            'RLayers'  : 'Size',
            'Size'     : 'i(10,100)',
            # Dense layers
            'DLayers'  : 'Count MaxSize MinSize FormatDen Act',
            'Act'      : 'sigmoid | relu | tanh',
            'MinSize'  : 'i(10,100)',
            'MaxSize'  : 'i(10,100)',
            # las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
            'FormatDen': 'grow | shrink | same',
            # Final layer
            'FLayer'   : 'sigmoid | softmax',

            # Text representation
            'Repr'     : 'Token Prep SemFeat PosPrep MulWords Embed',
            'Token'    : 'wordTok',
            'Prep'     : 'DelPunt StripAcc',
            'DelPunt'  : 'yes | no',
            'StripAcc' : 'yes | no',
            'PosPrep'  : 'StopW Stem',
            'Stem'     : 'yes | no',
            'StopW'    : 'yes | no',
            'SemFeat'  : 'PosTag Dep UMLS SNOMED',
            'PosTag'   : 'yes | no',
            'Dep'      : 'yes | no',
            'UMLS'     : 'yes | no',
            'SNOMED'   : 'yes | no',
            'MulWords' : 'countPhrase | freeling | Ngram',
            'Ngram'    : 'i(2,4)',
            'Embed'    : 'wordVec | onehot | none',
        }

    def evaluate(self, ind:Individual):
        FAST = True
        TEST = False

        # load training data
        dataset_path = Path.cwd() / 'hltopt' / 'datasets' / 'ehealthkd'
        dataset = TassDataset()

        for file in (dataset_path / 'training').iterdir():
            if file.name.startswith('input'):
                dataset.load(file)

                if FAST and len(dataset.texts) >= 100:
                    break

        if FAST:
            dataset.validation_size = int(0.2 * len(dataset.texts))
        else:
            validation = dataset_path / 'develop' / 'input_develop.txt'
            dataset.validation_size = dataset.load(validation)

            if TEST:
                test = dataset_path / 'test' / 'input_scenario1.txt'
                dataset.validation_size = dataset.load(test)

        return self._pipeline(ind, dataset)

    def _pipeline(self, ind, dataset):
        # 'Pipeline' : 'Repr A B C | Repr AB C |  Repr A BC | Repr ABC',
        choice = ind.choose('A B C', 'AB C', 'A BC', 'ABC')

        self._repr(ind, dataset)

        try:
            if choice == 'A B C':
                # Ejecutar tareas A, B y C en secuencia
                result_A = self._a(ind, dataset)
                val_labels = self._b(ind, dataset, result_A)
                val_relations = self._c(ind, dataset, val_labels)
            elif choice == 'AB C':
                # Ejecutar tareas AB juntas y C en secuencia
                val_labels = self._ab(ind, dataset)
                val_relations = self._c(ind, dataset, val_labels)
            elif choice == 'A BC':
                # Ejecutar tarea A y luego BC
                results_A = self._a(ind, dataset)
                val_labels, val_relations = self._bc(ind, dataset, results_A)
            else:
                # Ejecutar Tarea ABC junta
                val_labels, val_relations = self._abc(ind, dataset)

            return self._score(dataset.dev_labels, val_labels, dataset.dev_relations, val_relations)
        except ValueError as e:
            if 'must be non-negative' in str(e):
                raise InvalidPipeline(str(e))
            else:
                raise e

    def _score(self, train_labels, val_labels, train_relations, val_relations):
        assert len(train_labels) == len(val_labels)
        assert len(train_relations) == len(val_relations)

        # score counts
        correctA = 0
        partialA = 0
        missingA = 0
        spuriousA = 0

        correctB = 0
        incorrectB = 0

        correctC = 0
        missingC = 0
        spuriousC = 0

        #label maps
        t2v = {}
        v2t = {}

        for l1, l2 in szip(train_labels, val_labels):
            for start, end in l1:
                if (start, end) in l2:
                    t2v[l1[(start, end)][0]] = l2[(start,end)][0]
                    v2t[l2[(start, end)][0]] = l1[(start,end)][0]

                    correctA += 1
                    if l1[(start, end)][1] == l2[(start, end)][1]:
                        correctB += 1
                    else:
                        incorrectB += 1
                else:
                    missingA += 1

            for start, end in l2:
                if not (start, end) in l1:
                    spuriousA += 1

        for r1, r2 in szip(train_relations, val_relations):
            for r,to,td in r1:
                if not to in t2v or not td in t2v:
                    missingC += 1
                    continue

                vo = t2v[to]
                vd = t2v[td]

                if (r,vo,vd) in r2:
                    correctC += 1
                else:
                    missingC += 1

        for r1, r2 in szip(train_relations, val_relations):
            for r,vo,vd in r2:
                if not vo in v2t or not vd in v2t:
                    spuriousC += 1
                    continue

                to = v2t[vo]
                td = v2t[vd]

                if (r,to,td) not in r1:
                    spuriousC += 1

        print("[*] Task A: %0.2f" % sdiv(correctA, correctA + missingA + spuriousA))
        print("[*] Task B: %0.2f" % sdiv(correctB, correctB + incorrectB))
        print("[*] Task C: %0.2f" % sdiv(correctC, correctC + missingC + spuriousC))

        top = (correctA + 0.5 * partialA + correctB + correctC)
        spr = (correctA + partialA + correctB + incorrectB + spuriousA + correctC + spuriousC)
        precision = sdiv(top, spr)
        msn = (correctA + partialA + correctB + incorrectB + missingA + correctC + missingC)
        recall = sdiv(top, msn)

        return sdiv(2 * precision * recall, precision + recall)

    def _abc(self, ind, dataset):
        # calcular la forma de la entrada
        _, cols = dataset.vectors[0].shape
        intput_shape = cols
        output_shape = 10

        clss, clss_type = self._class(ind, intput_shape, output_shape)

        if clss_type == 'seq':
            xtrain, ytrain, xdev, mapping = dataset.task_abc_by_sentence()
        else:
            xtrain, ytrain, xdev, mapping = dataset.task_abc_by_word()
            clss = OneVsRestClassifier(clss)
            xtrain = np.vstack(xtrain)
            ytrain = np.vstack(ytrain)

        clss.fit(xtrain, ytrain)

        predictions = [clss.predict(x) for x in xdev]

        # Ids
        ids = 0
        val_labels = []
        for sent in dataset.dev_tokens:
            sentence_labels = {}
            for tok in sent:
                sentence_labels[(tok.init, tok.end)] = (ids, {'Concept':0, 'Action':0, 'None':0})
                ids += 1
            val_labels.append(sentence_labels)

        # labels map
        bc_labels_map = {
            (0,0): 'None',
            (1,0): 'Concept',
            (0,1): 'Action',
            (1,1): 'None'
        }

        # Labels
        for lbl, resC, mapC in szip(val_labels, predictions, mapping):
            for rels, (org, dest) in szip(resC, mapC):
                if org not in lbl or dest not in lbl:
                    continue
                orgid, orglbs = lbl[org]
                destid, destlbs = lbl[dest]

                orglbl = bc_labels_map[tuple(rels[-4:-2])]
                destlbl = bc_labels_map[tuple(rels[-2:])]

                orglbs[orglbl] += 1
                destlbs[destlbl] += 1

        val_labels = [{ tok: (ids, max(lbl, key=lbl.get)) for tok, (ids,lbl) in sent.items()} for sent in val_labels]
        val_labels = [{ tok: (ids, lbl) for tok, (ids,lbl) in sent.items() if lbl != 'None'} for sent in val_labels]

        # Relations
        val_relations = []
        for lbl, resC, mapC in szip(val_labels, predictions, mapping):
            sentence_rels = []
            for rels, (org, dest) in szip(resC, mapC):
                if org not in lbl or dest not in lbl:
                    continue
                orgid, orglb = lbl[org]
                destid, destlb = lbl[dest]

                rels = dataset.relmap.inverse_transform(rels[:-4].reshape(1,-1))[0]
                for r in rels:
                    if r in ['subject', 'target']:
                        if orglb == 'Action' and destlb == 'Concept':
                            sentence_rels.append((r, orgid, destid))
                    else:
                        if orglb == 'Concept' and destlb == 'Concept':
                            sentence_rels.append((r, orgid, destid))

            val_relations.append(sentence_rels)

        return val_labels, val_relations

    def _bc(self, ind, dataset, results_A):
        # calcular la forma de la entrada
        _, cols = dataset.vectors[0].shape
        intput_shape = cols
        output_shape = 10

        clss, clss_type = self._class(ind, intput_shape, output_shape)

        if clss_type == 'seq':
            xtrain, ytrain, xdev, mapping = dataset.task_bc_by_sentence()
        else:
            xtrain, ytrain, xdev, mapping = dataset.task_bc_by_word()
            clss = OneVsRestClassifier(clss)
            xtrain = np.vstack(xtrain)
            ytrain = np.vstack(ytrain)

        clss.fit(xtrain, ytrain)

        predictions = [clss.predict(x) for x in xdev]

        # Labels
        ids = 0
        val_labels = []
        for sent, resA, resB in szip(dataset.dev_tokens, results_A, predictions):
            sentence_labels = {}
            for tok, clsA in szip(sent, resA):
                if clsA:
                    sentence_labels[(tok.init, tok.end)] = (ids, {'Concept':0, 'Action':0})
                    ids += 1
            val_labels.append(sentence_labels)

        # compute actual labels
        # labels map
        bc_labels_map = {
            (0,0): 'None',
            (1,1): 'None',
            (1,0): 'Concept',
            (0,1):  'Action'
        }

        for lbl, resC, mapC in szip(val_labels, predictions, mapping):
            for rels, (org, dest) in szip(resC, mapC):
                if org not in lbl or dest not in lbl:
                    continue
                orgid, orglbs = lbl[org]
                destid, destlbs = lbl[dest]

                orglbl = bc_labels_map[tuple(rels[-4:-2])]
                destlbl = bc_labels_map[tuple(rels[-2:])]

                if orglbl in orglbs:
                    orglbs[orglbl] += 1
                if destlbl in  destlbs:
                    destlbs[destlbl] += 1

        val_labels = [{ tok: (ids, max(lbl, key=lbl.get)) for tok, (ids,lbl) in sent.items()} for sent in val_labels]

        # # compute relations
        val_relations = []
        for lbl, resC, mapC in szip(val_labels, predictions, mapping):
            sentence_rels = []
            for rels, (org, dest) in szip(resC, mapC):
                if org not in lbl or dest not in lbl:
                    continue
                orgid, orglb = lbl[org]
                destid, destlb = lbl[dest]

                rels = dataset.relmap.inverse_transform(rels[:-4].reshape(1,-1))[0]
                for r in rels:
                    if r in ['subject', 'target']:
                        if orglb == 'Action' and destlb == 'Concept':
                            sentence_rels.append((r, orgid, destid))
                    else:
                        if orglb == 'Concept' and destlb == 'Concept':
                            sentence_rels.append((r, orgid, destid))

            val_relations.append(sentence_rels)

        return val_labels, val_relations

    def _ab(self, ind, dataset):
        choice = ind.choose('class', 'seq')

        if choice == 'class':
            # classifier

            # calcular la forma de la entrada
            _, cols = dataset.vectors[0].shape
            intput_shape = cols

            clss, clss_type = self._class(ind, intput_shape, 3)

            if clss_type == 'seq':
                xtrain, ytrain, xdev = dataset.task_ab_by_sentence()
            else:
                xtrain, ytrain, xdev = dataset.task_ab_by_word()
                xtrain = np.vstack(xtrain)
                ytrain = np.hstack(ytrain)

            clss.fit(xtrain, ytrain)

            # construir la entrada dev
            prediction = [clss.predict(x) for x in xdev]
        else:
            # sequence classifier
            xtrain, ytrain, xdev = dataset.task_a_by_word()
            prediction = self._hmm(ind, xtrain, ytrain, xdev)

        # Labels
        ids = 0
        val_labels = []
        for sent, resAB in szip(dataset.dev_tokens, prediction):
            sentence_labels = {}
            for tok, clsAB in szip(sent, resAB):
                if clsAB:
                    sentence_labels[(tok.init, tok.end)] = (ids, clsAB)
                    ids += 1
            val_labels.append(sentence_labels)

        return val_labels

    def _a(self, ind:Individual, dataset:TassDataset):
        choice = ind.choose('class', 'seq')

        if choice == 'class':
            # classifier

            # calcular la forma de la entrada
            _, cols = dataset.vectors[0].shape
            intput_shape = cols

            clss, clss_type = self._class(ind, dataset, intput_shape, 1)

            if clss_type == 'seq':
                xtrain, ytrain, xdev = dataset.task_a_by_sentence()
                xtrain = np.asarray(xtrain)
                ytrain = np.asarray(ytrain)

                clss.fit(xtrain, ytrain, epochs=10)

                prediction = [clss.predict(x) for x in xdev]
                prediction = [(x > 0.5).astype(int) for x in prediction]
            else:
                xtrain, ytrain, xdev = dataset.task_a_by_word()
                xtrain = np.vstack(xtrain)
                ytrain = np.hstack(ytrain)

                if isinstance(clss, Model):
                    clss.fit(xtrain, ytrain, epochs=100)
                else:
                    clss.fit(xtrain, ytrain)

                prediction = [clss.predict(x) for x in xdev]
                prediction = [(x > 0.5).astype(int) for x in prediction]
        else:
            # sequence classifier
            xtrain, ytrain, xdev = dataset.task_a_by_word()
            prediction = self._hmm(ind, xtrain, ytrain, xdev)

        results = []
        ids = 0

        for sentence, pred in szip(dataset.dev_tokens, prediction):
            new_sentence = []
            for token, r in szip(sentence, pred):
                ids += 1
                new_sentence.append((token.init, token.end, r==1, ids))
            results.append(new_sentence)

        return results

    def _hmm(self, ind:Individual, xtrain, ytrain, xdev):
        lengths = [x.shape[0] for x in xtrain]

        xtrain = np.vstack(xtrain).astype(int)
        ytrain = np.hstack(ytrain)

        try:
            hmm = MultinomialHMM(decode=ind.choose('viterbi', 'bestfirst'), alpha=ind.nextfloat())
            hmm.fit(xtrain, ytrain, lengths)
            xdev = [x.astype(int) for x in xdev]
            return [hmm.predict(x) for x in xdev]
        except ValueError as e:
            if 'non-negative integers' in str(e):
                raise InvalidPipeline(str(e))
            elif 'unknown categories' in str(e):
                raise InvalidPipeline(str(e))
            else:
                raise

    def _crf(self, ind:Individual, xtrain, ytrain, xdev):
        raise NotImplementedError()

        crf = CRF()
        crf.fit(xtrain, ytrain)
        return [crf.predict(x) for x in xdev]

    def _b(self, ind:Individual, dataset:TassDataset, result_A):
        # compute input shape (for neural networks)
        _, cols = dataset.vectors[0].shape
        intput_shape = cols

        clss, clss_type = self._class(ind, dataset, intput_shape, 3)

        if clss_type == 'seq':
            xtrain, ytrain, xdev = dataset.task_b_by_sentence()
            xtrain = np.asarray(xtrain)
            ytrain = np.asarray(ytrain)

            clss.fit(xtrain, ytrain, epochs=10)

            prediction = [clss.predict(x) for x in xdev]
            print(prediction)

        else:
            xtrain, ytrain, xdev = dataset.task_b_by_word()
            xtrain = np.vstack(xtrain)
            ytrain = np.hstack(ytrain)

            clss.fit(xtrain, ytrain)

            # construir la entrada dev
            prediction = [clss.predict(x) for x in xdev]

        results_B = []

        for sentence, pred in szip(result_A, prediction):
            new_sentence = {}
            for (start, end, kw, ids), lbl in szip(sentence, pred):
                if kw:
                    new_sentence[(start, end)] = (ids, lbl)
            results_B.append(new_sentence)

        return results_B

    def _c(self, ind:Individual, dataset, val_labels):
        # compute input shape (for neural networks)
        _, cols = dataset.vectors[0].shape
        intput_shape = cols

        clss, clss_type = self._class(ind, intput_shape, 6)

        if clss_type == 'seq':
            xtrain, ytrain, xdev, mapping = dataset.task_c_by_sentence()
        else:
            clss = OneVsRestClassifier(clss)
            xtrain, ytrain, xdev, mapping = dataset.task_c_by_word()
            xtrain = np.vstack(xtrain)
            ytrain = np.vstack(ytrain)

        clss.fit(xtrain, ytrain)

        # construir la entrada dev
        prediction = [clss.predict(x) for x in xdev]
        val_relations = []

        for lbl, resC, mapC in szip(val_labels, prediction, mapping):
                sentence_rels = []
                for rels, (org, dest) in szip(resC, mapC):
                    if org not in lbl or dest not in lbl:
                        continue
                    orgid, orglb = lbl[org]
                    destid, destlb = lbl[dest]

                    rels = dataset.relmap.inverse_transform(rels.reshape(1,-1))[0]
                    for r in rels:
                        if r in ['subject', 'target']:
                            if orglb == 'Action' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))
                        else:
                            if orglb == 'Concept' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))

                val_relations.append(sentence_rels)

        return val_relations

    def _class(self, ind:Individual, dataset, input_shape=None, output_shape=None):
        #LR | nb | SVM | dt | NN
        des = ind.choose('lr', 'nb', 'svm', 'dt', 'nn')
        clss = None

        if des == 'lr':
            clss = self._lr(ind)
        elif des == 'nb':
            clss = MultinomialNB()
        elif des == 'svm':
            clss = self._svm(ind)
        elif des == 'dt':
            clss = DecisionTreeClassifier()
        else:
            return self._nn(ind, dataset, input_shape, output_shape)

        return clss, 'word'

    def _lr(self, i):
        return LogisticRegression(C=self._reg(i), penalty=self._penalty(i))

    def _reg(self, i):
        return i.nextfloat()

    def _penalty(self, i):
        return i.choose('l1', 'l2')

    def _svm(self, i):
        return SVC(kernel=self._kernel(i))

    def _kernel(self, i):
        #linear | rbf | poly
        return i.choose('linear', 'rbf', 'poly')

    def _nn(self, i, dataset, input_size, output_size):
        try:
            # CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop
            model = Sequential()
            option = i.choose('conv', 'rec', 'deep')

            dropout = self._drop(i)
            clss_type = 'seq'

            if option == 'conv':
                x = Input(shape=(dataset.max_length, input_size+1))
                y = self._cvlayers(i, x, dropout)
                y = self._dlayers(i, y, dropout)
                y = self._flayer(i, y, output_size, dropout)
            elif option == 'rec':
                x = Input(shape=(dataset.max_length, input_size+1))
                y = self._rlayers(i, x, dropout)
                y = self._dlayers(i, y, dropout)
                y = self._flayer(i, y, output_size, dropout)
            else:
                clss_type = 'word'
                x = Input(shape=(input_size,))
                y = self._dlayers(i, x, dropout)
                y = self._flayer(i, y, output_size, dropout)

            model = Model(inputs=x, outputs=y)

            if output_size == 1:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'

            model.compile(optimizer='adam', loss=loss)

            return model, clss_type
        except ValueError as e:
            msg = str(e)
            if 'out of bounds' in msg:
                raise InvalidPipeline('Bad NN architecture')
            if 'Negative dimension' in msg:
                raise InvalidPipeline('Bad NN architecture')
            else:
                raise e

    def _drop(self, i):
        return i.nextfloat()

    def _cvlayers(self, i, model, dropout):
        # 'CVLayers' : 'Count MinFilter MaxFilter FormatCon',
        count = self._count(i)
        minfilter = self._minfilter(i)
        maxfilter = max(minfilter, self._maxfilter(i))
        formatcon = self._formatcon(i)

        for _ in range(count):
            layers = []
            for kernel in range(minfilter, maxfilter + 1):
                if formatcon == 'all':
                    kernel_size = 2*kernel+1
                elif formatcon == 'same':
                    kernel_size = 2*minfilter+1
                else:
                    kernel_size = random.randint(2*minfilter+1, 2*maxfilter+2)

                layer = Conv1D(filters=1, kernel_size=kernel_size)(model)
                drop = Dropout(dropout)(layer)
                pool = MaxPooling1D(3)(drop)
                layers.append(pool)

            if len(layers) > 1:
                model = concatenate(layers)
            else:
                model = layers[0]

        return model

    def _count(self, i):
        return i.nextint() + 1

    def _minfilter(self, i):
        return i.nextint()

    def _maxfilter(self, i):
        return i.nextint()

    def _formatcon(self, i):
        return i.choose('same', 'all', 'rand')

    def _rlayers(self, i, model, dropout):
        # 'RLayers'  : 'Size',
        size = self._size(i)
        lstm = LSTM(size, dropout=dropout, recurrent_dropout=dropout)(model)
        return lstm

    def _size(self, i):
        return i.nextint()

    def _dlayers(self, i, model, dropout):
        # Dense layers
        # 'DLayers'  : 'Count MaxSize MinSize FormatDen Act',
        # 'Act'      : 'sigmoid | relu | tanh',
        # 'MinSize'  : 'i(10,100)',
        # 'MaxSize'  : 'i(10,100)',
        # #las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
        # 'FormatDen': 'grow | shrink | same',

        count = self._count(i)
        minsize = self._minsize(i)
        maxsize = max(minsize, self._maxsize(i))
        activation = i.choose('sigmoid', 'relu', 'tanh')
        formatden = i.choose('grow', 'shrink', 'same')

        if formatden == 'grow':
            sizes = list(np.linspace(minsize, maxsize, count, dtype=int))
        elif formatden == 'shrink':
            sizes = list(np.linspace(maxsize, minsize, count, dtype=int))
        else:
            sizes = [minsize] * count

        for s in sizes:
            layer = Dense(s, activation=activation)(model)
            model = Dropout(dropout)(layer)

        return model

    def _minsize(self, i):
        return i.nextint()

    def _maxsize(self, i):
        return i.nextint()

    def _flayer(self, i, model, output_size, dropout):
        activation = i.choose('sigmoid', 'softmax')

        if output_size == 1 and activation == 'softmax':
            raise InvalidPipeline("Cannot use softmax with one output")

        z = Dense(output_size, activation=activation)(model)
        z = Dropout(dropout)(z)
        return z

    def _repr(self, i, dataset):
        # 'Prep Token SemFeat PosPrep MulWords Embed',
        texts = dataset.texts
        texts = self._prep(i, texts)
        tokens = self._token(i, texts)
        tokens = self._semfeat(i, tokens)
        tokens = self._posprep(i, tokens)
        tokens = self._mulwords(i, tokens)
        vectors = self._embed(i, tokens)

        dataset.vectors = vectors
        dataset.tokens = tokens

        return vectors, tokens

    def _prep(self, i, texts):
        #'DelPunt StripAcc'
        met = self._delpunt(i, texts)
        return self._stripacc(i, met)

    def _delpunt(self, i, texts):
        #yes | no
        if i.nextbool():
            return [t.translate({c:" " for c in string.punctuation}) for t in texts]
        else:
            return texts

    def _stripacc(self, i, texts):
        #yes | no
        if i.nextbool():
            return [gensim.utils.deaccent(t) for t in texts]
        else:
            return texts

    def _token(self, i, texts):
        return [[Token(w.text, w.idx, w.norm_, w.pos_, w.tag_, w.dep_, w.vector) for w in self.spacy_nlp(t)] for t in texts]

    def _posprep(self, i, tokens):
        tokens = self._stem(i, tokens)
        return self._stopw(i, tokens)

    def _stem(self, i, tokens):
        if i.nextbool():
            for tok in tokens:
                for t in tok:
                    t.norm = self.stemmer.stem(t.norm)

        return tokens

    def _stopw(self, i, tokens):
        if i.nextbool():
            sw = set(stopwords.words('spanish'))
        else:
            sw = set()

        return [[t for t in tok if not t.norm in sw] for tok in tokens]

    def _semfeat(self, i, tokens):
        # incluir pos-tag?
        if not i.nextbool():
            for tok in tokens:
                for t in tok:
                    del t.pos
                    del t.tag

        # incluir dependencias
        if not i.nextbool():
            for tok in tokens:
                for t in tok:
                    del t.dep

        self._umls(i, tokens)
        self._snomed(i, tokens)

        return tokens

    def _umls(self, i, tokens):
        if i.nextbool():
            return ""

    def _snomed(self, i, tokens):
        if i.nextbool():
            return ""

    def _mulwords(self, i, tokens):
        #'MulWords' : 'countPhrase | freeling | Ngram',
        choice = i.choose('countPhrase', 'freeling', 'Ngram')

        if choice == 'Ngram':
            ngram = i.nextint()

        return tokens

    def _ngram(self, i, tokens):
        #i(2,4)
        n = i.nextint() + 2
        result = tokens
        for i in range(1, n+1):
            result += list(nltk.ngrams(tokens, i))
        return result

    def _embed(self, i, tokens):
        # 'Embed' : 'wordVec | onehot | none',
        choice = i.choose('wv', 'onehot', 'none')

        # los objetos a codificar
        objs = [[dict(t.__dict__) for t in tok] for tok in tokens]
        # eliminar las propiedades inútiles
        for tok in objs:
            for t in tok:
                t.pop('init')
                t.pop('end')
                t.pop('text')

        if choice == 'wv':
            # eliminar el texto y codificar normal
            for tok in objs:
                for t in tok:
                    t.pop('norm')
                    t.pop('vector')

            vectors = self._dictvect(objs)

            matrices = []

            for tok,vec in szip(tokens, vectors):
                tok_matrix = np.vstack([t.vector for t in tok])
                matrices.append(np.hstack((tok_matrix, vec)))

            return matrices

        elif choice == 'onehot':
            # eliminar el vector y codificar onehot
            for tok in objs:
                for t in tok:
                    t.pop('vector')

            return self._dictvect(objs)

        else:
            # eliminar texto y vector y codificar normal
            for tok in objs:
                for t in tok:
                    del t['norm']
                    del t['vector']

            return self._dictvect(objs)

    def _dictvect(self, objs):
        dv = DictVectorizer()
        dv.fit(t for sent in objs for t in sent)
        return [dv.transform(sent).toarray() for sent in objs]
#

def main():
    grammar = TassGrammar()

    for i in range(203, 100000):
        random.seed(i)
        print("-------\nRandom seed %i" % i)

        ind = Individual([random.uniform(0,1) for _ in range(100)], grammar)
        sample = ind.sample()

        try:
            assert sample['Pipeline'][0]['Repr'][3]['PosPrep'][0]['StopW'] == ['no']
            assert sample['Pipeline'][0]['Repr'][5]['Embed'][0] == 'wordVec'
            sample['Pipeline'][2]['B'][0]['Class'][0]['NN']
        except:
            continue

        print(yaml.dump(sample))
        ind.reset()

        try:
            print(grammar.evaluate(ind))
            break
        except InvalidPipeline as e:
            print("Error", str(e))
            continue

if __name__ == '__main__':
    main()
