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

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Input, concatenate

from ..ge import Grammar, PGE, Individual, InvalidPipeline


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


def cached(func):
    result = None
    @functools.wraps(func)
    def f(*args, **kwargs):
        nonlocal result
        if result is None:
            result = func(*args, **kwargs)
        return result
    return f


class Dataset:
    def __init__(self):
        self.texts = []
        self.labels = []
        self.relations = []
        self.validation_size = 0
        self.vectors = None
        self.tokens = None

    def load(self, finput:Path):
        goldA = finput.parent / ('output_A_' + finput.name[6:])
        goldB = finput.parent / ('output_B_' + finput.name[6:])
        goldC = finput.parent / ('output_C_' + finput.name[6:])

        text = finput.open().read()
        sentences = [s for s in text.split('\n') if s]

        self.texts.extend(sentences)
        self._parse_ann(sentences, goldA, goldB, goldC)

        return len(sentences)

    def _parse_ann(self, sentences, goldA, goldB, goldC):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        labelsA_doc = [{} for _ in sentences]
        relations_doc = [[] for _ in sentences]
        labelsB = {}
        sent_map = {}

        for line in goldB.open():
            lid, lbl = line.split()
            labelsB[int(lid)] = lbl

        for line in goldA.open():
            lid, start, end = (int(i) for i in line.split())

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, start)
            if i > 0:
                start -= sentences_length[i-1] + 1
                end -= sentences_length[i-1] + 1
            labelsA_doc[i][(start,end)] = (lid, labelsB[lid])
            sent_map[lid] = i

        for line in goldC.open():
            rel, org, dest = line.split()
            org, dest = int(org), int(dest)
            sent = sent_map[org]
            assert sent == sent_map[dest]
            relations_doc[sent].append((rel, org, dest))

        self.labels.extend(labelsA_doc)
        self.relations.extend(relations_doc)

    def _check_repr(self):
        if self.vectors is None or self.tokens is None:
            raise ValueError("Preprocesing and representation is not ready yet.")

    @cached
    def _labels_map(self):
        labels_map = []

        for sent, lbls in zip(self.tokens, self.labels):
            sent_map = []
            for t in sent:
                if (t.init, t.end) in lbls:
                    lbl = lbls[(t.init, t.end)][1]
                    sent_map.append(lbl)
                else:
                    sent_map.append('')

            labels_map.append(sent_map)

        return labels_map

    def task_a_by_word(self):
        self._check_repr()

        labels_map = self._labels_map()
        xtrain = self.vectors[:-self.validation_size]
        ytrain = [np.asarray([1 if l else 0 for l in sent]) for sent in labels_map[:-self.validation_size]]
        xdev = self.vectors[-self.validation_size:]

        return xtrain, ytrain, xdev

    def task_b_by_word(self, result_A):
        self._check_repr()

        xtrain = []
        ytrain = []

        for sentence, labels in zip(self.vectors, self._labels_map()):
            new_sent = []
            new_lbl = []
            for word, lbl in zip(sentence, labels):
                if lbl:
                    new_sent.append(word)
                    new_lbl.append(lbl)
            if new_sent:
                new_sent = np.vstack(new_sent)
                new_lbl = np.hstack(new_lbl)

            xtrain.append(new_sent)
            ytrain.append(new_lbl)

        xdev = []


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
            'NN'       : 'CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop',
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
            # 'WordVec'  : 'yes | no',
            # 'SenseVec' : 'yes | no',
            # 'CharEmbed': 'yes | no',
        }

    def evaluate(self, ind:Individual):
        FAST = True
        TEST = False

        # load training data
        dataset_path = Path.cwd() / 'hltopt' / 'examples' / 'datasets' / 'tass18_task3'
        dataset = Dataset()

        for file in (dataset_path / 'training').iterdir():
            if file.name.startswith('input'):
                dataset.load(file)

                if FAST:
                    break

        if FAST:
            dataset.validation_size = 10
        else:
            validation = dataset_path / 'develop' / 'input'/ 'input_develop.txt'
            dataset.validation_size = dataset.load(validation)

            if TEST:
                test = dataset_path / 'test' / 'input'/ 'scenario1-ABC' / 'input_scenario1.txt'
                dataset.validation_size = dataset.load(test)

        return self._pipeline(ind, dataset)

    def _pipeline(self, ind, dataset):
        # 'Pipeline' : 'Repr A B C | Repr AB C |  Repr A BC | Repr ABC',
        choice = ind.choose('A B C', 'AB C', 'A BC', 'ABC')

        # repr es una lista de matrices (una matriz por cada oración):
        # [
        #   array([           <- oración 0
        #       0.1, 0.4, ... <- token 0
        #       0.2, 0.6, ... <- token 1
        #       ...
        #       0.9, 0.2, ... <- token n
        #   ]),
        #   ...               <- oración 1
        # ]
        self._repr(ind, dataset)

        # mapping de la tarea A, B
        # labels_map = []

        # for sent, lbls in zip(tokens, labels):
        #     sent_map = []
        #     for t in sent:
        #         if (t.init, t.end) in lbls:
        #             lbl = lbls[(t.init, t.end)][1]
        #             sent_map.append(lbl)
        #         else:
        #             sent_map.append('')

        #     labels_map.append(sent_map)

        # # mapping de la tarea C
        # mappingC = DictVectorizer()
        # mappingC.fit([{k:True for k in [
        #     'is-a',
        #     'part-of',
        #     'property-of',
        #     'same-as',
        #     'subject',
        #     'target',
        # ]}])

        if choice == 'A B C':
            # Ejecutar tareas A, B y C en secuencia

            # Tarea A
            result_A = self._a(ind, dataset)

            # Tarea B
            result_B = self._b(ind, dataset, result_A)

            # Tarea C
            trainCx, trainCy, valCmapping = self._build_c_mapping_pairs(rep, tokens, labels, relations, mappingC)
            result_C = self._c(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])

            # Calcular fitness
            # Labels
            ids = 0
            val_labels = []
            for sent, resA, resB in zip(tokens[-validation_size:], result_A, result_B):
                sentence_labels = {}
                for tok, clsA, clsB in zip(sent, resA, resB):
                    if clsA == 1:
                        sentence_labels[(tok.init, tok.end)] = (ids, clsB)
                        ids += 1
                val_labels.append(sentence_labels)

            # relations
            val_relations = []
            for lbl, resC, mapC in zip(val_labels, result_C, valCmapping[-validation_size:]):
                sentence_rels = []
                for rels, (org, dest) in zip(resC, mapC):
                    if org not in lbl or dest not in lbl:
                        continue
                    orgid, orglb = lbl[org]
                    destid, destlb = lbl[dest]

                    rels = mappingC.inverse_transform(rels.reshape(1,-1))[0]
                    for r in rels:
                        if r in ['subject', 'target']:
                            if orglb == 'Action' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))
                        else:
                            if orglb == 'Concept' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))

                val_relations.append(sentence_rels)

            return self._score(labels[-validation_size:], val_labels, relations[-validation_size:], val_relations)

        elif choice == 'AB C':
            # Ejecutar tareas AB juntas y C en secuencia

            # Tarea AB
            labels_AB = [np.asarray(sent) for sent in labels_map]
            result_AB = self._ab(ind, train, labels_AB[:-validation_size], dev)

            # Tarea C
            trainCx, trainCy, valCmapping = self._build_c_mapping_pairs(rep, tokens, labels, relations, mappingC)
            result_C = self._c(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])

            # Calcular fitness
            # Labels
            ids = 0
            val_labels = []
            for sent, resAB in zip(tokens[-validation_size:], result_AB):
                sentence_labels = {}
                for tok, clsAB in zip(sent, resAB):
                    if clsAB:
                        sentence_labels[(tok.init, tok.end)] = (ids, clsAB)
                        ids += 1
                val_labels.append(sentence_labels)

            # relations
            val_relations = []
            for lbl, resC, mapC in zip(val_labels, result_C, valCmapping[-validation_size:]):
                sentence_rels = []
                for rels, (org, dest) in zip(resC, mapC):
                    if org not in lbl or dest not in lbl:
                        continue
                    orgid, orglb = lbl[org]
                    destid, destlb = lbl[dest]

                    rels = mappingC.inverse_transform(rels.reshape(1,-1))[0]
                    for r in rels:
                        if r in ['subject', 'target']:
                            if orglb == 'Action' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))
                        else:
                            if orglb == 'Concept' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))

                val_relations.append(sentence_rels)

            return self._score(labels[-validation_size:], val_labels, relations[-validation_size:], val_relations)

        elif choice == 'A BC':
            # Ejecutar tarea A y luego BC

            # Tarea A
            labels_A = [np.asarray([1 if l else 0 for l in sent]) for sent in labels_map]
            result_A = self._a(ind, train, labels_A[:-validation_size], dev)

            # Tarea BC
            trainCx, trainCy, valCmapping = self._build_c_mapping_pairs(rep, tokens, labels, relations, mappingC, build_b=True)
            result_BC = self._bc(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])

            # Calcular fitness
            # Labels
            ids = 0
            val_labels = []
            for sent, resA, resB in zip(tokens[-validation_size:], result_A, result_BC):
                sentence_labels = {}
                for tok, clsA in zip(sent, resA):
                    if clsA == 1:
                        sentence_labels[(tok.init, tok.end)] = (ids, {'Concept':0, 'Action':0})
                        ids += 1
                val_labels.append(sentence_labels)

            # compute actual labels
            # labels map
            bc_labels_map = {
                (0,0): 'None',
                (1,0): 'Concept',
                (0,1):  'Action'
            }

            for lbl, resC, mapC in zip(val_labels, result_BC, valCmapping[-validation_size:]):
                for rels, (org, dest) in zip(resC, mapC):
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
            for lbl, resC, mapC in zip(val_labels, result_BC, valCmapping[-validation_size:]):
                sentence_rels = []
                for rels, (org, dest) in zip(resC, mapC):
                    if org not in lbl or dest not in lbl:
                        continue
                    orgid, orglb = lbl[org]
                    destid, destlb = lbl[dest]

                    rels = mappingC.inverse_transform(rels[:-4].reshape(1,-1))[0]
                    for r in rels:
                        if r in ['subject', 'target']:
                            if orglb == 'Action' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))
                        else:
                            if orglb == 'Concept' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))

                val_relations.append(sentence_rels)

            return self._score(labels[-validation_size:], val_labels, relations[-validation_size:], val_relations)

        else:
            # Ejecutar Tarea ABC junta

            # Tarea ABC
            trainCx, trainCy, valCmapping = self._build_c_mapping_pairs(rep, tokens, labels, relations, mappingC, build_b=True, build_a=True)
            result_ABC = self._abc(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])

            # Calcular fitness

            # compute actual labels
            # labels map
            bc_labels_map = {
                (0,0): 'None',
                (1,0): 'Concept',
                (0,1):  'Action'
            }

            # Labels
            ids = 0
            val_labels = []
            for sent in tokens[-validation_size:]:
                sentence_labels = {}
                for tok in sent:
                    sentence_labels[(tok.init, tok.end)] = (ids, {'Concept':0, 'Action':0, 'None':0})
                    ids += 1
                val_labels.append(sentence_labels)

            # compute actual labels
            # labels map
            bc_labels_map = {
                (0,0): 'None',
                (1,0): 'Concept',
                (0,1): 'Action',
                (1,1): 'None'
            }

            for lbl, resC, mapC in zip(val_labels, result_ABC, valCmapping[-validation_size:]):
                for rels, (org, dest) in zip(resC, mapC):
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

            # # compute relations
            val_relations = []
            for lbl, resC, mapC in zip(val_labels, result_ABC, valCmapping[-validation_size:]):
                sentence_rels = []
                for rels, (org, dest) in zip(resC, mapC):
                    if org not in lbl or dest not in lbl:
                        continue
                    orgid, orglb = lbl[org]
                    destid, destlb = lbl[dest]

                    rels = mappingC.inverse_transform(rels[:-4].reshape(1,-1))[0]
                    for r in rels:
                        if r in ['subject', 'target']:
                            if orglb == 'Action' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))
                        else:
                            if orglb == 'Concept' and destlb == 'Concept':
                                sentence_rels.append((r, orgid, destid))

                val_relations.append(sentence_rels)

            return self._score(labels[-validation_size:], val_labels, relations[-validation_size:], val_relations)

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

        for l1, l2 in zip(train_labels, val_labels):
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

        for r1, r2 in zip(train_relations, val_relations):
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

        for r1, r2 in zip(train_relations, val_relations):
            for r,vo,vd in r2:
                if not vo in v2t or not vd in v2t:
                    spuriousC += 1
                    continue

                to = v2t[vo]
                td = v2t[vd]

                if (r,to,td) not in r1:
                    spuriousC += 1


        top = (correctA + 0.5 * partialA + correctB + correctC)
        precision = top / (correctA + partialA + correctB + incorrectB + spuriousA + correctC + spuriousC)
        recall = top / (correctA + partialA + correctB + incorrectB + missingA + correctC + missingC)

        return 2 * precision * recall / (precision + recall)

    def _build_c_mapping_pairs(self, rep, tokens, labels, relations, mappingC, build_b=False, build_a=False):
        # mapping de la tarea C (perdón)
        trainCx = []
        trainCy = []
        valCmapping = []

        label_maps = {
            'Action': [0,1],
            'Concept': [1,0],
            None: [0,0]
        }

        for feats, sent, lbls, rels in zip(rep, tokens, labels, relations):
            rel_pairs = []
            rel_map = []
            rel_clss = []
            sent_mapp = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls and not build_a:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls and not build_a:
                        continue

                    pair_map = {}

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    if build_b:
                        lbl1e = label_maps[lbl1]
                        lbl2e = label_maps[lbl2]
                        lble = np.asarray(lbl1e + lbl2e)
                        rel_clss.append(lble)

                    rel_pairs.append(np.hstack((feats[i], feats[j])))

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    rel_map.append(pair_map)
                    sent_mapp.append(((t1.init, t1.end),(t2.init, t2.end)))

            valCmapping.append(sent_mapp)
            rel_pairs = np.vstack(rel_pairs)
            rel_map = mappingC.transform(rel_map).toarray()

            if build_b:
                rel_clss = np.vstack(rel_clss)

            if build_b:
                rel_map = np.hstack((rel_map, rel_clss))

            trainCx.append(rel_pairs)
            trainCy.append(rel_map)

        return trainCx, trainCy, valCmapping

    def _abc(self, i, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        # calcular la forma de la entrada
        rows, cols = trainX[0].shape
        intput_shape = cols
        rows, cols = trainY[0].shape
        output_shape = cols

        clss = self._class(i, intput_shape, output_shape)

        # construir la entrada train
        trainX = np.vstack(trainX)
        trainY = np.vstack(trainY)

        clss.fit(trainX, trainY)

        return [clss.predict(x) for x in devX]

    def _bc(self, i, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        # calcular la forma de la entrada
        rows, cols = trainX[0].shape
        intput_shape = cols
        rows, cols = trainY[0].shape
        output_shape = cols

        clss = self._class(i, intput_shape, output_shape)

        # construir la entrada train
        trainX = np.vstack(trainX)
        trainY = np.vstack(trainY)

        clss.fit(trainX, trainY)

        return [clss.predict(x) for x in devX]

    def _ab(self, i, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        choice = i.nextint()

        if choice == 0:
            # classifier

            # calcular la forma de la entrada
            rows, cols = trainX[0].shape
            intput_shape = cols

            clss = self._class(i, intput_shape, 1)

            # construir la entrada train
            trainX = np.vstack(trainX)
            trainY = np.hstack(trainY)

            clss.fit(trainX, trainY)

            # construir la entrada dev
            return [clss.predict(x) for x in devX]
        else:
            # sequence classifier
            raise InvalidPipeline("Sequence not supported yet")

    def _a(self, ind:Individual, dataset:Dataset):
        choice = ind.choose('class', 'seq')

        if choice == 'class':
            # classifier

            # calcular la forma de la entrada
            rows, cols = dataset.vectors[0].shape
            intput_shape = cols

            clss = self._class(ind, intput_shape, 1)

            if isinstance(clss, Model):
                xtrain, ytrain, xdev = dataset.task_a_by_sentence()
            else:
                xtrain, ytrain, xdev = dataset.task_a_by_word()
                xtrain = np.vstack(xtrain)
                ytrain = np.hstack(ytrain)

            clss.fit(xtrain, ytrain)

            # construir la entrada dev
            return [clss.predict(x) for x in xdev]
        else:
            # sequence classifier
            # alg = ind.choose('hmm', 'crf')

            xtrain, ytrain, xdev = dataset.task_a_by_word()
            return self._hmm(ind, xtrain, ytrain, xdev)

            # if alg == 'hmm':
            # else:
            #     raise InvalidPipeline('CRF not implemented')
            #     return self._crf(ind, xtrain, ytrain, xdev)

    def _hmm(self, ind:Individual, trainX, trainY, devX):
        lengths = [x.shape[0] for x in trainX]

        trainX = np.vstack(trainX).astype(int)
        trainY = np.hstack(trainY)

        try:
            hmm = MultinomialHMM(decode=ind.choose('viterbi', 'bestfirst'), alpha=ind.nextfloat())
            hmm.fit(trainX, trainY, lengths)
            devX = [x.astype(int) for x in devX]
            return [hmm.predict(x) for x in devX]
        except ValueError as e:
            if 'non-negative integers' in str(e):
                raise InvalidPipeline(str(e))
            elif 'unknown categories' in str(e):
                raise InvalidPipeline(str(e))
            else:
                raise

    def _crf(self, ind:Individual, trainX, trainY, devX):
        crf = CRF()
        crf.fit(trainX, trainY)
        return [crf.predict(x) for x in devX]

    def _b(self, ind:Individual, dataset:Dataset, result_A):
        # calcular la forma de la entrada
        rows, cols = dataset.vectors[0].shape
        intput_shape = cols

        clss = self._class(ind, intput_shape, 1)

        if isinstance(clss, Model):
            xtrain, ytrain, xdev = dataset.task_b_by_sentence(result_A)
        else:
            xtrain, ytrain, xdev = dataset.task_b_by_word(result_A)
            xtrain = np.vstack(xtrain)
            ytrain = np.hstack(ytrain)

        clss.fit(xtrain, ytrain)

        # construir la entrada dev
        return [clss.predict(x) for x in xdev]

    def _c(self, ind:Individual, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        # calcular la forma de la entrada
        rows, cols = trainX[0].shape
        intput_shape = cols
        rows, cols = trainY[0].shape
        output_shape = cols

        clss = self._class(ind, intput_shape, output_shape)

        # construir la entrada train
        trainX = np.vstack(trainX)
        trainY = np.vstack(trainY)

        clss.fit(trainX, trainY)

        return [clss.predict(x) for x in devX]

    def _seq(self, i):
        if i.nextbool():
            #crf
            return None
        else:
            #hmm
            return None

    def _class(self, ind:Individual, input_shape=None, output_shape=None):
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
            return self._nn(ind, input_shape, output_shape)

        if output_shape > 1 and des != 'nn':
            clss = OneVsRestClassifier(clss)

        return clss

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

    def _nn(self, i, input_size, output_size):
        try:
            # CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop
            model = Sequential()
            option = i.nextint()

            x = Input(shape=(input_size, 50))
            dropout = self._drop(i)

            if option == 0:
                y = self._cvlayers(i, x, dropout)
                y = self._dlayers(i, y, dropout)
                y = self._flayer(i, y, output_size, dropout)
            elif option == 1:
                y = self._rlayers(i, x, dropout)
                y = self._dlayers(i, y, dropout)
                y = self._flayer(i, y, output_size, dropout)
            else:
                y = self._dlayers(i, x, dropout)
                y = self._flayer(i, y, output_size, dropout)

            model = Model(inputs=x, outputs=y)
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            return model
        except ValueError as e:
            msg = str(e)
            if 'out of bounds' in msg:
                raise InvalidPipeline('Bad NN architecture')
            else:
                raise e

    def _drop(self, i):
        return i.nextfloat(0.1, 0.5)

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

            for tok,vec in zip(tokens, vectors):
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

    for i in range(0, 100000):
        random.seed(i)
        print("-------\nRandom seed %i" % i)

        ind = Individual([random.uniform(0,1) for _ in range(100)], grammar)
        sample = ind.sample()

        try:
            assert sample['Pipeline'][0]['Repr'][5]['Embed'][0] == 'onehot'
            sample['Pipeline'][1]['A'][0]['Seq'][0]['HMM']
            sample['Pipeline'][2]['B'][0]['Class'][0]['LR']
            sample['Pipeline'][3]['C'][0]['Class'][0]['LR']
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
