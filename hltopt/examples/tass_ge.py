# coding: utf-8

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

from pathlib import Path

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier

from gensim.models import Word2Vec
import gensim.downloader as api

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Input, concatenate

from ..ge import GrammarGE, GE, Individual, InvalidPipeline


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


class MyGrammar(GrammarGE):
    def __init__(self):
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

            'Seq'      : 'hmm | crf',
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
            #las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
            'FormatDen': 'grow | shrink | same',
            # Final layer
            'FLayer'   : 'crf | lr',

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
        # load training data
        dataset_path = Path.cwd() / 'hltopt' / 'examples' / 'datasets' / 'tass18_task3'

        texts = []
        labels = []
        relations = []

        for file in (dataset_path / 'training' / 'input').iterdir():
            if file.name.endswith('.txt'):
                goldA = dataset_path / 'training' / 'gold' / ('output_A_' + file.name[6:])
                goldB = dataset_path / 'training' / 'gold' / ('output_B_' + file.name[6:])
                goldC = dataset_path / 'training' / 'gold' / ('output_C_' + file.name[6:])

                text = file.open().read()
                sentences = [s for s in text.split('\n') if s]
                texts.extend(sentences)

                self._parse_ann(sentences, goldA, goldB, goldC, labels, relations)

        validation = dataset_path / 'develop' / 'input'/ 'input_develop.txt'
        validation_A = dataset_path / 'develop' / 'gold' / 'output_A_develop.txt'
        validation_B = dataset_path / 'develop' / 'gold' / 'output_B_develop.txt'
        validation_C = dataset_path / 'develop' / 'gold' / 'output_C_develop.txt'

        validation_sents = [s for s in validation.open().read().split('\n') if s]
        validation_size = len(validation_sents)

        texts.extend(validation_sents)

        self._parse_ann(validation_sents, validation_A, validation_B, validation_C, labels, relations)

        return self._pipeline(ind, texts, validation_size, labels, relations)

    def _parse_ann(self, sentences, goldA, goldB, goldC, labels, relations):
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

        labels.extend(labelsA_doc)
        relations.extend(relations_doc)

    def _pipeline(self, ind, texts, validation_size, labels, relations):
        # 'Pipeline' : 'Repr A B C | Repr AB C |  Repr A BC | Repr ABC',
        choice = ind.nextint(4)

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
        rep, tokens = self._repr(ind, texts)
        train = rep[:-validation_size]
        dev = rep[-validation_size:]

        # make sure we split right
        assert len(rep) == len(train) + len(dev)

        # mapping de la tarea A, B
        labels_map = []

        for sent, lbls in zip(tokens, labels):
            sent_map = []
            for t in sent:
                if (t.init, t.end) in lbls:
                    lbl = lbls[(t.init, t.end)][1]
                    sent_map.append(lbl)
                else:
                    sent_map.append('')

            labels_map.append(sent_map)

        mappingC = DictVectorizer()
        mappingC.fit([{k:True for k in [
            'is-a',
            'part-of',
            'property-of',
            'same-as',
            'subject',
            'target',
        ]}])

        # mapping de la tarea C (perdón)
        trainCx = []
        trainCy = []

        for feats, sent, lbls, rels in zip(rep, tokens, labels, relations):
            rel_pairs = []
            rel_map = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls:
                        continue

                    pair_map = {}

                    # id1, id2 son los id de 2 tokens
                    # que son keywords porque están en el mapping A
                    id1, lbl1 = lbls[(t1.init, t1.end)]
                    id2, lbl2 = lbls[(t2.init, t2.end)]

                    rel_pairs.append(np.hstack((feats[i], feats[j])))

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    rel_map.append(pair_map)

            assert len(rel_pairs) == len(rel_map)

            rel_pairs = np.vstack(rel_pairs)
            rel_map = mappingC.transform(rel_map).toarray()

            trainCx.append(rel_pairs)
            trainCy.append(rel_map)

        if choice == 0:
            # Ejecutar tareas A, B y C en secuencia

            # Tarea A
            labels_A = [np.asarray([1 if l else 0 for l in sent]) for sent in labels_map]
            result_A = self._a(ind, train, labels_A[:-validation_size], dev)

            # Tarea B
            trainX = []
            trainY = []

            for sent, lbls in zip(train, labels_map):
                lbls = np.asarray(lbls)
                idx = lbls != ''
                trainX.append(sent[idx])
                trainY.append(lbls[idx])

            result_B = self._b(ind, trainX, trainY, dev)

            # Tarea C
            result_C = self._c(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])

            # Calcular fitness
            test = labels_map[-validation_size:]
            test_A = np.hstack([np.asarray([1 if l else 0 for l in sent]) for sent in test])
            result_A = np.hstack(result_A)

            test_B = np.hstack([np.asarray(sent) for sent in test])[(test_A == 1) & (result_A == 1)]
            result_B = np.hstack(result_B)[(test_A == 1) & (result_A == 1)]

            test_C = np.vstack(trainCy[-validation_size:])
            result_C = np.vstack(result_C)

            # return ((test_A == result_A).sum() + (test_B == result_B).sum() + (test_C == result_C).sum()) / \
            #        (len(test_A) + len(test_B) + len(test_C) * 6)

            return (test_C == result_C).sum() / (len(test_C) * 6)
        elif choice == 1:
            # Ejecutar tareas AB juntas y C en secuencia

            # Tarea AB
            labels_AB = [np.asarray(sent) for sent in labels_map]
            result_AB = self._a(ind, train, labels_A[:-validation_size], dev)

            # Tarea C
            result_C = self._c(ind, trainCx[:-validation_size], trainCy[:-validation_size], trainCx[-validation_size:])
        else:
            raise InvalidPipeline("Only A B C supported so far")

    def _abc(self, i):
        return self._class(i)

    def _bc(self, i):
        return self._class(i)

    def _ab(self, i, trainX, trainY, devX)):
        if i.nextbool():
            return self._class(i)
        else:
            # sequence classifier
            raise InvalidPipeline("Sequence not supported yet")

    def _a(self, i, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        choice = i.nextint(2)

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

    def _b(self, i, trainX, trainY, devX):
        assert len(trainX) == len(trainY)

        # calcular la forma de la entrada
        rows, cols = trainX[0].shape
        intput_shape = cols

        clss = self._class(i, intput_shape, 1)

        # construir la entrada train
        trainX = np.vstack(trainX)
        trainY = np.hstack(trainY)

        clss.fit(trainX, trainY)

        return [clss.predict(x) for x in devX]

    def _c(self, i, trainX, trainY, devX):
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

    def _seq(self, i):
        if i.nextbool():
            #crf
            return None
        else:
            #hmm
            return None

    def _class(self, i, input_shape, output_shape):
        #LR | nb | SVM | dt | NN
        des = i.nextint(5)
        clss = None
        if des == 0:
            clss = self._lr(i)
        elif des == 1:
            clss = MultinomialNB()
        elif des == 2:
            clss = self._svm(i)
        elif des == 3:
            clss = DecisionTreeClassifier()
        else:
            return self._nn(i, input_shape, output_shape)

        if output_shape > 1:
            clss = OneVsRestClassifier(clss)

        return clss

    def _lr(self, i):
        return LogisticRegression(C=self._reg(i), penalty=self._penalty(i))

    def _reg(self, i):
        return i.nextfloat(0.01, 100)

    def _penalty(self, i):
        return i.choose('l1', 'l2')

    def _svm(self, i):
        return SVC(kernel=self._kernel(i))

    def _kernel(self, i):
        #linear | rbf | poly
        return i.choose('lineal', 'rbf', 'poly')

    def _nn(self, i, input_size, output_size):
        # CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop
        model = Sequential()
        option = i.nextint(3)

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
        # model.compile()
        return model

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
        return i.nextint(5) + 1

    def _minfilter(self, i):
        return i.nextint(5)

    def _maxfilter(self, i):
        return i.nextint(5)

    def _formatcon(self, i):
        return i.choose('same', 'all', 'rand')

    def _rlayers(self, i, model, dropout):
        # 'RLayers'  : 'Size',
        size = self._size(i)
        lstm = LSTM(size, dropout=dropout, recurrent_dropout=dropout)(model)
        return lstm

    def _size(self, i):
        return i.nextint(90) + 10

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
        return i.nextint(90)+10

    def _maxsize(self, i):
        return i.nextint(90)+10

    def _flayer(self, i, model, output_size, dropout):
        if output_size == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        z = Dense(output_size, activation=activation)(model)
        z = Dropout(dropout)(z)
        return z

    def _repr(self, i, texts):
        # 'Prep Token SemFeat PosPrep MulWords Embed',
        texts = self._prep(i, texts)
        tokens = self._token(i, texts)
        tokens = self._semfeat(i, tokens)
        tokens = self._posprep(i, tokens)
        tokens = self._mulwords(i, tokens)
        vectors = self._embed(i, tokens)
        return vectors, tokens

    def _prep(self, i, texts):
        #'DelPunt StripAcc'
        met = self._delpunt(i, texts)
        return self._stripacc(i, met)

    def _delpunt(self, i, texts):
        #yes | no
        if i.nextbool():
            return [t.translate({c:None for c in string.punctuation}) for t in texts]
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
        choice = i.nextint(3)

        if choice == 2:
            ngram = i.nextint(2) + 2

        return tokens

    def _ngram(self, i, tokens):
        #i(2,4)
        n = i.nextint(3) + 2
        result = tokens
        for i in range(1, n+1):
            result += list(nltk.ngrams(tokens, i))
        return result

    def _embed(self, i, tokens):
        # 'Embed' : 'wordVec | onehot | none',
        choice = i.nextint(3)

        # los objetos a codificar
        objs = [[dict(t.__dict__) for t in tok] for tok in tokens]
        # eliminar las propiedades inútiles
        for tok in objs:
            for t in tok:
                t.pop('init')
                t.pop('end')
                t.pop('text')

        if choice == 0:
            # eliminar el texto y codificar normal
            for tok in objs:
                for t in tok:
                    t.pop('norm')
                    t.pop('vector')

            vectors = self._dictvect(objs)
            vectors = [v.toarray() for v in vectors]

            matrices = []

            for tok,vec in zip(tokens, vectors):
                tok_matrix = np.vstack([t.vector for t in tok])
                matrices.append(np.hstack((tok_matrix, vec)))

            return matrices

        elif choice == 1:
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
        return [dv.transform(sent) for sent in objs]


def main():
    grammar = MyGrammar()

    i = Individual([0] * 100)
    print(yaml.dump(grammar.sample(i)))
    i.reset()
    print(grammar.evaluate(i))

    # ge = GE(grammar)
    # ge.run(100)


if __name__ == '__main__':
    main()
