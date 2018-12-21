# coding: utf-8

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import gensim.downloader as api

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Input, concatenate

from ..ge import GrammarGE, GE, Individual


class Token:
    def __init__(self, text, label, pos_init, normalized=None):
        self.text = text
        self.label = label
        self.pos_init = pos_init
        self.pos_end = pos_init + len(text)
        self.normalized = normalized or text

    def __and__(self, other):
        return max(0, min(self.pos_end, other.pos_end) - max(self.pos_init, other.pos_init)) > 0

    def __repr__(self):
        return self.text

    def __json__(self):
        return self.__dict__


class MyGrammar(GrammarGE):
    def __init__(self):
        self.stemmer = SnowballStemmer("spanish")
        self.spacy_nlp = spacy.load('es')

    def grammar(self):
        return {
            'Pipeline' : 'Repr AB C | Repr A B C | Repr ABC | Repr A BC',
            'ABC'      : 'Class',
            'BC'       : 'Class',
            'AB'       : 'Seq | Class',
            'A'        : 'Seq | Class',
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
            # Con el objetivo de eliminar la recursividad de la grmática y controlar el tamaño de las capas
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
            'Embed'    : 'WordVec SenseVec CharEmbed | Vectorizer',
            'WordVec'  : 'yes | no',
            'SenseVec' : 'yes | no',
            'CharEmbed': 'yes | no',
            'Vectorizer': 'countVect | tfxIdfVect',
        }

    def evaluate(self, i:Individual):
        # load training data
        dataset_path = Path.cwd() / 'hltopt' / 'examples' / 'datasets' / 'tass18_task3'

        texts = []

        for file in (dataset_path / 'training' / 'input').iterdir():
            if file.name.endswith('.txt'):
                goldA = dataset_path / 'training' / 'gold' / ('output_A_' + file.name[6:])
                goldB = dataset_path / 'training' / 'gold' / ('output_B_' + file.name[6:])
                goldC = dataset_path / 'training' / 'gold' / ('output_C_' + file.name[6:])

                text = file.open().read()
                texts.append(text)

        return self.__Pipeline__(i, texts)

    def __Pipeline__(self, i, texts):
        # 'Pipeline' : 'Repr AB C | Repr A B C | Repr ABC | Repr A BC',
        choice = 1 #i.nextint(4)

        if choice == 0:
            pass
        elif choice == 1:
            rep = self.__Repr__(i, texts)
        elif choice == 2:
            pass
        else:
            pass

    def __ABC__(self, i):
        return self.__Class__(i)

    def __BC__(self, i):
        return self.__Class__(i)

    def __AB__(self, i):
        if i.nextbool():
            return self.__Class__(i)
        else:
            return self.__Seq__(i)

    def __A__(self, i):
        if i.nextbool():
            return self.__Class__(i)
        else:
            return self.__Seq__(i)

    def __B__(self, i):
        return self.__Class__(i)

    def __C__(self, i):
        return self.__Class__(i)

    def __Seq__(self, i):
        if i.nextbool():
            #crf
            return None
        else:
            #hmm
            return None

    def __Class__(self, i):
        #LR | nb | SVM | dt | NN
        des = i.nextint(5)
        if des == 0:
            return self.__LR__(i)
        elif des == 1:
            return MultinomialNB()
        elif des == 2:
            return self.__SVM__(i)
        elif des == 3:
            return DecisionTreeClasifier()
        else:
            return self.__NN__(i)

    def __LR__(self, i):
        return LogisticRegression(C=self.__Reg__(i), penalty=self.__Penalty__(i))

    def __Reg__(self, i):
        return i.nextfloat(0.01, 100)

    def __Penalty__(self, i):
        return i.choose('l1', 'l2')

    def __SVM__(self, i):
        return SVC(kernel=self.__Kernel__(i))

    def __Kernel__(self, i):
        #linear | rbf | poly
        return i.choose('lineal', 'rbf', 'poly')

    def __NN__(self, i, input_size):
        # CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop
        model = Sequential()
        option = i.nextint(3)

        x = Input(shape=(input_size, 50))
        dropout = self.__Drop__(i)

        if option == 0:
            y = self.__CVLayers__(i, x, dropout)
            y = self.__DLayers__(i, y, dropout)
            y = self.__FLayer__(i, y, dropout)
        elif option == 1:
            y = self.__RLayers__(i, x, dropout)
            y = self.__DLayers__(i, y, dropout)
            y = self.__FLayer__(i, y, dropout)
        else:
            y = self.__DLayers__(i, x, dropout)
            y = self.__FLayer__(i, y, dropout)

        model = Model(inputs=x, outputs=y)
        # model.compile()
        return model

    def __Drop__(self, i):
        return i.nextfloat(0.1, 0.5)

    def __CVLayers__(self, i, model, dropout):
        # 'CVLayers' : 'Count MinFilter MaxFilter FormatCon',
        count = self.__Count__(i)
        minfilter = self.__MinFilter__(i)
        maxfilter = max(minfilter, self.__MaxFilter__(i))
        formatcon = self.__FormatCon__(i)

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

    def __Count__(self, i):
        return i.nextint(5) + 1

    def __MinFilter__(self, i):
        return i.nextint(5)

    def __MaxFilter__(self, i):
        return i.nextint(5)

    def __FormatCon__(self, i):
        return i.choose('same', 'all', 'rand')

    def __RLayers__(self, i, model, dropout):
        # 'RLayers'  : 'Size',
        size = self.__Size__(i)
        lstm = LSTM(size, dropout=dropout, recurrent_dropout=dropout)(model)
        return lstm

    def __Size__(self, i):
        return i.nextint(90) + 10

    def __DLayers__(self, i, model, dropout):
        # Dense layers
        # 'DLayers'  : 'Count MaxSize MinSize FormatDen Act',
        # 'Act'      : 'sigmoid | relu | tanh',
        # 'MinSize'  : 'i(10,100)',
        # 'MaxSize'  : 'i(10,100)',
        # #las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
        # 'FormatDen': 'grow | shrink | same',

        count = self.__Count__(i)
        minsize = self.__MinSize__(i)
        maxsize = max(minsize, self.__MaxSize__(i))
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

    def __MinSize__(self, i):
        return i.nextint(90)+10

    def __MaxSize__(self, i):
        return i.nextint(90)+10

    def __FLayer__(self, i, model, dropout):
        return model

    def __Repr__(self, i, texts):
        # 'Prep Token SemFeat PosPrep MulWords Embed',
        texts = self.__Prep__(i, texts)


    def __Prep__(self, i, texts):
        #'DelPunt StripAcc'
        met = self.__DelPunt__(i, texts)
        return self.__StripAcc__(i, met)

    def __DelPunt__(self, i, texts):
        #yes | no
        if i.nextbool():
            return texto.translate(None, string.punctuation)
        else:
            return texts

    def __StripAcc__(self, i, texto):
        #yes | no
        if i.nextbool():
            return gensim.utils.deaccent(texto)
        else:
            return texto

    def __Token__(self, i, texto):
        tokens = self.spacy_nlp(texto)
        return [t.text for t in tokens]

    def __PosTag__(self, i, texto):
        tokens = self.spacy_nlp(texto)
        return [t.pos_ + t.tag_ for t in tokens]

    def __Dep__(self, i, texto):
        tokens = self.spacy_nlp(texto)
        return [t.dep_ for t in tokens]

    def __PosPrep__(self, i, tokens):
        tokens = self.__Stem__(i, tokens)
        return self.__StopW__(i, tokens)

    def __Stem__(self, i, tokens):
        if i.nextbool():
            new = []
            for t in tokens:
                new.append(self.stemmer.stem(t))
            return new
        else:
            return tokens

    def __StopW__(self, i, tokens):
        if i.nextbool():
            sw = set(stopwords.words('spanish'))
        else:
            sw = set()

        return [t for t in tokens if not t.lower() in sw]

    def __SemFeat__(self, i, tokens, texto):
        """Recibe el texto divido en tokens y el texto plano y devuelve un diccionario que
        tiene como llave el tipo de feature extraido y el resultado de este proceso como
        valor."""
        #'PosTag
        new = {}
        new['postag'] = self.__PosTag__(i, texto)
        new['dep'] = self.__Dep__(i, texto)
        new['umls'] = self.__UMLS__(i)
        new['snomed'] = self.__SNOMED__(i)
        return new

    def __UMLS__(self, i):
        return False

    def __SNOMED__(self, i):
        return False

    def __MulWords__(self, i):
        pass

    def __Ngram__(self, i, tokens):
        #i(2,4)
        n = i.nextint(3) + 2
        result = tokens
        for i in range(1, n+1):
            result += list(nltk.ngrams(tokens, i))
        return result

    def __Embed__(self, i, text, tokenized):
        if i.nextbool():
            return self.__WordVec__(i, text)
        else:
            return self.__Vectorizer__(i, tokenized)

    def __WordVec__(self, i, documents):
        return [[t.vector for t in nlp(sentence)] for sentence in documents]

    def __SenseVec__(self, i):
        pass

    def __CharEmbed__(self, i):
        pass

    def __Vectorizer__(self, i, documents):
        """Devuelve la representación vectorizada como un
        vector de numpy."""
        if i.nextbool():
            vectorizer = CountVectorizer(analyzer = lambda x: x)
            X = vectorizer.fit_transform(documents)
            return X
        else:
            vectorizer = TfidfVectorizer(analyzer = lambda x: x)
            X = vectorizer.fit_transform(documents)
            return X

def main():
    grammar = MyGrammar()
    # random.seed(42)
    i = Individual([random.uniform(0,1) for i in range(100)])
    print(yaml.dump(grammar.sample(i)))
    i.reset()

    grammar.evaluate(i)


if __name__ == '__main__':
    main()
