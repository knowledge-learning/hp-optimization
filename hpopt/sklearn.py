# coding: utf-8

import numpy as np
from scipy import sparse as sp
from collections import Counter

import spacy
import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# classifiers
## bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

## linear
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Perceptron

## svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

## trees
from sklearn.tree import DecisionTreeClassifier

## knn
from sklearn.neighbors import KNeighborsClassifier

## discriminant
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## neural networks
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

## ensembles
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier

# data preprocesing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import robust_scale
from sklearn.impute import SimpleImputer

# feature preprocessing
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import Nystroem
# from sklearn.feature_selection import SelectPercentile

grammar = {
    'Pipeline'     : 'DataPrep FeatPrep Class',

    'DataPrep'     : 'Encoding Rescaling Imputation Balancing',
    'Encoding'     : 'none | onehot',
    'Rescaling'    : 'none | minmax | standard | quantile',
    'Imputation'   : 'none | mean | median | most_frequent',
    'Balancing'    : 'none | weight',
    'FeatPrep'     : 'none | Decomp | FeatSel',

    'Decomp'       : 'FastICA | PCA | TruncSVD | KernelPCA',
    'FastICA'      : 'f(0.01,0.5)',
    'PCA'          : 'f(0.01,0.5)',
    'TruncSVD'     : 'f(0.01,0.5)',
    'KernelPCA'    : 'KPCAn KPCAk',
    'KPCAn'        : 'f(0.01,0.5)',
    'KPCAk'        : 'linear | poly | rbf | sigmoid | cosine',

    'FeatSel'      : 'FeatAgg | Poly | Nystrom',
    'FeatAgg'      : 'f(0.01,0.5)',
    'Poly'         : 'i(2,3)',
    'Nystrom'      : 'f(0.01,0.5)',

    'Class'        : 'Bayes | Linear | SVC | Tree | KNN | Discriminant | MLP',
    'Bayes'        : 'gaussNB | mNB | cNB | nNB',
    'Linear'       : 'SGD | Ridge | PA | LR | Lasso | Perceptron',
    'SGD'          : 'hinge | log | modified_huber | squared_hinge | perceptron',
    'Ridge'        : 'f(0.01, 10)',
    'PA'           : 'f(0.01, 10)',
    'LR'           : 'LRloss LRreg',
    'LRloss'       : 'l1 | l2',
    'LRreg'        : 'f(0.01, 10)',
    'Lasso'        : 'f(0.01, 10)',
    'Perceptron'   : 'l1 | l2 | elasticnet',
    'SVC'          : 'LinearSVC | KernelSVC',
    'LinearSVC'    : 'LinearSVCp LinearSVCr',
    'LinearSVCp'   : 'l1 | l2',
    'LinearSVCr'   : 'f(0.01,10)',
    'KernelSVC'    : 'KernelSVCk KernelSVCr',
    'KernelSVCk'   : 'rbf | poly | sigmoid',
    'KernelSVCr'   : 'f(0.01,10)',
    'Tree'         : 'gini | entropy',
    'KNN'          : 'i(1,10)',
    'Discriminant' : 'qda | lda',
    'MLP'          : 'MLPn MLPl MLPa',
    'MLPn'         : 'i(10,100)',
    'MLPl'         : 'i(1,5)',
    'MLPa'         : 'identity | logistic | tanh | relu',
}


from sklearn.model_selection import train_test_split
from .ge import Grammar, PGE
from .utils import InvalidPipeline


class SklearnGrammar(Grammar):
    def __init__(self, X, y):
        super().__init__()

        self.X = X
        self.y = y

    def grammar(self):
        return grammar

    def evaluate(self, ind):
        # 'Pipeline'     : 'DataPrep FeatPrep Class',
        X, y = self.X, self.y
        X, balance = self._data_prep(ind, X)
        X = self._feat_prep(ind, X)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

        classifier = self._classifier(ind, balance)

        try:
            classifier.fit(Xtrain, ytrain)
        except TypeError as e:
            if 'sparse' in str(e) and hasattr(Xtrain, 'toarray'):
                Xtrain = Xtrain.toarray()
                Xtest = Xtest.toarray()
                classifier.fit(Xtrain, ytrain)
            else:
                raise e
        except ValueError as e:
            if 'must be non-negative' in str(e):
                raise InvalidPipeline()
            raise e

        return classifier.score(Xtest, ytest)

    def train(self, ind, X, y):
        X, balance = self._data_prep(ind, X)
        X = self._feat_prep(ind, X)

        classifier = self._classifier(ind, balance)
        classifier.fit(X, y)

        return classifier

    def process(self, ind, X):
        X, _ = self._data_prep(ind, X)
        X = self._feat_prep(ind, X)
        return X

    def _data_prep(self, ind, X):
        # 'DataPrep'     : 'Encoding Rescaling Imputation Balancing',
        X = self._encoding(ind, X)
        X = self._rescaling(ind, X)
        X = self._imputation(ind, X)
        balance = 'balanced' if ind.choose('none', 'weight') == 'weight' else None

        return X, balance

    def _encoding(self, ind, X):
        # 'Encoding'     : 'none | onehot',
        if ind.choose('none', 'onehot') == 'onehot':
            try:
                X = OneHotEncoder(categories='auto').fit_transform(X)
            except TypeError as e:
                if 'dense data is required' in str(e):
                    raise InvalidPipeline(str(e))
                else:
                    raise

        return X

    def _rescaling(self, ind, X):
        # 'Rescaling'    : 'none | minmax | standard | quantile',
        scaling = ind.choose(None, minmax_scale, robust_scale, quantile_transform)

        if scaling:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = scaling(X)

        return X

    def _imputation(self, ind, X):
        # 'Imputation'   : 'none | mean | median | most_frequent',
        method = ind.choose('none', 'mean', 'median', 'most_frequent')

        if method != 'none':
            X = SimpleImputer(strategy=method).fit_transform(X)

        return X

    def _feat_prep(self, ind, X):
        # 'FeatPrep'     : 'none | Decomp | FeatSel',
        method = ind.choose(None, self._decompose, self._feat_sel)

        if method:
            X = method(ind, X)

        return X

    def _decompose(self, ind, X):
        # 'Decomp'       : 'FastICA | PCA | TruncSVD | KernelPCA',
        method = ind.choose(self._fastica, self._pca, self._truncsvd, self._kpca)
        return method(ind, X)

    def _ncomp(self, ind, X):
        return max(2, int(ind.nextfloat() * min(X.shape)))

    def _fastica(self, ind, X):
        # 'FastICA'      : 'i(2,100)',
        if hasattr(X, 'toarray'):
            X = X.toarray()

        return FastICA(n_components=self._ncomp(ind, X)).fit_transform(X)

    def _pca(self, ind, X):
        # 'PCA'          : 'i(2,100)',
        if hasattr(X, 'toarray'):
            X = X.toarray()

        return PCA(n_components=self._ncomp(ind, X)).fit_transform(X)

    def _truncsvd(self, ind, X):
        # 'TruncSVD'     : 'i(2,100)',
        return TruncatedSVD(n_components=self._ncomp(ind, X)).fit_transform(X)

    def _kpca(self, ind, X):
        # 'KernelPCA'    : 'KPCAn | KPCAk',
        # 'KPCAn'        : 'f(0.01,0.5)' ,
        # 'KPCAk'        : 'linear | poly | rbf | sigmoid | cosine',
        return KernelPCA(n_components=self._ncomp(ind, X),
                         kernel=ind.choose('linear', 'poly', 'rbf', 'sigmoid', 'cosine')).fit_transform(X)

    def _feat_sel(self, ind, X):
        # 'FeatSel'      : 'FeatAgg | Poly | Nystrom ',
        method = ind.choose(self._featagg, self._poly, self._nystrom)
        return method(ind, X)

    def _featagg(self, ind, X):
        # 'FeatAgg'      : 'f(0.01,0.5)',
        if hasattr(X, 'toarray'):
            X = X.toarray()

        return FeatureAgglomeration(n_clusters=self._ncomp(ind, X)).fit_transform(X)

    def _poly(self, ind, X):
        # 'Poly'         : 'i(2,3)',
        return PolynomialFeatures(degree=ind.nextint()).fit_transform(X)

    def _nystrom(self, ind, X):
        # 'Nystrom'      : 'f(0.01,0.5)',
        return Nystroem(n_components=self._ncomp(ind, X)).fit_transform(X)

    def _classifier(self, ind, balance):
        # 'Class'        : 'Bayes | Linear | SVC | Tree | KNN | Discriminat | MLP
        return ind.choose(self._bayes,
                          self._linear,
                          self._svc,
                          self._tree,
                          self._knn,
                          self._discr,
                          self._mlp)(ind, balance)

    def _bayes(self, ind, balance):
        # 'Bayes'        : 'gaussNB | mNB | cNB | nNB',
        return ind.choose(GaussianNB, MultinomialNB, ComplementNB, BernoulliNB)()

    def _linear(self, ind, balance):
        # 'Linear'       : 'SGD | Ridge | PA | LR | Lasso | Perceptron',
        return ind.choose(self._sgd, self._ridge, self._pa, self._lr, self._lasso, self._perceptron)(ind, balance)

    def _sgd(self, ind, balance):
        # 'SGD'          : 'hinge | log | modified_huber | squared_hinge | perceptron',
        loss = ind.choose('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron')
        return SGDClassifier(loss=loss,
                             class_weight=balance)

    def _ridge(self, ind, balance):
        # 'Ridge'        : 'f(0.01, 10)',
        return RidgeClassifier(alpha=ind.nextfloat(),
                               class_weight=balance)

    def _pa(self, ind, balance):
        # 'PA'           : 'f(0.01, 10)',
        return PassiveAggressiveClassifier(C=ind.nextfloat(),
                                           class_weight=balance)

    def _lr(self, ind, balance):
        # 'LR'           : 'LRloss LRreg',
        # 'LRloss'       : 'l1 | l2',
        # 'LRReg'        : 'f(0.01, 10)',
        return LogisticRegression(penalty=ind.choose('l1', 'l2'),
                                  C=ind.nextfloat(),
                                  solver='saga',
                                  class_weight=balance)

    def _lasso(self, ind, balance):
        # 'Lasso'        : 'f(0.01, 10)',
        return Lasso(alpha=ind.nextfloat())

    def _perceptron(self, ind, balance):
        # 'Perceptron'   : 'l1 | l2 | elasticnet',
        return Perceptron(penalty=ind.choose('l1', 'l2', 'elasticnet'))

    def _svc(self, ind, balance):
        # 'SVC'          : 'LinearSVC | KernelSVC',
        return ind.choose(self._linearsvc, self._kernelsvc)(ind, balance)

    def _linearsvc(self, ind, balance):
        # 'LinearSVC'    : 'LinearSVCp | LinearSVCl | LinearSVCr',
        # 'LinearSVCp'   : 'l1 | l2',
        # 'LinearSVCr'   : 'f(0.01,10)',
        return LinearSVC(penalty=ind.choose('l1', 'l2'),
                         C=ind.nextfloat(),
                         dual=False,
                         class_weight=balance)

    def _kernelsvc(self, ind, balance):
        # 'KernelSVC'    : 'KernelSVCk | KernelSVCr',
        # 'KernelSVCk'   : 'rbf | poly | sigmoid',
        # 'KernelSVCr'   : 'f(0.01,10)',
        return SVC(kernel=ind.choose('rbf', 'poly', 'sigmoid'),
                   C=ind.nextfloat(),
                   class_weight=balance,
                   gamma='auto')

    def _tree(self, ind, balance):
        # 'Tree'         : 'gini | entropy',
        return DecisionTreeClassifier(criterion=ind.choose('gini', 'entropy'),
                                      class_weight=balance)

    def _knn(self, ind, balance):
        # 'KNN'          : 'i(1,10)',
        return KNeighborsClassifier(n_neighbors=ind.nextint())

    def _discr(self, ind, balance):
        # 'Discriminant' : 'qda | lda',
        return ind.choose(QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis)()

    def _mlp(self, ind, balance):
        # 'MPL'          : 'MLPn | MLPl | MLPla',
        # 'MLPn'         : 'i(10,100)',
        # 'MLPl'         : 'i(1,5)',
        # 'MPLa'         : 'identity | logistic | tanh | relu',
        neurons = ind.nextint()
        layers = ind.nextint()
        activation = ind.choose('identity', 'logistic', 'tanh', 'relu')
        return MLPClassifier(hidden_layer_sizes=[neurons] * layers, activation=activation)


class SklearnNLPGrammar(SklearnGrammar):
    def __init__(self, X, y, *args, **kwargs):
        super().__init__(X=X, y=y, *args, **kwargs)

        print("Loading spacy...", end="", flush=True)
        self.nlp = spacy.load('en')
        print("done")

        print("Preprocessing sentences...", flush=True)
        self.sentences = [self.nlp(s) for s in tqdm.tqdm(X)]

    def grammar(self):
        g = {}
        g.update(grammar)
        g.update({
            'Pipeline'  : 'TextPrep DataPrep FeatPrep Class',
            'TextPrep'  : 'Clean Semantic Vect',

            'Encoding'  : 'none',

            'Clean'     : 'Stopwords',
            'Stopwords' :'yes | no',

            'Semantic'  : 'Pos Tag Dep',
            'Pos'       : 'yes | no',
            'Tag'       : 'yes | no',
            'Dep'       : 'yes | no',

            'Vect'      : 'CV | TF | TFIDF',
            'CV'        : 'i(1,3)',
            'TF'        : 'i(1,3)',
            'TFIDF'     : 'i(1,3)',
        })

        return g

    def evaluate(self, ind):
        # 'Pipeline'     : 'TextPrep DataPrep FeatPrep Class',
        X, y = self.X, self.y
        X = self._text_prep(ind, X)
        X, balance = self._data_prep(ind, X)
        X = self._feat_prep(ind, X)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

        classifier = self._classifier(ind, balance)

        try:
            classifier.fit(Xtrain, ytrain)
        except TypeError as e:
            if 'sparse' in str(e) and hasattr(Xtrain, 'toarray'):
                Xtrain = Xtrain.toarray()
                Xtest = Xtest.toarray()
                classifier.fit(Xtrain, ytrain)
            else:
                raise e
        except ValueError as e:
            if 'must be non-negative' in str(e):
                raise InvalidPipeline()
            raise e

        return classifier.score(Xtest, ytest)

    def _encoding(self, ind, X):
        return X

    def _text_prep(self, ind, X):
        # 'TextPrep'  : 'Clean Vect Semantic',
        sw = self._clean(ind)
        F = self._semantic(ind, X)
        X = self._vect(ind, X, sw)

        if F is None:
            return X

        if isinstance(X, np.ndarray):
            return np.hstack((X, F))
        else:
            return sp.hstack((X, F))

    def _clean(self, ind):
        # preprocesamiento
        if ind.nextbool():
            return stopwords.words('english')

        return set()

    def _semantic(self, ind, X):
        use_pos = ind.nextbool()
        use_tag = ind.nextbool()
        use_dep = ind.nextbool()

        if not any((use_pos, use_tag, use_dep)):
            return None

        features = []

        for sentence in self.sentences:
            counter = Counter()
            for token in sentence:
                if use_pos:
                    counter[token.pos_] += 1
                if use_tag:
                    counter[token.tag_] += 1
                if use_dep:
                    counter[token.dep_] += 1
            features.append(counter)

        self.dv = DictVectorizer()
        return self.dv.fit_transform(features)

    def _vect(self, ind, X, sw):
        vect = ind.choose(self._cv, self._tf, self._tfidf)
        ngram = ind.nextint()
        v = vect(ngram, sw)

        return v.fit_transform(X)

    def _cv(self, ngram, sw):
        return CountVectorizer(stop_words=sw, ngram_range=(1, ngram))

    def _tf(self, ngram, sw):
        return TfidfVectorizer(stop_words=sw, ngram_range=(1, ngram), use_idf=False)

    def _tfidf(self, ngram, sw):
        return TfidfVectorizer(stop_words=sw, ngram_range=(1, ngram), use_idf=True)


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, popsize=100, select=0.2, learning=0.25, iters=100, fitness_evaluations=1, timeout=None, verbose=False, global_timeout=None):
        self.popsize = popsize
        self.select = select
        self.learning = learning
        self.iters = iters
        self.timeout = timeout
        self.verbose = verbose
        self.fitness_evaluations = fitness_evaluations
        self.global_timeout = global_timeout

    def fit(self, X, y):
        self.grammar_ = SklearnGrammar(X, y)
        ge = PGE(self.grammar_, popsize=self.popsize, selected=self.select, learning=self.learning, timeout=self.timeout, verbose=self.verbose, fitness_evaluations=self.fitness_evaluations, global_timeout=self.global_timeout)
        self.best_ = ge.run(self.iters)
        self.best_sample_ = self.best_.sample()

        self.best_.reset()
        self.classifier_ = self.grammar_.train(self.best_, X, y)

    def predict(self, X):
        self.best_.reset()
        X = self.grammar_.process(self.best_, X)
        return self.classifier_.predict(X)


class SklearnNLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, popsize=100, select=0.2, learning=0.25, iters=100, timeout=None, fitness_evaluations=1, verbose=False):
        self.popsize = popsize
        self.select = select
        self.learning = learning
        self.iters = iters
        self.timeout = timeout
        self.verbose = verbose
        self.fitness_evaluations = fitness_evaluations

    def fit(self, X, y):
        self.grammar_ = SklearnNLPGrammar(X, y)
        ge = PGE(self.grammar_, popsize=self.popsize, selected=self.select, learning=self.learning, timeout=self.timeout, verbose=self.verbose, fitness_evaluations=self.fitness_evaluations)
        self.best_ = ge.run(self.iters)
        self.best_sample_ = self.best_.sample()

        self.best_.reset()
        self.classifier_ = self.grammar_.train(self.best_, X, y)

    def predict(self, X):
        self.best_.reset()
        X = self.grammar_.process(self.best_, X)
        return self.classifier_.predict(X)
