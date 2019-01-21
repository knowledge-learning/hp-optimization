# coding: utf-8

__all__ = ['SklearnGrammar']

from sklearn.base import BaseEstimator, ClassifierMixin

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
            X = OneHotEncoder(categories='auto').fit_transform(X)

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
        return max(2, int(ind.nextfloat() * X.shape[1]))

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


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, popsize=100, select=0.2, learning=0.25, iters=100, timeout=None, verbose=False):
        self.popsize = popsize
        self.select = select
        self.learning = learning
        self.iters = iters
        self.timeout = timeout
        self.verbose = verbose

    def fit(self, X, y):
        self.grammar_ = SklearnGrammar(X, y)
        ge = PGE(self.grammar_, popsize=self.popsize, selected=self.select, learning=self.learning, timeout=self.timeout, verbose=self.verbose)
        self.best_ = ge.run(self.iters)
        self.best_sample_ = self.best_.sample()

        self.best_.reset()
        self.classifier_ = self.grammar_.train(self.best_, X, y)

    def predict(self, X):
        self.best_.reset()
        X = self.grammar_.process(self.best_, X)
        return self.classifier_.predict(X)
