# coding: utf-8

import pprint
import random
import string
import nltk
import spacy

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from nltk.stem.snowball import SnowballStemmer

from ..ge import GrammarGE, GE, Individual


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
            
            'Repr'     : 'Prep Token SemFeat PosPrep MulWords Embed',
            'Prep'     : 'DelPunt StripAcc',
            'DelPunt'  : 'yes | no',
            'StripAcc' : 'yes | no',
            'Token'    : 'wordTok',
            'PosTag'   : 'yes | no',
            'Dep'      : 'yes | no',
            'PosPrep'  : 'StopW Stem',
            'Stem'     : 'yes | no',
            'StopW'    : 'yes | no',
            'SemFeat'  : 'PosTag Dep UMLS SNOMED',
            'UMLS'     : 'yes | no',
            'SNOMED'   : 'yes | no',
            'MulWords' : 'countPhrase | freeling | Ngram',
            'Ngram'    : 'i(2,4)',
            'Embed'    : 'WordVec SenseVec CharEmbed',
            'WordVec'  : 'yes | no',
            'SenseVec' : 'yes | no',
            'CharEmbed': 'yes | no',
        }

    def evaluate(self, i:Individual):
        
        def Pipeline(self, i):
            pass
            
        def ABC(self, i):
            return self.Class(i)
            
        def BC(self, i):
            return self.Class(i)
            
        def AB(self, i):
            if i.nextbool():
                return self.Class(i)  
            else:
                return self.Seq(i)

        def A(self, i):
            if i.nextbool():
                return self.Class(i)  
            else:
                return self.Seq(i)
            
        def B(self, i):
            return self.Class(i)
            
        def C(self, i):
            return self.Class(i)
            
        def Seq(self, i):
            if i.nextbool():
                #crf
                return None
            else:
                #hmm
                return None
                
        def Class(self, i):
            #LR | nb | SVM | dt | NN
            des = i.nextint(5)
            if des == 0:
                return self.LR(i)
            elif des == 1:
                return MultinomialNB()
            elif des == 2:
                return self.SVM(i)
            elif des == 3:
                return DecisionTreeClasifier()
            else:
                return self.NN(i)
        
        def LR(self, i):
            return LogisticRegression(C=self.Reg(i), penalty=self.Penalty(i))
            
        def Reg(self, i):
            return i.nextfloat(0.01, 100)
            
        def Penalty(self, i):
            return i.choose('l1', 'l2')
            
        def SVM(self, i):
            return SVC(kernel=self.Kernel(i))
            
        def Kernel(self, i):
            #linear | rbf | poly
            return i.choose('lineal', 'rbf', 'poly')
            
        def NN(self, i):
            pass
            
        def Drop(self, i):
            pass
            
        def CVLayers(self, i):
            pass
            
        def Count(self, i):
            pass
            
        def MinFilter(self, i):
            pass
            
        def MaxFilter(self, i):
            pass
            
        def FormatCon(self, i):
            pass
            
        def RLayers(self, i):
            pass
            
        def Size(self, i):
            pass
            
        def DLayers(self, i):
            pass
            
        def Act(self, i):
            pass
            
        def MinSize(self, i):
            pass
            
        def MaxSize(self, i):
            pass
            
        def FormatDen(self, i):
            pass
            
        def FLayer(self, i):
            pass
            
        def Repr(self, i):
            pass
            
        def Prep(self, i, texto):
            #'DelPunt StripAcc'
            met = self.DelPunt(i, texto)
            return self.StripAcc(i, met)
            
        def DelPunt(self, i, texto):
            #yes | no
            if i.nextbool():
                return s.translate(None, string.punctuation)
            else:
                return texto

        def StripAcc(self, i, texto):
            #yes | no
            if i.nextbool():
                pass #llamar al método de nltk (quitar acento)
            else:
                return texto
            
        def Token(self, i, texto):
            tokens = self.spacy_nlp(texto)
            return [t.text for text in tokens]

        def PosTag(self, i, texto):
            tokens = self.spacy_nlp(texto)
            return [t.pos_ + t.tag_ for t in tokens]
            
        def Dep(self, i, texto):
            tokens = self.spacy_nlp(texto)
            return [t.dep_ for t in tokens]
            
        def PosPrep(self, i):
            pass
            
        def Stem(self, i, tokens):
            new = []
            for t in tokens:
                new.append(self.stemmer.stem(t))
            return new
            
        def StopW(self, i):
            tokens = self.spacy_nlp(texto)
            return [t for t in tokens if not t.is_stop]
            
        def SemFeat(self, i, tokens):
            #'PosTag Dep UMLS SNOMED'
            new = {}
            new['postag'] = self.PosTag(i, texto)
            new['dep'] = self.Dep(i,texto)
            new['umls'] = self.UMLS()
            new['snomed'] = self.SNOMED()
            return new

        def UMLS(self, i):
            pass
            
        def SNOMED(self, i):
            pass
            
        def MulWords(self, i):
            pass
            
        def Ngram(self, i):
            pass
            
        def Embed(self, i):
            pass
            
        def WordVec(self, i):
            pass
            
        def SenseVec(self, i):
            pass
            
        def CharEmbed(self, i):
            pass
            


def main():
    grammar = MyGrammar()
    i = Individual([random.uniform(0,1) for i in range(100)])
    print(grammar.sample(i))


if __name__ == '__main__':
    main()