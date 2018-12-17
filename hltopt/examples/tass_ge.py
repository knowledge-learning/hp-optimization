# coding: utf-8

import pprint
import random

from ..ge import GrammarGE, GE, Individual


class MyGrammar(GrammarGE):
    def __init__(self):
        pass
 
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
            'LR'       : 'C Penalty',
            'C'        : 'f(0.01,100)',
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
            
            'Repr'     : 'Prep Token PosPrep SemFeat MulWords Embed',
            'Prep'     : 'DelPunt StripAcc',
            'DelPunt'  : 'yes | no',
            'StripAcc' : 'yes | no',
            'Token'    : 'PosTag Dep Stem',
            'PosTag'   : 'yes | no',
            'Dep'      : 'yes | no',
            'PosPrep'  : 'StopW Stem',
            'Stem'     : 'yes | no',
            'StopW'    : 'yes | no',
            'SemFeat'  : 'UMLS | SNOMED',
            'UMLS'     : 'yes | no',
            'SNOMED'   : 'yes | no',
            'MulWords' : 'countPhrase | freeling | Ngram',
            'Ngram'    : 'i(2,4)',
            'Embed'    : 'WordVec | SenseVec | CharEmbed',
            'WordVec'  : 'yes | no',
            'SenseVec' : 'yes | no',
            'CharEmbed': 'yes | no',
        }

    def evaluate(self, i:Individual):
        print(i)


def main():
    grammar = MyGrammar()

    print(grammar.complexity())


if __name__ == '__main__':
    main()