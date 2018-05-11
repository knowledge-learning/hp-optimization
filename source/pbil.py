# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 04:22:57 2015

@author: suilan
"""

import random

from metaheuristic import Metaheuristic


class PBIL(Metaheuristic):

    def __init__(self, popsize, indiv_count, learn_rate, values_genes):
        """
        popsize = tamaño de la población
        indiv_count = cantidad de individuos a seleccionar para crear la nueva distribución
        learn_rate = factor para hacer update a la distribución
        values_genes = es una lista con la cantidad de valores que puede tomar cada gen
        """
        self.popsize = popsize
        self.indiv_count = indiv_count
        self.learn_rate = learn_rate
        self.distributions = []
        self.step = 0
        self.population = []

        #mejor solución encontrada y su evaluación
        self.best = self.__sample_distribution__(self.distributions)
        self.bestfitness = 0

        for i in values_genes:
            self.distributions.append([1.0/i]*i)

    def fitness(self, solt):
        """Evalúa la solución y devuelve la precisión"""
        raise NotImplementedError("You have to implement this method.")

    def run(self, evals):
        """Corre la metaheurística hasta el número de evaluaciones indicado"""

        while self.step < evals:
            #fin de generación
            if len(self.population) == self.popsize:
                p = self.population
                #selección por truncamiento
                #se dejan solo los indiv_count mejores
                p.sort(key=lambda x: x[1], reverse=True)
                p = p[:self.indiv_count]

                #calculando la distribución a partir de los individuos de p
                n = self.__calculate_distribution__(p)
                #le hace update a las distribuciones guardadas
                #dj = (1-learn_rate)*dj + learn_rate*n
                self.distributions = self.__sum__(self.__mult__(1-self.learn_rate,self.distributions),
                                                      self.__mult__(self.learn_rate,n))

                #cambiar de generación
                self.step += 1
                self.population = []

                print("Generation: {0}".format(self.step))

            #individuo construido construido para cada gen j random
            #por debajo de la distribución Dj
            indiv = self.__sample_distribution__(self.distributions)
            print(indiv)
            fitn = self.fitness(indiv)

            #se encontró una solución mejor que el óptimo
            if fitn > self.bestfitness:
                self.best = indiv
                self.bestfitness = fitn

                with open("best_pbil.txt", "a") as fp:
                    fp.write("%s\n%s\n" % (self.best, self.bestfitness))

            self.population.append((indiv, fitn))

            #salvar a este individuo
            with open("experiments_pbil.txt", "a") as fp:
                fp.write("%s\n%s\n" % (indiv, fitn))

            #salvar el algoritmo
            self.save()

        return self.best

    def __sample_distribution__(self, distributions):
        """Devuelve un individuo random que cumpla con la distribucion"""

        pos = 0
        indiv = []

        for g in distributions:
            r = random.uniform(0,1)
            idx = len(g) - 1
            for i,v in enumerate(g):
                pos += v
                if r <= pos:
                    idx = i
                    break
            indiv.append(idx)
            pos = 0

        return indiv

    def __calculate_distribution__(self, individuals):
        """Calcula para cada gen la distribucion utilizando
        la informacion de la poblacion"""

        # creando una copia en blanco de la estructura de las distribuciones
        new_distributions = self.__clean_distribution__()
        #print(new_distributions)

        # actualizando el valor de la nueva distribucion
        for i,f in individuals:
            for g,v in enumerate(i):
                new_distributions[g][v] += 1.0/len(individuals)

        return new_distributions

    def __mult__(self, val, dist):
        """Multiplica un valor por la estructura de las distribuciones"""

        for g in range(len(dist)):
            for v in range(len(dist[g])):
                dist[g][v] *= val

        return dist

    def __sum__(self, dist1, dist2):
        """Multiplica un valor por la estructura de las distribuciones"""

        resul_dist = self.__clean_distribution__()

        for g in range(len(dist1)):
            for v in range(len(dist1[g])):
                resul_dist[g][v] = dist1[g][v] + dist2[g][v]

        return resul_dist

    def __clean_distribution__(self):
        """Crea una copia en blanco de la estructura de las distribuciones"""

        new_distributions = []

        for d in self.distributions:
            new_distributions.append([0]*len(d))

        return new_distributions

    def __save_distribution__(self):
        """Salva en un archivo la estructura de las distribuciones"""

        f = open("distribution.txt","a")
        f.write(str(self.distributions) + "/n")
        f.close()
