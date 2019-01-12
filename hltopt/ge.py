# -*- coding: utf-8 -*-

import yaml
import random
import numpy as np
import json

from .metaheuristic import Metaheuristic
from random import gauss


class InvalidPipeline(ValueError):
    """Raise when a pipeline is not valid after construction."""


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


class Individual:
    def __init__(self, values):
        self._values = [min(0.99999,max(0.00001,v)) for v in values]
        self._current = 0

    def reset(self):
        self._current = 0

    def value(self):
        return self._values[self._current]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def _advance(self):
        self._current += 1

        if self._current >= len(self._values):
            self._current = 0

    def nextint(self, n:int):
        i = int(self._values[self._current] * n)
        self._advance()
        return i

    def nextbool(self):
        return self.nextint(2) == 0

    def nextfloat(self, a:float, b:float):
        x = self._values[self._current] * (b-a) + a
        self._advance()
        return x

    def choose(self, *choices):
        return choices[self.nextint(len(choices))]

    def __repr__(self):
        return "Individual({0})".format(self._values)


class PIndividual:
    """Representa un individuo de una gramática probabilística."""
    def __init__(self, values, grammar):
        self.values = values
        self.current = 0
        self.grammar = grammar
        self.state = self._sample('Pipeline')

    def reset(self):
        self.current = 0
        self.state = self._sample('Pipeline')

    def choose(self, *values):
        value = next(self.state)

        if not isinstance(value, tuple):
            raise ValueError('Cannot apply `choose` at this point (%s).' % str(value))

        options, i = value

        if options != len(values):
            raise ValueError('Need to provide exactly %i values.' % options)

        return values[i]

    def next(self):
        value = self.values[self.current]
        self.current += 1
        return value

    def nextint(self):
        value = next(self.state)

        if not isinstance(value, int):
            raise ValueError('Cannot apply `nextint` at this point (%s).' % str(value))

        return value

    def nextfloat(self):
        value = next(self.state)

        if not isinstance(value, float):
            raise ValueError('Cannot apply `nextfloat` at this point (%s).' % str(value))

        return value

    def sample(self, symbol='Pipeline'):
        production = self.grammar[symbol]
        value = production.sample(self)

        if isinstance(value, (int, float)):
            return value
        else:
            rule, options, index = value

            rule_repr = []

            for s in rule.body:
                if s[0].isupper():
                    rule_repr.append({ s: self.sample(s) })
                else:
                    rule_repr.append(s)

            return rule_repr

    def _sample(self, symbol):
        production = self.grammar[symbol]
        value = production.sample(self)

        if isinstance(value, (int, float)):
            yield value
        else:
            rule, options, index = value

            if options > 1:
                yield options, index

            for s in rule.body:
                if s[0].isupper():
                    yield from self._sample(s)



class GE(Metaheuristic):
    def __init__(self, grammar, popsize=100, selected=0.1, rate=0.9):
        """Representa una metaheurística de Evolución Gramatical.

        - `popsize`: tamaño de la población
        - `indsize`: tamaño de cada individuo
        - `selected`: cantidad de individuos seleccionados en cada paso
        """
        super().__init__()

        self._grammar = grammar
        self.popsize = popsize
        self.indsize = self._grammar.complexity()
        self.rate = rate

        if isinstance(selected, float):
            selected = int(selected * popsize)

        self.selected = selected

    def _init_population(self):
        """Construye la población inicial"""
        population = []

        for _ in range(self.popsize):
            values = []
            for _ in range(self.indsize):
                values.append(random.uniform(0,1))
            population.append(Individual(values))

        list_distances = []
        for a in population:
            for b in population:
                list_distances.append(self._grammar.distance(a,b))

        # self.threshold = sorted(list_distances)[len(list_distances)//2]
        self.threshold = max(list_distances) / 2
        print("Initial threshold:", self.threshold)

        return population

    def _select(self, pop, fit):
        """Selecciona los mejores {self.indsize} elementos de la población."""
        sorted_pop = sorted(zip(pop,fit), key=lambda t: t[1], reverse=True)
        return [t[0] for t in sorted_pop[:self.selected]]

    def _breed(self, pop):
        """Construye una nueva población de tamaño {self.popsize} a partir de los individuos en pop."""
        new_pop = []

        while len(new_pop) < self.popsize:
            parent = pop.pop(0)
            new_pop.append(self._mutate(parent))
            pop.append(parent)

        return new_pop

    def _mutate(self, ind:Individual) -> Individual:
        """Construye un nuevo individuo mutado a partir de `ind`."""

        lmin = 0
        lmax = 1
        iters = 0

        while True:
            iters += 1
            print('.', end='')

            mutation = make_rand_vector(len(ind))
            lmid = (lmin + lmax)/2
            new_ind = Individual([v + lmid*x for v,x in zip(ind,mutation)])
            dist = self._grammar.distance(ind, new_ind)

            if iters > 100:
                print('*')
                return new_ind

            if dist < self.threshold:
                lmin = lmid
            elif dist > 2 * self.threshold:
                lmax = lmid
            else:
                print('x')
                return new_ind

    def _evaluate(self, ind:Individual):
        """Computa el fitness de un individuo."""

        print(yaml.dump(self._grammar.sample(ind)))

        try:
            ind.reset()
            f = self._grammar.evaluate(ind)
        except InvalidPipeline as e:
            print(str(e))
            f = 0

        return f

    def run(self, evals:int):
        """Ejecuta la metaheurística hasta el número de evaluaciones indicado"""

        self.it = 0
        self.population = self._init_population()
        self.fitness = [self._evaluate(i) for i in self.population]
        self.current_best, self.current_fn = None, 0

        while self.it < evals:
            best_individuals = self._select(self.population, self.fitness)
            self.population = self._breed(best_individuals)
            self.fitness = [self._evaluate(i) for i in self.population]

            GEEncoder.grammar = self._grammar
            self.save(GEEncoder)

            for ind, fn in zip(self.population, self.fitness):
                if fn > self.current_fn:
                    self.current_best = ind
                    self.current_fn = fn
                    print("Updated best: ", self.current_fn)

            self.threshold *= self.rate
            self.it += 1
            print("Threshold:", self.threshold)

        return self.current_best


class PGE(Metaheuristic):
    def __init__(self, grammar, popsize=100, selected=0.1, learning=0.9):
        """Representa una metaheurística de Evolución Gramatical Probabilística.

        - `popsize`: tamaño de la población
        - `selected`: cantidad de individuos seleccionados en cada paso
        - `learning`: factor de aprendizaje para ajustar las probabilidades
        """
        super().__init__()

        self._grammar = grammar
        self.popsize = popsize
        self.indsize = self._grammar.complexity()
        self.learning = learning

        if isinstance(selected, float):
            selected = int(selected * popsize)

        self.selected = selected

    def _sample_population(self):
        """Construye la población inicial"""
        population = []

        for _ in range(self.popsize):
            values = []
            for _ in range(self.indsize):
                values.append(random.uniform(0,1))
            population.append(PIndividual(values, self._grammar))

        return population

    def _select(self, pop, fit):
        """Selecciona los mejores {self.indsize} elementos de la población."""
        sorted_pop = sorted(zip(pop,fit), key=lambda t: t[1], reverse=True)
        return [t[0] for t in sorted_pop[:self.selected]]

    def _update_model(self, best):
        for ind in best:
            ind.reset()
            pipe = ind.sample()

            print(pipe)

    def _evaluate(self, ind:PIndividual):
        """Computa el fitness de un individuo."""

        print(yaml.dump(ind.sample()))

        try:
            ind.reset()
            f = self._grammar.evaluate(ind)
        except InvalidPipeline as e:
            print(str(e))
            f = 0

        return f

    def run(self, evals:int):
        """Ejecuta la metaheurística hasta el número de evaluaciones indicado"""

        it = 0
        self.current_best, self.current_fn = None, 0

        while it < evals:
            self.population = self._sample_population()
            self.fitness = [self._evaluate(i) for i in self.population]

            GEEncoder.grammar = self._grammar
            self.save(GEEncoder)

            for ind, fn in zip(self.population, self.fitness):
                if fn > self.current_fn:
                    self.current_best = ind
                    self.current_fn = fn
                    print("Updated best: ", self.current_fn)

            best = self._select(self.population, self.fitness)
            self._update_model(best)
            it += 1

        return self.current_best


class GEEncoder(json.encoder.JSONEncoder):
    grammar = None

    def default(self, obj):
        if isinstance(obj, Individual):
            obj.reset()
            enc = GEEncoder.grammar.sample(obj)
            obj.reset()
            return enc


class GrammarGE:
    def evaluate(self, element) -> float:
        """Recibe un elemento de la gramática y devuelve un valor de fitness creciente."""
        pass

    def grammar(self):
        raise NotImplementedError()

    def parse(self):
        """Devuelve la gramática parseada como un árbol para que sea fácil de interpretar."""
        return self._parse_symbol(self.grammar(), 'Pipeline')

    def complexity(self):
        """Calcula la máxima complejidad de una solución en la gramática."""
        tree = self.parse()
        return self._complexity(tree)

    def importance(self, i:Individual):
        """Computa la importancia de cada componente del individuo en la gramática."""
        return self._importance(i, [], 'Pipeline', self.parse(), 0)

    def distance(self, a:Individual, b:Individual):
        return self._distance(a, 'Pipeline', self.parse(), b, 'Pipeline', self.parse(), 1)

    def _distance(self, a:Individual, asymb, agram, b:Individual, bsymb, bgram, alpha):
        # quedarme con los símbolos en sí
        if isinstance(asymb, dict):
            asymb = list(asymb.keys())[0]
        if isinstance(bsymb, dict):
            bsymb = list(bsymb.keys())[0]

        # ambos tienen que ser iguales, o la hemos liado
        assert asymb == bsymb

        # si ambos son hojas de valores continuos
        if asymb[0:1] in ['i(', 'f(']:
            return abs(a.value() - b.value()) * alpha

        # si ambos son hojas de valores discretos
        if asymb[0].islower():
            if asymb == bsymb:
                # si son iguales
                return 0 # abs(a.value() - b.value()) * alpha
            else:
                # si son distintos la distancia es máxima
                return alpha

        # si son no-terminales, cogemos la producción
        aprods = agram[asymb]
        bprods = bgram[bsymb]
        n = len(aprods)

        # calcular por dónde baja cada uno
        if len(aprods) == 1:
            aprod = aprods[0]
            bprod = bprods[0]
            equal = True
        else:
            asel = a.nextint(n)
            bsel = b.nextint(n)
            aprod = aprods[asel]
            bprod = bprods[bsel]
            equal = asel == bsel

        # si son iguales, bajar recursivamente
        if equal:
            dist = 0
            # computar la distancia entre los subárboles
            for i, (sa, sb) in enumerate(zip(aprod, bprod)):
                dist += self._distance(a, sa, aprod[i], b, sb, bprod[i], alpha / n)
            return dist

        # si no son iguales, la diferencia es máxima
        # pero hay que consumir los valores correspondientes
        self._sample(a, asymb, agram)
        self._sample(b, bsymb, bgram)

        return alpha

    def sample(self, ind:Individual):
        return self._sample(ind, 'Pipeline', self.parse())

    def _sample(self, ind:Individual, symbol, grammar):
        if isinstance(symbol, dict):
            symbol = list(symbol.keys())[0]

        if symbol[0:2] == 'i(':
            a,b = eval(symbol[1:])
            return ind.nextint(b - a) + a

        if symbol[0:2] == 'f(':
            a,b = eval(symbol[1:])
            return ind.nextfloat(a, b)

        if symbol in ['yes', 'no']:
            return symbol == 'yes'

        if symbol[0].islower():
            return symbol

        productions = grammar[symbol]
        n = len(productions)

        if n == 1:
            prod = productions[0]
        else:
            prod = productions[ind.nextint(n)]

        values = {}

        for i, s in enumerate(prod):
            sname = s if isinstance(s, str) else list(s.keys())[0]
            values[sname] = self._sample(ind, s, prod[i])

        if len(values) == 1:
            key = list(values.keys())[0]
            if key[0].islower():
                return values[key]

        return values

    def _importance(self, ind:Individual, imp, symbol, grammar, depth):
        if isinstance(symbol, dict):
            symbol = list(symbol.keys())[0]

        if symbol[0].islower():
            if symbol[0:2] in ['i(', 'f(']:
                imp.append(depth)

            return imp

        productions = grammar[symbol]

        if len(productions) > 1:
            selected = ind.nextint(len(productions))
            imp.append(depth)
            p = productions[selected]
        else:
            p = productions[0]

        for i, symb in enumerate(p):
            imp = self._importance(ind, imp, symb, p[i], depth + 1)

        return imp

    def _complexity(self, tree):
        if isinstance(tree, str):
            return 0

        key = list(tree.keys())[0]

        if key in ['integer', 'float']:
            return 1

        productions = tree[key]
        compl = max(self._complexity_prod(p) for p in productions)

        if len(productions) > 1:
            return 1 + compl
        else:
            return compl

    def _complexity_prod(self, prod):
        return sum(self._complexity(p) for p in prod)

    def _parse_symbol(self, grammar, symbol):
        productions = []

        for prod in grammar[symbol].split('|'):
            symbols = []

            for symb in prod.split():
                if symb[0].isupper():
                    symbols.append(self._parse_symbol(grammar, symb))
                elif symb[0:1] == 'i(':
                    symbols.append({
                        'integer': list(eval(symb[1:]))
                    })
                elif symb[0:1] == 'f(':
                    symbols.append({
                        'float': list(eval(symb[1:]))
                    })
                else:
                    symbols.append(symb)

            productions.append(symbols)

        return {
            symbol: productions
        }


class Production:
    def __init__(self, symbol, rules):
        self.symbol = symbol
        self.rules = rules

    def __repr__(self):
        return "Production(%s,%s)" % (self.symbol, repr(self.rules))

    def normalize(self):
        total = sum(r.prob for r in self.rules)
        for r in self.rules:
            r.prob /= total

    def complexity(self, grammar):
        if len(self.rules) == 1:
            return self.rules[0].complexity(grammar)

        return 1 + sum(r.complexity(grammar) for r in self.rules)

    def sample(self, ind:PIndividual):
        if len(self.rules) == 1:
            return self.rules[0], 1, 0

        value = ind.next()
        p = 0

        for i,r in enumerate(self.rules):
            p += r.prob
            if value <= p:
                return r, len(self.rules), i

        return self.rules[-1], len(self.rules), len(self.rules) - 1


class Rule:
    def __init__(self, body, prob:float):
        self.body = body
        self.prob = prob

    def __repr__(self):
        return "Rule(%s,%s)" % (repr(self.body), self.prob)

    def complexity(self, grammar):
        c = 0

        for symbol in self.body:
            if symbol[0].isupper():
                c += grammar.complexity(symbol)

        return c


class IntProduction(Production):
    def __init__(self, symbol, min:int, max:int):
        self.symbol = symbol
        self.min = min
        self.max = max
        self.mean = (max + min) / 2
        self.dev = (max - min) / 2

    def normalize(self):
        pass

    def complexity(self, grammar):
        return 1

    def sample(self, ind:PIndividual):
        value = int(self.mean + self.dev * (ind.next() - 0.5) * 2)
        return max(self.min, min(self.max, value))


class FloatProduction(IntProduction):
    def sample(self, ind:PIndividual):
        value = self.mean + self.dev * (ind.next() - 0.5) * 2
        return max(self.min, min(self.max, value))


class GrammarPGE:
    def __init__(self):
        self._grammar = {}

        for symbol, productions in self.grammar().items():
            self._grammar[symbol] = []
            productions = productions.split('|')

            if len(productions) == 1:
                p = productions[0]

                if p.startswith('f('):
                    min, max = tuple(float(i) for i in p[2:-1].split(','))
                    self._grammar[symbol] = FloatProduction(symbol, min, max)
                    continue
                if p.startswith('i('):
                    min, max = tuple(int(i) for i in p[2:-1].split(','))
                    self._grammar[symbol] = IntProduction(symbol, min, max)
                    continue

            rules = []

            for p in productions:
                if p.startswith('f(') or p.startswith('i('):
                    raise ValueError('Numeric rules must be the only ones.')

                rules.append(Rule(p.split(), 1))

            production = Production(symbol, rules)
            production.normalize()

            self._grammar[symbol] = production

    def __getitem__(self, key):
        return self._grammar[key]

    def evaluate(self, ind:PIndividual) -> float:
        """Recibe un elemento de la gramática y devuelve un valor de fitness creciente."""
        raise NotImplementedError()

    def grammar(self):
        raise NotImplementedError()

    def complexity(self, symbol='Pipeline'):
        """Calcula la máxima complejidad de una solución en la gramática."""
        return self._grammar[symbol].complexity(self)
