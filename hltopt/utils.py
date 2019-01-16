def pow2value(number):
    """Devuelve una lista de potencia de 2 hasta el número pasado"""
    n = 1
    while n<number:
        n = 2*n
        yield n

def pow2(exp):
    """Devuelve una lista de potencia de 2 con el exponente pasado"""
    n = 1
    e = 1
    while e<=exp:
        n = 2*n
        e += 1
        yield n

def convert_BtoI(vector):
    """Convierte un vector booleano a un vector de ceros y unos"""
    result = []
    for i in vector:
        if i == True:
            result.append(1)
        else:
            result.append(0)
    return result

def convert_ItoB(vector):
    """Convierte un vector de ceros y unos a un vector booleano"""
    result = []
    for i in vector:
        if i == 1:
            result.append(True)

        else:
            result.append(False)
    return result

def index(vector, count):
    value = 0
    newvector = convert_BtoI(vector)

    l = pow2(len(vector))
    l = list(l)

    for i in range(len(vector)):
        value += newvector[i]*(1.0/l[i])

    return int(value * count)


class InvalidPipeline(ValueError):
    """Raise when a pipeline is not valid after construction."""


def szip(*items):
    sizes = set(len(i) for i in items)

    if len(sizes) != 1:
        raise ValueError("All collections should be the same size.")

    return zip(*items)

def sdiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
