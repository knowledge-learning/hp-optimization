import random
from ..pbil import PBIL


best = [round(random.uniform(0,1),2) for i in range(10)]


def fitness_function(solt):
    return sum(1-abs(x-y) for x,y in zip(best, solt))


def main():
    pbil = PBIL(100, 20, 0.1, [2] * 10, fitness_function)
    top = pbil.run(100)
    print(f"Expected:   {best}")
    print(f"Best found: {top}")


if __name__ == '__main__':
    main()
