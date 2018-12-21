from ..ge import GrammarGE, GE


class MyGrammar(GrammarGE):
    def grammar(self):
        return {
            'Pipeline': ''
        }


def main():
    ge = GE(MyGrammar())


if __name__ == "__main__":
    main()
