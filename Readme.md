# Hierarchical Pipeline Optimization

To try the examples run:

```bash
python -m hpopt.examples.movie_reviews
```

## Requirements

Basic requirements are `Python 3.5` or greater.

The `sklearn_opinion` example requires `sklearn`, `nltk` and the `movie_reviews` corpus.
To install these requirements, follow the instructions [here](https://scikit-learn.org/stable/install.html)
and [here](https://www.nltk.org/install.html).

If you have `pip` installed, some quick steps are:

```python
pip install -U sklearn
pip install -U nltk
python

>>> import nltk
>>> nltk.download("movie_reviews")
```

## Docker support

We have added a `docker-compose.yml` configuration that will run the framework inside our custom machine learning image. To try it just type:

```bash
docker-compose up
```

## How to cite

Please cite this work with the following this preliminary BibTeX:
```
@inproceedings{estevesvelarde-etal-2019-hp-optimization,
    title = "{AutoML} strategy based on grammatical evolution: A case study about knowledge discovery from text",
    author = "Estevez-Velarde, Suilan and Guti{\'e}rrez, Yoan and Montoyo, Andr{\'e}s and Almeida-Cruz, Yudivi{\'a}n",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "",
    pages = "",
    abstract = "The process of extracting knowledge from natural language text poses a complex problem that requires both a combination of machine learning techniques and proper feature selection. Recent advances in Automatic Machine Learning (AutoML) provide effective tools to explore large sets of algorithms, hyper-parameters and features to find out the most suitable combination of them. This paper proposes a novel AutoML strategy based on probabilistic grammatical evolution, which is evaluated on the health domain by facing the knowledge discovery challenge in Spanish text documents. Our approach achieves state-of-the-art results and provides interesting insights into the best combination of parameters and algorithms to use when dealing with this challenge. Source code is provided for the research community."
}

```

## License

Licensed under the [MIT](https://opensource.org/licenses/MIT) open source license.

> MIT License
>
> Copyright (c) 2019 Knowledge Learning Project
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
