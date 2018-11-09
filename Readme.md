# Human Language Technologies Optimization Framework

To try the examples run

```bash
python -m hltopt.examples.simple_function
```

Or

```bash
python -m hltopt.examples.sklearn_opinion
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

## License

Licensed under the [MIT](https://opensource.org/licenses/MIT) open source license.
