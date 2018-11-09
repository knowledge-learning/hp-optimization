# PBIL for pipeline optimization

To try the examples run

```
python -m pbil.examples.simple_function
```

Or

```
python -m pbil.examples.sklearn_opinion
```

## Requirements

Basic requirements are `Python 3.5` or greater.

The `sklearn_opinion` example requires `sklearn`, `nltk` and the `movie_review` corpus.
To install these requirements, follow the instructions [here](https://scikit-learn.org/stable/install.html)
and [here](https://www.nltk.org/install.html).

If you have `pip` installed, some quick steps are:

```
pip install -U sklearn
pip install -U nltk
python

>>> import nltk
>>> nltk.download("movie_reviews")
```

## License

Licensed under the [MIT](https://opensource.org/licenses/MIT) open source license.
