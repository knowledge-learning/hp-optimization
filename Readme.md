# Hierarchical Pipeline Optimization

To try the examples run:

```bash
python -m hltopt.examples.movie_reviews
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
