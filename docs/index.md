---
title: Home
permalink: /
---

# Hierarchical Pipeline Optimization

`hpopt` stands for Hierarchical Pipeline Optimization, a Python module for automatic configuration of software pipelines, specifically, machine learning pipelines.

A pipeline is simply a sequence or chain of processes (the pipeline steps) that iteratively transforms an input into an output.
A machine learning pipeline is often composed of steps such as feature extraction, preprocessing, vectorization, dimensionality reduction and classification (or regression). These pipelines often have many configurable _hyper-parameters_, like the actual classification algorithm to use, or the strength of a regularization factor. `hpopt` gives you tools to define custom pipelines, with any degree of complexity, and automatically find the best configuration for a given problem (read, for a given dataset).
This task (finding the optimal configuration of machine learning pipeline) is often called **Auto-ML**.

The hierarchical part comes from the fact that in `hpopt` you define the set of all possible pipelines using a hierarchical structure. In formal terms it is a context-free grammar, but informally, it is basically a hierarchical definition that starts top-level steps (such as preprocessing, vectorization, etc.), which are themselves defined recursively in terms of simpler steps, down to algorithms and their hyper-parameters.

If you want to know more about the inner workings of `hpopt` we recommend reaing [our paper](https://www.aclweb.org/anthology/papers/P/P19/P19-1428/). Here is the source code [click here](https://github.com/knowledge-learning/hp-optimization)


## Quick start

`hpopt` can be used in many different ways, with varying degrees of customization. The easiest way to use `hpopt` is a black-box Auto-ML tool.
`hpopt` provides several `scikit-learn` compatible estimators that you can readily use:

```python
from hpopt.sklearn import SklearnClassifier

X, y = get_dataset() # <-- custom data logic
clf = SklearnClassifier()
clf.fit(X, y)
```

<!-- By default, `hpopt` will search for an optimal configuration... -->
