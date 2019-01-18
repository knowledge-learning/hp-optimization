# coding: utf-8

import numpy as np
import bisect
import functools
from pathlib import Path
from ...utils import szip
from sklearn.feature_extraction import DictVectorizer


def cached(func):
    result = None
    @functools.wraps(func)
    def f(*args, **kwargs):
        nonlocal result
        if result is None:
            result = func(*args, **kwargs)
        return result
    return f


relation_mapper = DictVectorizer().fit({r:True} for r in "is-a same-as part-of property-of subject target".split())


class TassDataset:
    def __init__(self, sentences=None, validation_size=0, max_length=0):
        self.sentences = sentences or []
        self.validation_size = validation_size
        self.max_length = max_length

    def clone(self):
        return TassDataset([s.clone() for s in self.sentences], self.validation_size)

    def by_word(self):
        raise NotImplementedError("Need to specialize dataset for a task first.")

    def by_sentence(self):
        raise NotImplementedError("Need to specialize dataset for a task first.")

    def __len__(self):
        return len(self.sentences)

    def load(self, finput:Path):
        goldA = finput.parent / ('output_A_' + finput.name[6:])
        goldB = finput.parent / ('output_B_' + finput.name[6:])
        goldC = finput.parent / ('output_C_' + finput.name[6:])

        text = finput.open().read()
        sentences = [s for s in text.split('\n') if s]

        self._parse_ann(sentences, goldA, goldB, goldC)

        return len(sentences)

    def _parse_ann(self, sentences, goldA, goldB, goldC):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        sentences_obj = [Sentence(text) for text in sentences]
        labels_by_id = {}
        sentence_by_id = {}

        for line in goldB.open():
            lid, lbl = line.split()
            labels_by_id[int(lid)] = lbl

        for line in goldA.open():
            lid, start, end = (int(i) for i in line.split())

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, start)
            # correct the annotation spans
            if i > 0:
                start -= sentences_length[i-1] + 1
                end -= sentences_length[i-1] + 1

            # store the annotation in the corresponding sentence
            the_sentence = sentences_obj[i]
            the_sentence.keyphrases.append(Keyphrase(the_sentence, None, labels_by_id[lid], lid, start, end))
            sentence_by_id[lid] = the_sentence

        for line in goldC.open():
            rel, org, dest = line.split()
            org, dest = int(org), int(dest)

            # find the sentence this relation belongs ti
            the_sentence = sentence_by_id[org]
            assert the_sentence == sentence_by_id[dest]
            # and store it
            the_sentence.relations.append(Relation(the_sentence, org, dest, rel))

        self.sentences.extend(sentences_obj)

    @property
    def feature_size(self):
        return self.sentences[0].tokens[0].features.shape[0]

    def split(self):
        train = TassDataset(self.sentences[:-self.validation_size], max_length=self.max_length)
        dev = TassDataset(self.sentences[-self.validation_size:], max_length=self.max_length)

        if self.__class__ != TassDataset:
            train = self.__class__(train)
            dev = self.__class__(dev)

        return train, dev

    def token_pairs(self, enums=False):
        for sentence in self.sentences:
            for i, k1 in enumerate(sentence.tokens):
                # if k1.label == '':
                #     continue

                for j, k2 in enumerate(sentence.tokens):
                    # if k2.label == '':
                    #     continue

                    if enums:
                        yield i, j, k1, k2
                    else:
                        yield k1, k2

    def task_a(self):
        return TaskADataset(self)

    def task_b(self):
        return TaskBDataset(self)

    def task_c(self):
        return TaskCDataset(self)

    def task_ab(self):
        return TaskABDataset(self)

    def task_bc(self):
        return TaskBCDataset(self)


class TaskADataset(TassDataset):
    def __init__(self, dataset:TassDataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 1

    def by_word(self):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                X.append(token.features)
                y.append(token.keyword_label)

        return np.vstack(X), np.hstack(y)

    def by_sentence(self):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                X.append(sentence.token_features(self.max_length, i))
                y.append(token.keyword_label)

        return np.asarray(X), np.hstack(y)


class TaskBDataset(TassDataset):
    def __init__(self, dataset:TassDataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 1

    def by_word(self):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.label != '':
                    X.append(token.features)
                    y.append(token.binary_label)

        return np.vstack(X), np.hstack(y)

    def by_sentence(self):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                if token.label != '':
                    X.append(sentence.token_features(self.max_length, i))
                    y.append(token.binary_label)

        return np.asarray(X), np.hstack(y)


class TaskCDataset(TassDataset):
    def __init__(self, dataset:TassDataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 6

    def by_word(self):
        X = []
        y = []

        for k1, k2 in self.token_pairs():
            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            feature_vector = np.hstack((k1.features, k2.features))

            X.append(feature_vector)
            y.append(relation_vector)

        return np.vstack(X), np.vstack(y)

    def by_sentence(self):
        X = []
        y = []

        for i, j, k1, k2 in self.token_pairs(enums=True):
            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            feature_vector = k1.sentence.token_features(self.max_length, i, j)

            X.append(feature_vector)
            y.append(relation_vector)

        return np.asarray(X), np.vstack(y)


class TaskABDataset(TassDataset):
    def __init__(self, dataset:TassDataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 3

    def by_word(self):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                X.append(token.features)
                y.append(token.ternary_label)

        return np.vstack(X), np.hstack(y)

    def by_sentence(self):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                X.append(sentence.token_features(self.max_length, i))
                y.append(token.ternary_label)

        return np.asarray(X), np.hstack(y)


class TaskBCDataset(TassDataset):
    def __init__(self, dataset:TassDataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 8

    def by_word(self):
        X = []
        y = []

        for k1, k2 in self.token_pairs():
            labels = np.asarray([k1.binary_label, k2.binary_label])

            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)
            relation_vector = np.hstack((relation_vector, labels))

            feature_vector = np.hstack((k1.features, k2.features))

            X.append(feature_vector)
            y.append(relation_vector)

        return np.vstack(X), np.vstack(y)

    def by_sentence(self):
        X = []
        y = []

        for i, j, k1, k2 in self.token_pairs(enums=True):
            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            feature_vector = k1.sentence.token_features(self.max_length, i, j)

            X.append(feature_vector)
            y.append(relation_vector)

        return np.asarray(X), np.vstack(y)


class Keyphrase:
    def __init__(self, sentence, features, label, id, start, end):
        self.sentence = sentence
        self.features = features
        self.label = label
        self.id = id
        self.start = start
        self.end = end

    def clone(self, sentence):
        return Keyphrase(sentence, self.features, self.label, self.id, self.start, self.end)

    @property
    def text(self) -> str:
        return self.sentence.text[self.start:self.end]

    @property
    def keyword_label(self) -> int:
        return 0 if self.label == '' else 1

    @property
    def binary_label(self) -> int:
        return 0 if self.label == 'Action' else 1

    @property
    def ternary_label(self) -> int:
        return 0 if self.label == '' else 1 if self.label == 'Action' else 2

    def mark_keyword(self, value):
        if hasattr(self, 'is_kw'):
            raise ValueError("Already marked!")

        self.is_kw = value

    def mark_label(self, value):
        if not hasattr(self, 'is_kw'):
            raise ValueError("Must be marked as keyword first")

        if isinstance(value, int):
            value = ['Action', 'Concept'][value]

        self.label = value if self.is_kw else ''

    def mark_ternary(self, value):
        if hasattr(self, 'is_kw'):
            raise ValueError("Already marked!")

        # if isinstance(value, np.ndarray):
        #     value = np.argmax(a)

        if isinstance(value, int):
            value = ['', 'Action', 'Concept'][value]

        self.label = value
        self.is_kw = self.label != ''

    def __repr__(self):
        return "Keyphrase(text=%r, label=%r, id=%r)" % (self.text, self.label, self.id)


class Relation:
    def __init__(self, sentence, origin, destination, label):
        self.sentence = sentence
        self.origin = origin
        self.destination = destination
        self.label = label

    def clone(self, sentence):
        return Relation(sentence, self.origin, self.destination, self.label)

    @property
    def from_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.origin)

    @property
    def to_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.destination)

    def __repr__(self):
        return "Relation(from=%r, to=%r, label=%r)" % (self.from_phrase.text, self.to_phrase.text, self.label)


class Sentence:
    def __init__(self, text:str):
        self.text = text
        self.keyphrases = []
        self.relations = []
        self.tokens = []
        self.predicted_relations = []

    def clone(self):
        s = Sentence(self.text)
        s.keyphrases = [k.clone(s) for k in self.keyphrases]
        s.relations = [r.clone(s) for r in self.relations]
        s.tokens = [k.clone(s) for k in self.tokens]
        return s

    def find_keyphrase(self, id=None, start=None, end=None) -> Keyphrase:
        if id is not None:
            return self._find_keyphrase_by_id(id)
        return self._find_keyphrase_by_spans(start, end)

    def find_relations(self, orig, dest):
        results = []

        for r in self.relations:
            if r.origin == orig and r.destination == dest:
                results.append(r)

        return results

    def find_relation(self, orig, dest, label=None):
        for r in self.relations:
            if r.origin == orig and r.destination == dest:
                if (label and r.label == label) or label is None:
                    return r

        return None

    def _find_keyphrase_by_id(self, id):
        for k in self.keyphrases:
            if k.id == id:
                return k

        return None

    def _find_keyphrase_by_spans(self, start, end):
        for k in self.keyphrases:
            if k.start == start and k.end == end:
                return k

        return None

    def token_features(self, max_length:int, *index:int):
        X = []

        for token in self.tokens:
            X.append(token.features)

        X = np.asarray(X)
        padding = max_length - len(X)

        if padding > 0:
            _, cols = X.shape
            X = np.vstack((X, np.zeros((padding, cols))))

        idxcols = np.zeros((max_length, len(index)))
        for col,row in enumerate(index):
            idxcols[row, col] = 1.0

        return np.hstack((X, idxcols))

    def add_predicted_relations(self, k1, k2, relations):
        if not isinstance(relations, list):
            relations = list(relation_mapper.inverse_transform(relations.reshape(1,-1))[0])

        for relation in relations:
            if relation in ['subject', 'target']:
                if k1.label != 'Action' or k2.label != 'target':
                    continue
            else:
                if k1.label != 'Concept' or k2.label != 'Concept':
                    continue

            self.predicted_relations.append(Relation(self, k1.id, k2.id, relation))

    def invert(self):
        s = Sentence(self.text)
        s.keyphrases = [t for t in self.tokens if t.is_kw]
        s.relations = self.predicted_relations
        return s

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r)" % (self.text, self.keyphrases, self.relations)
