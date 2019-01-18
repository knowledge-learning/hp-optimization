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


class TassDataset:
    def __init__(self, sentences=None, validation_size=0):
        self.sentences = sentences or []
        self.validation_size = 0

        # self.texts = []
        # self.labels = []
        # self.relations = []
        # self.vectors = []
        # self.tokens = []
        # self.relmap = DictVectorizer().fit({r:True} for r in "is-a same-as part-of property-of subject target".split())

    def clone(self):
        return TassDataset([s.clone() for s in self.sentences], self.validation_size)

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

    # def _check_repr(self):
    #     if self.vectors is None or self.tokens is None:
    #         raise ValueError("Preprocesing and representation is not ready yet.")

    #     self.labels_map = self._labels_map()

    #     for v, t, l, s in szip(self.vectors, self.tokens, self.labels_map, self.texts):
    #         assert len(v) == len(t) == len(l)

    #     self.max_length = max(map(lambda n: n.shape[0], self.vectors))

    # def _labels_map(self):
    #     print("(!) Computing labels mapping for dataset")
    #     labels_map = []

    #     for sent, lbls in szip(self.tokens, self.labels):
    #         sent_map = []
    #         for t in sent:
    #             if (t.init, t.end) in lbls:
    #                 lbl = lbls[(t.init, t.end)][1]
    #                 sent_map.append(lbl)
    #             else:
    #                 sent_map.append('')

    #         labels_map.append(sent_map)

    #     return labels_map

    def split(self):
        train = TassDataset(self.sentences[:-self.validation_size])
        dev = TassDataset(self.sentences[-self.validation_size:])
        return train, dev

    # @property
    # def train_vectors(self):
    #     return self.train(self.vectors)

    # @property
    # def dev_vectors(self):
    #     return self.dev(self.vectors)

    # @property
    # def train_tokens(self):
    #     return self.train(self.tokens)

    # @property
    # def dev_tokens(self):
    #     return self.dev(self.tokens)

    # @property
    # def train_labels(self):
    #     return self.train(self.labels)

    # @property
    # def train_relations(self):
    #     return self.train(self.relations)

    # @property
    # def dev_labels(self):
    #     return self.dev(self.labels)

    # @property
    # def dev_relations(self):
    #     return self.dev(self.relations)

    def task_a_by_word(self):
        self._check_repr()

        xtrain = self.train_vectors
        ytrain = [np.asarray([1 if l else 0 for l in sent]) for sent in self.train(self.labels_map)]
        xdev = self.dev_vectors

        return xtrain, ytrain, xdev

    def task_a_by_sentence(self):
        self._check_repr()

        xtrain = []
        ytrain = []

        for sentence, labels in szip(self.train_vectors, self.train(self.labels_map)):
            padding = self.max_length - len(sentence)

            if padding < 0:
                sentence = sentence[:self.max_length,:]
                padding = None
            else:
                _, cols = self.vectors[0].shape
                padding = np.zeros((padding, cols))
                sentence = np.vstack((sentence, padding))

            for i, lb in enumerate(labels):
                col = np.zeros((self.max_length, 1))
                col[i,0] = 1.0
                x = np.hstack((sentence, col))
                y = 0 if lb == '' else 1

                xtrain.append(x)
                ytrain.append(y)

        xdev = []

        for sentence in self.dev_vectors:
            xdev_sent = []
            sentence_size = len(sentence)
            padding = self.max_length - len(sentence)

            if padding < 0:
                sentence = sentence[:self.max_length,:]
                padding = None
            else:
                _, cols = self.vectors[0].shape
                padding = np.zeros((padding, cols))
                sentence = np.vstack((sentence, padding))

            for i in range(sentence_size):
                col = np.zeros((self.max_length, 1))
                col[i,0] = 1.0
                x = np.hstack((sentence, col))

                xdev_sent.append(x)
            xdev.append(np.asarray(xdev_sent))

        return xtrain, ytrain, xdev

    def task_b_by_word(self):
        self._check_repr()

        xtrain = []
        ytrain = []

        ymap = {
            'Action': 0,
            'Concept': 1,
        }

        for sentence, labels in szip(self.train_vectors, self.train(self.labels_map)):
            new_sent = []
            new_lbl = []
            for word, lbl in szip(sentence, labels):
                if lbl:
                    new_sent.append(word)
                    new_lbl.append(ymap[lbl])
            if new_sent:
                new_sent = np.vstack(new_sent)
                new_lbl = np.hstack(new_lbl)

            xtrain.append(new_sent)
            ytrain.append(new_lbl)

        xdev = self.dev_vectors

        return xtrain, ytrain, xdev

    def task_b_by_sentence(self):
        self._check_repr()

        xtrain = []
        ytrain = []

        ymap = {
            'Action': 0,
            'Concept': 1,
        }

        for sentence, labels in szip(self.train_vectors, self.train(self.labels_map)):
            padding = self.max_length - len(sentence)

            if padding < 0:
                sentence = sentence[:self.max_length,:]
                padding = None
            else:
                _, cols = self.vectors[0].shape
                padding = np.zeros((padding, cols))
                sentence = np.vstack((sentence, padding))

            for i, lb in enumerate(labels):
                col = np.zeros((self.max_length, 1))
                col[i,0] = 1.0
                x = np.hstack((sentence, col))
                y = ymap[lb]

                xtrain.append(x)
                ytrain.append(y)

        xdev = []

        for sentence in self.dev_vectors:
            xdev_sent = []
            sentence_size = len(sentence)
            padding = self.max_length - len(sentence)

            if padding < 0:
                sentence = sentence[:self.max_length,:]
                padding = None
            else:
                _, cols = self.vectors[0].shape
                padding = np.zeros((padding, cols))
                sentence = np.vstack((sentence, padding))

            for i in range(sentence_size):
                col = np.zeros((self.max_length, 1))
                col[i,0] = 1.0
                x = np.hstack((sentence, col))

                xdev_sent.append(x)
            xdev.append(np.asarray(xdev_sent))

        return xtrain, ytrain, xdev

    def task_c_by_word(self):
        self._check_repr()

        x = []
        y = []
        mapping = []

        for feats, sent, lbls, rels in szip(self.vectors, self.tokens, self.labels, self.relations):
            rel_pairs = []
            rel_map = []
            sent_mapp = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls:
                        continue

                    pair_map = {}

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    rel_pairs.append(np.hstack((feats[i], feats[j])))

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    rel_map.append(pair_map)
                    sent_mapp.append(((t1.init, t1.end),(t2.init, t2.end)))

            mapping.append(sent_mapp)
            rel_pairs = np.vstack(rel_pairs)
            rel_map = self.relmap.transform(rel_map).toarray()

            x.append(rel_pairs)
            y.append(rel_map)

        xtrain = self.train(x)
        ytrain = self.train(y)
        xdev = self.dev(x)
        mapping = self.dev(mapping)

        return xtrain, ytrain, xdev, mapping

    def task_c_by_sentence(self):
        self._check_repr()

        xtrain = []
        ytrain = []

        for feats, sent, lbls, rels in szip(self.train_vectors, self.train_tokens, self.train_labels, self.train_relations):
            sent_mapp = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls:
                        continue

                    pair_map = {}

                    xmatrix = feats

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    rows, cols = xmatrix.shape
                    padding = self.max_length - rows

                    if padding:
                        xmatrix = np.vstack((xmatrix, np.zeros((padding, cols))))

                    indicators = np.zeros((self.max_length, 2))
                    indicators[i:-2] = 1
                    indicators[j:-1] = 1

                    xmatrix = np.hstack((xmatrix, indicators))

                    xtrain.append(xmatrix)

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    ytrain.append(self.relmap.transform([pair_map]).reshape(1,-1).toarray())

        xdev = []
        mapping = []

        for feats, sent, lbls, rels in szip(self.dev_vectors, self.dev_tokens, self.dev_labels, self.dev_relations):
            sent_mapp = []
            xdev_sent = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls:
                        continue

                    pair_map = {}

                    xmatrix = feats

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    rows, cols = xmatrix.shape
                    padding = self.max_length - rows

                    if padding:
                        xmatrix = np.vstack((xmatrix, np.zeros((padding, cols))))

                    indicators = np.zeros((self.max_length, 2))
                    indicators[i:-2] = 1
                    indicators[j:-1] = 1

                    xmatrix = np.hstack((xmatrix, indicators))

                    xdev_sent.append(xmatrix)
                    sent_mapp.append(((t1.init, t1.end),(t2.init, t2.end)))

            xdev_sent = np.asarray(xdev_sent)
            xdev.append(xdev_sent)
            mapping.append(sent_mapp)

        return xtrain, ytrain, xdev, mapping

    def task_ab_by_word(self):
        self._check_repr()

        xtrain = self.train_vectors
        ytrain = [np.asarray(sent) for sent in self.train(self.labels_map)]
        xdev = self.dev_vectors

        return xtrain, ytrain, xdev

    def task_bc_by_word(self):
        self._check_repr()

        x = []
        y = []
        mapping = []

        label_maps = {
            'Action': [0,1],
            'Concept': [1,0],
            None: [0,0]
        }

        for feats, sent, lbls, rels in szip(self.vectors, self.tokens, self.labels, self.relations):
            rel_pairs = []
            rel_map = []
            rel_clss = []
            sent_mapp = []
            for i,t1 in enumerate(sent):
                if (t1.init, t1.end) not in lbls:
                    continue

                for j,t2 in enumerate(sent):
                    if (t2.init, t2.end) not in lbls:
                        continue

                    pair_map = {}

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    rel_pairs.append(np.hstack((feats[i], feats[j])))

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    rel_map.append(pair_map)
                    sent_mapp.append(((t1.init, t1.end),(t2.init, t2.end)))

                    # task B specific representation
                    lbl1e = label_maps[lbl1]
                    lbl2e = label_maps[lbl2]
                    lble = np.asarray(lbl1e + lbl2e)
                    rel_clss.append(lble)

            mapping.append(sent_mapp)
            rel_pairs = np.vstack(rel_pairs)
            rel_map = self.relmap.transform(rel_map).toarray()

            # task B specific representation
            rel_clss = np.vstack(rel_clss)
            rel_map = np.hstack((rel_map, rel_clss))

            x.append(rel_pairs)
            y.append(rel_map)

        xtrain = self.train(x)
        ytrain = self.train(y)
        xdev = self.dev(x)
        mapping = self.dev(mapping)

        return xtrain, ytrain, xdev, mapping

    def task_abc_by_word(self):
        self._check_repr()

        x = []
        y = []
        mapping = []

        label_maps = {
            'Action': [0,1],
            'Concept': [1,0],
            None: [0,0]
        }

        for feats, sent, lbls, rels in szip(self.vectors, self.tokens, self.labels, self.relations):
            rel_pairs = []
            rel_map = []
            rel_clss = []
            sent_mapp = []
            for i,t1 in enumerate(sent):
                for j,t2 in enumerate(sent):
                    pair_map = {}

                    # id1, id2 son los id de 2 tokens
                    id1, lbl1 = lbls.get((t1.init, t1.end), (None, None))
                    id2, lbl2 = lbls.get((t2.init, t2.end), (None, None))

                    rel_pairs.append(np.hstack((feats[i], feats[j])))

                    # calculamos todas las relaciones entre id1 y id2
                    for rel, org, dest in rels:
                        if org == id1 and dest == id2:
                            pair_map[rel] = True

                    rel_map.append(pair_map)
                    sent_mapp.append(((t1.init, t1.end),(t2.init, t2.end)))

                    # task B specific representation
                    lbl1e = label_maps[lbl1]
                    lbl2e = label_maps[lbl2]
                    lble = np.asarray(lbl1e + lbl2e)
                    rel_clss.append(lble)

            mapping.append(sent_mapp)
            rel_pairs = np.vstack(rel_pairs)
            rel_map = self.relmap.transform(rel_map).toarray()

            # task B specific representation
            rel_clss = np.vstack(rel_clss)
            rel_map = np.hstack((rel_map, rel_clss))

            x.append(rel_pairs)
            y.append(rel_map)

        xtrain = self.train(x)
        ytrain = self.train(y)
        xdev = self.dev(x)
        mapping = self.dev(mapping)

        return xtrain, ytrain, xdev, mapping


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

    def __repr__(self):
        return "Keyphrase(text=%r, label=%r)" % (self.text, self.label)


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
            if r.orig == orig and r.destination == dest:
                results.append(r)

        return results

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

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r, tokens=%r)" % (self.text, self.keyphrases, self.relations, self.tokens)
