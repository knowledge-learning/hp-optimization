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
    def __init__(self):
        self.texts = []
        self.labels = []
        self.relations = []
        self.validation_size = 0
        self.vectors = []
        self.tokens = []
        self.relmap = DictVectorizer().fit({r:True} for r in "is-a same-as part-of property-of subject target".split())

    def load(self, finput:Path):
        goldA = finput.parent / ('output_A_' + finput.name[6:])
        goldB = finput.parent / ('output_B_' + finput.name[6:])
        goldC = finput.parent / ('output_C_' + finput.name[6:])

        text = finput.open().read()
        sentences = [s for s in text.split('\n') if s]

        self.texts.extend(sentences)
        self._parse_ann(sentences, goldA, goldB, goldC)

        return len(sentences)

    def _parse_ann(self, sentences, goldA, goldB, goldC):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        labelsA_doc = [{} for _ in sentences]
        relations_doc = [[] for _ in sentences]
        labelsB = {}
        sent_map = {}

        for line in goldB.open():
            lid, lbl = line.split()
            labelsB[int(lid)] = lbl

        for line in goldA.open():
            lid, start, end = (int(i) for i in line.split())

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, start)
            if i > 0:
                start -= sentences_length[i-1] + 1
                end -= sentences_length[i-1] + 1
            labelsA_doc[i][(start,end)] = (lid, labelsB[lid])
            sent_map[lid] = i

        for line in goldC.open():
            rel, org, dest = line.split()
            org, dest = int(org), int(dest)
            sent = sent_map[org]
            assert sent == sent_map[dest]
            relations_doc[sent].append((rel, org, dest))

        self.labels.extend(labelsA_doc)
        self.relations.extend(relations_doc)

    def _check_repr(self):
        if self.vectors is None or self.tokens is None:
            raise ValueError("Preprocesing and representation is not ready yet.")

    @property
    @cached
    def labels_map(self):
        print("(!) Computing labels mapping for dataset")
        labels_map = []

        for sent, lbls in zip(self.tokens, self.labels):
            sent_map = []
            for t in sent:
                if (t.init, t.end) in lbls:
                    lbl = lbls[(t.init, t.end)][1]
                    sent_map.append(lbl)
                else:
                    sent_map.append('')

            labels_map.append(sent_map)

        return labels_map

    def train(self, x):
        return x[:-self.validation_size]

    def dev(self, x):
        return x[-self.validation_size:]

    @property
    def train_vectors(self):
        self._check_repr()
        return self.train(self.vectors)

    @property
    def dev_vectors(self):
        self._check_repr()
        return self.dev(self.vectors)

    @property
    def train_tokens(self):
        self._check_repr()
        return self.train(self.tokens)

    @property
    def dev_tokens(self):
        self._check_repr()
        return self.dev(self.tokens)

    @property
    def train_labels(self):
        return self.train(self.labels)

    @property
    def train_relations(self):
        return self.train(self.relations)

    @property
    def dev_labels(self):
        return self.dev(self.labels)

    @property
    def dev_relations(self):
        return self.dev(self.relations)

    def task_a_by_word(self):
        self._check_repr()

        xtrain = self.train_vectors
        ytrain = [np.asarray([1 if l else 0 for l in sent]) for sent in self.train(self.labels_map)]
        xdev = self.dev_vectors

        return xtrain, ytrain, xdev

    def task_b_by_word(self):
        self._check_repr()

        xtrain = []
        ytrain = []

        for sentence, labels in szip(self.train_vectors, self.train(self.labels_map)):
            new_sent = []
            new_lbl = []
            for word, lbl in szip(sentence, labels):
                if lbl:
                    new_sent.append(word)
                    new_lbl.append(lbl)
            if new_sent:
                new_sent = np.vstack(new_sent)
                new_lbl = np.hstack(new_lbl)

            xtrain.append(new_sent)
            ytrain.append(new_lbl)

        xdev = self.dev_vectors

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
