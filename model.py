import numpy as np
import pandas as pd
import os
import re

import parsers.obo as obo

from sklearn.svm import SVC

ONTOLOGIES = ['biological_process', 'cellular_component', 'molecular_function']
ontology_path = '../datasets/raw/obo/go-basic.obo'
gos, ontology_gos, go_alt_ids, ontology_graphs = obo.parse_obo(ontology_path)


def find_root(graph, node=None):
    if node == None:
        node = list(graph.nodes())[0]
    parents = list(graph.successors(node))
    if len(parents) == 0: return node
    else: return find_root(graph, parents[0])


def siblings(graph, node):
    parents = list(graph.successors(node))
    if len(parents) == 0: return {} # root node
    else:
        siblings_nodes = [set(list(graph.predecessors(node))) for node in parents]
        siblings_nodes = set.union(*siblings_nodes)
        if len(siblings_nodes) > 1:
            return siblings_nodes - {node}
        else:
            return set.union(*[siblings(graph, node) for node in parents]) - set(parents) # without siblings


def load_data(go_id, ontology_subgraph, annots_train, annots_test, data_train, data_test):
    sibling_nodes = list(siblings(ontology_subgraph, go_id))
    y_train_true = annots_train['go_id'] == go_id
    y_test_true = annots_test['go_id'] == go_id

    y_train_false = False
    y_test_false = False
    for sibling_node in sibling_nodes:
        y_train_false = y_train_false | (annots_train['go_id'] == sibling_node)
        y_test_false = y_test_false | (annots_test['go_id'] == sibling_node)

    y_train_false = y_train_false & (~(y_train_false & y_train_true))
    y_test_false = y_test_false & (~(y_test_false & y_test_true))

    index_train_true = annots_train[y_train_true].index.drop_duplicates()
    index_train_false = annots_train[y_train_false].index.drop_duplicates()
    index_test_true = annots_test[y_test_true].index.drop_duplicates()
    index_test_false = annots_test[y_test_false].index.drop_duplicates()

    X_train = np.concatenate((data_train[data_train.index.isin(index_train_true)],
                                data_train[data_train.index.isin(index_train_false)]))
    X_test = np.concatenate((data_test[data_test.index.isin(index_test_true)],
                                data_test[data_test.index.isin(index_test_false)]))

    y_train = np.concatenate((np.ones_like(index_train_true, dtype='int'), np.zeros_like(index_train_false, dtype='int')), axis=None)
    y_test = np.concatenate((np.ones_like(index_test_true, dtype='int'), np.zeros_like(index_test_false, dtype='int')), axis=None)
    return X_train, y_train, X_test, y_test


def model(organism_id, ontology):
    data_path = '../datasets/processed/{}/'.format(organism_id)
    genome = pd.read_csv('../datasets/preprocessed/{}/genome.csv'.format(organism_id), sep='\t')
    genome_train = pd.read_csv('{}/genome_train.csv'.format(data_path, organism_id), sep='\t')
    genome_test = pd.read_csv('{}/genome_test.csv'.format(data_path, organism_id), sep='\t')
    len_chromosomes = dict(genome.groupby('seqname').size())

    data_path = '{}/{}/'.format(data_path, ontology)
    annots_train = pd.read_csv('{}/annots_train.csv'.format(data_path), sep='\t')
    annots_test = pd.read_csv('{}/annots_test.csv'.format(data_path), sep='\t')

    genome_train = genome_train.set_index(['pos', 'seqname'])
    genome_test = genome_test.set_index(['pos', 'seqname'])
    annots_train = annots_train.set_index(['pos', 'seqname'])
    annots_test = annots_test.set_index(['pos', 'seqname'])

    go_ids = list(annots_train['go_id'].unique())
    ontology_subgraph = ontology_graphs[ontology].subgraph(go_ids)

    print(go_ids[63:67])
    # for go_id in go_ids:
    #     df = pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t')
    #     print(np.array(df.isna()).sum())
    #     if np.array(df.isna()).sum() > 0:
    #         print(go_id)

    data = pd.concat([pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t') for go_id in go_ids], axis=1)
    duplicated_columns = np.isin(data.columns, ['pos', 'seqname'])
    duplicated_columns[0] = duplicated_columns[1] = False
    data = data.loc[:,~duplicated_columns]
    data = data.set_index(['pos', 'seqname'])

    print(list(zip(list(data.columns), np.array(data.isna()).sum(axis=0))))

    data_train = data[data.index.isin(annots_train.index)]
    data_test = data[data.index.isin(annots_test.index)]

    root = find_root(ontology_subgraph)
    for node in ontology_subgraph:
        if node == root:
            continue
        X_train, y_train, X_test, y_test = load_data(node, ontology_subgraph, annots_train, annots_test, data_train, data_test)

        # MODEL, implementar un GridSearch
        print(node, y_test.mean(), y_train.mean())
        clf = SVC(probability=True, gamma='auto')
        print(np.isfinite(X_train).mean(), np.isfinite(X_test).mean(), np.isfinite(y_train).mean(), np.isfinite(y_test).mean())
        clf.fit(X_train, y_train)
        print(clf.predict_proba(X_test))
        # print(clf.score(X_test, y_test))
        print('=============================')


model('celegans', 'cellular_component')
