import numpy as np
import pandas as pd
import os
import re
import joblib

import ast

import parsers.obo as obo

from sklearn.ensemble import RandomForestClassifier

ONTOLOGIES = ['biological_process', 'cellular_component', 'molecular_function']
ORGANISMS_ID = ['scer', 'celegans', 'dmel', 'hg', 'mm']
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

def closest_family(graph, node):
    parents = set(graph.successors(node))
    childrens = set(graph.predecessors(node))
    siblings = [set(list(graph.predecessors(node))) for node in parents] + [set(list(graph.successors(node))) for node in childrens]
    siblings = set.union(*siblings)
    closest = set(parents | childrens | siblings)
    return closest

def load_data(go_id, go_ids, ontology_subgraph, annots_train, annots_test, data_train, data_test, data):
    closest_nodes = sorted(list(closest_family(ontology_subgraph, go_id)))
    closest_mask = np.isin(go_ids, closest_nodes)
    sibling_nodes = sorted(list(siblings(ontology_subgraph, go_id)))

    y_train_true = annots_train['go_id'] == go_id
    y_test_true = annots_test['go_id'] == go_id
    y_train_false = annots_train['go_id'].isin(sibling_nodes)
    y_test_false = annots_test['go_id'].isin(sibling_nodes)

    index_train_true = annots_train[y_train_true].index.drop_duplicates()
    index_train_false = annots_train[y_train_false & ~annots_train.index.isin(index_train_true)].index.drop_duplicates()
    index_test_true = annots_test[y_test_true].index.drop_duplicates()
    index_test_false = annots_test[y_test_false & ~annots_test.index.isin(index_test_true)].index.drop_duplicates()

    index_train = index_train_true.append(index_train_false)
    index_test = index_test_true.append(index_test_false)

    X_train = np.array(data_train[data_train.index.isin(index_train)])
    X_test = np.array(data_test[data_test.index.isin(index_test)])

    times_to_repeat = int(X_train.shape[1] / closest_mask.shape[0])
    closest_mask = np.repeat(closest_mask, times_to_repeat)
    X = np.array(data)[:,closest_mask]
    X_train = X_train[:,closest_mask]
    X_test = X_test[:,closest_mask]

    y_train = np.concatenate((np.ones_like(index_train_true, dtype='int'), np.zeros_like(index_train_false, dtype='int')), axis=None)
    y_test = np.concatenate((np.ones_like(index_test_true, dtype='int'), np.zeros_like(index_test_false, dtype='int')), axis=None)
    return X, X_train, y_train, X_test, y_test, index_train, index_test


def prediction(organism_id, ontology, parameters):
    data_path = '../datasets/processed/{}/'.format(organism_id)
    genome = pd.read_csv('../datasets/preprocessed/{}/genome.csv'.format(organism_id), sep='\t', dtype={'seqname':str})
    if organism_id in ['hg', 'mm'] and ontology == 'biological_process':
        genome_train = pd.read_csv('{}/genome_train_8020.csv'.format(data_path, organism_id), sep='\t', dtype={'seqname':str})
        genome_test = pd.read_csv('{}/genome_test_8020.csv'.format(data_path, organism_id), sep='\t', dtype={'seqname':str})
    else:
        genome_train = pd.read_csv('{}/genome_train.csv'.format(data_path, organism_id), sep='\t', dtype={'seqname':str})
        genome_test = pd.read_csv('{}/genome_test.csv'.format(data_path, organism_id), sep='\t', dtype={'seqname':str})
    len_chromosomes = dict(genome.groupby('seqname').size())

    data_path = '{}/{}/'.format(data_path, ontology)
    annots_train = pd.read_csv('{}/annots_train.csv'.format(data_path), sep='\t', dtype={'seqname':str})
    annots_test = pd.read_csv('{}/annots_test.csv'.format(data_path), sep='\t', dtype={'seqname':str})

    genome = genome.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])
    genome_train = genome_train.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])
    genome_test = genome_test.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])
    annots_train = annots_train.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])
    annots_test = annots_test.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])

    index = genome.index
    index_train = genome_train.index
    index_test = genome_test.index

    go_ids = sorted(list(annots_train['go_id'].unique()))
    ontology_subgraph = ontology_graphs[ontology].subgraph(go_ids)

    # for go_id in go_ids:
    #     df = pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t')
    #     print(go_id, np.array(df.drop(['seqname'], axis=1)).sum())

    columns = ['pos', 'seqname', 'lea_5', 'lea_10', 'lea_20', 'lea_50', 'lea_100']
    data = []
    for go_id in go_ids:
        df = pd.read_csv('{}/{}.csv'.format(data_path, go_id, dtype={'seqname':str}), sep='\t')[columns]
        seqnames = df.seqname.unique()
        df = [df]
        for seqname in len_chromosomes:
            if seqname not in seqnames:
                dff = pd.DataFrame(columns=columns)
                dff['pos'] = np.arange(len_chromosomes[seqname])
                dff['seqname'] = str(seqname)
                df.append(dff)
        df = pd.concat(df)
        df = df.sort_values(by=['seqname', 'pos'])
        df = df.reset_index()[columns]
        data.append(df)
    data = pd.concat(data, axis=1)
    data = data.fillna(0)

    duplicated_index = np.isin(data.columns, ['pos', 'seqname'])
    duplicated_index[0] = duplicated_index[1] = False
    data = data.loc[:,~duplicated_index]
    data = data.sort_values(by=['seqname', 'pos']).set_index(['seqname', 'pos'])

    # data = pd.concat([pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t') for go_id in go_ids], axis=1)
    # duplicated_index = np.isin(data.columns, ['pos', 'seqname'])
    # duplicated_index[0] = duplicated_index[1] = False
    # data = data.loc[:,~duplicated_index]
    # data = data.set_index(['pos', 'seqname'])

    # data = data.fillna(0) # no se porque aparecen nan ni donde, asi que se soluciona asi

    # data_train = data[data.index.isin(annots_train.index)]
    # data_test = data[data.index.isin(annots_test.index)]

    data_train = data[data.index.isin(genome_train.index)]
    data_test = data[data.index.isin(genome_test.index)]

    root = find_root(ontology_subgraph)
    datas = data

    results = pd.DataFrame(index=datas.index)
    for node in sorted(ontology_subgraph.nodes):
        if node == root or node not in parameters:
            results[node] = 1
            continue
        X, X_train, y_train, X_test, y_test, index_go_train, index_go_test = load_data(node, go_ids, ontology_subgraph, annots_train, annots_test, data_train, data_test, datas)

        if y_train.mean() < 1.0:
            param = parameters[node]
            clf = RandomForestClassifier(
                max_features=param['max_features'],
                n_estimators=param['n_estimators'],
                max_depth=param['max_depth'],
                random_state=param['random_state'],
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            prior_probs = clf.predict_proba(X)[:,1]
        else:
            prior_probs = np.ones_like(len(X))
        # results[node] = 1.0
        # results[node][index.isin(index_go_test)] = prior_probs
        # results[node][index.isin(index_go_train)] = prior_probs_train
        results[node] = prior_probs

    results.to_csv('complete/complete_model_{}_{}.csv'.format(organism_id, ontology), index=True, sep='\t')

m = re.compile('(GO:\d+)\s({.+})')
for organism_id in ORGANISMS_ID:
    for ontology in ONTOLOGIES:
        print(organism_id, ontology)
        fileObject = open('./parameters/f1_{}_{}.txt'.format(organism_id, ontology), 'r')
        lines = fileObject.readlines()
        parameters = {match.group(1): ast.literal_eval(match.group(2)) for match in map(lambda l: m.search(l), lines) if match}
        prediction(organism_id, ontology, parameters)
