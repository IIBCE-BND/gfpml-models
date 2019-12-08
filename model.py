#!/usr/bin/python3

import click
import numpy as np
import pandas as pd
import os
import re
import sys
import time
import wandb

import parsers.obo as obo

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

from evaluation import evaluate, Hprecision_micro, Hrecall_micro


ONTOLOGIES = ['biological_process', 'cellular_component', 'molecular_function']
ontology_path = '../datasets/raw/obo/go-basic.obo'
gos, ontology_gos, go_alt_ids, ontology_graphs = obo.parse_obo(ontology_path)

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())


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


def load_data(go_id, go_ids, ontology_subgraph, annots_train, annots_test, data_train, data_test):
    closest_nodes = list(closest_family(ontology_subgraph, go_id))
    closest_mask = np.isin(go_ids, closest_nodes)
    sibling_nodes = list(siblings(ontology_subgraph, go_id))

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
    X_train = X_train[:,closest_mask]
    X_test = X_test[:,closest_mask]

    y_train = np.concatenate((np.ones_like(index_train_true, dtype='int'), np.zeros_like(index_train_false, dtype='int')), axis=None)
    y_test = np.concatenate((np.ones_like(index_test_true, dtype='int'), np.zeros_like(index_test_false, dtype='int')), axis=None)
    return X_train, y_train, X_test, y_test, index_train, index_test


@click.command()
@click.argument('organism_id')
@click.argument('ontology')
@click.option('--wandb-project', envvar='WANDB_PROJECT', default='gfpml')
def model(organism_id, ontology, wandb_project):
    run_name = '{}_{}_{}'.format(time.strftime('%Y%m%d%H%M%S'), organism_id, ontology)
    click.echo('Starting run "{}"'.format(run_name))

    wandb.init(
        anonymous='allow',
        project=wandb_project,
        name=run_name
    )

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

    index_train = genome_train.index
    index_test = genome_test.index

    go_ids = list(annots_train['go_id'].unique())
    ontology_subgraph = ontology_graphs[ontology].subgraph(go_ids)

    # for go_id in go_ids:
    #     df = pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t')
    #     print(go_id, np.array(df.drop(['seqname'], axis=1)).sum())

    data = pd.concat([pd.read_csv('{}/{}.csv'.format(data_path, go_id), sep='\t') for go_id in go_ids], axis=1)
    duplicated_index = np.isin(data.columns, ['pos', 'seqname'])
    duplicated_index[0] = duplicated_index[1] = False
    data = data.loc[:,~duplicated_index]
    data = data.set_index(['pos', 'seqname'])

    data = data.fillna(0) # no se porque aparecen nan ni donde, asi que se soluciona asi

    # mask = np.array(data.isna()).sum(axis=0) > 0
    # print(mask, np.repeat(go_ids, 3)[mask])

    # print(np.array(data.isna()).shape)

    # print(len(np.repeat(go_ids, 1)), len(data.columns), np.array(data.isna()).sum())
    # print(list(zip(list(np.repeat(go_ids, 3)), list(np.array(data.isna()).sum(axis=0)))))
    # print(list(zip(list(data.columns), np.array(data.isna()).sum(axis=0))))

    data_train = data[data.index.isin(annots_train.index)]
    data_test = data[data.index.isin(annots_test.index)]

    root = find_root(ontology_subgraph)
    results = pd.DataFrame(index=index_test)

    wandb.config.organism_id = organism_id
    wandb.config.ontology = ontology
    wandb.config.go_ids_len = len(go_ids)

    metrics = {
        'acc': {},
        'precision': {},
        'recall': {},
        'f1': {}
    }

    metric_cols = metrics.keys()
    metric_table = wandb.Table(
        columns=[
            'node', 'params', 'x_train', 'x_test', 'y_train_mean', 'y_test_mean'
        ] + list(metric_cols)
    )

    for node in ontology_subgraph:
        if node == root:
            results[node] = 1
            continue

        # if len(metrics['acc']) > 10:
        #     break

        click.echo('Training model for node {}'.format(node))
        X_train, y_train, X_test, y_test, index_go_train, index_go_test = load_data(
            node, go_ids, ontology_subgraph, annots_train, annots_test, data_train, data_test
        )

        # MODEL
        # print(node, y_test.mean(), y_train.mean(), X_train.shape)

        # parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}
        parameters = {'kernel': ['rbf'], 'gamma': [1e-4], 'C': [1]}

        # parameters = {'kernel': ['linear'], 'gamma': [1e-4], 'C': [10]}

        try:
            clf = GridSearchCV(
                SVC(probability=True), 
                parameters,
                cv=5, 
                scoring='neg_log_loss', 
                n_jobs=-1, 
                verbose=100
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            pred_prob = clf.predict_proba(X_test)[:,1]

            results[node] = 0.0
            results[node][index_test.isin(index_go_test)] = pred_prob

            metrics['acc'][node] = accuracy_score(y_test, pred)
            metrics['recall'][node] = recall_score(y_test, pred)
            metrics['precision'][node] = precision_score(y_test, pred)
            metrics['f1'][node] = f1_score(y_test, pred)

            metric_table.add_data(
                node, clf.best_params_, X_train.shape, X_test.shape, y_train.mean(), y_test.mean(),
                *[metrics[m][node] for m in metric_cols]
            )

        except Exception as ex:
            results[node] = 1
            click.echo('Failed training model for node {} with error: {}'.format(node, ex))

    run_dir = os.path.join('runs', run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    results_path = os.path.join(run_dir, 'results_model_{}_{}.csv'.format(organism_id, ontology)) 
    results.to_csv( results_path,index=True, sep='\t')

    wandb.save(results_path)

    # Report metrics.
    wandb.log({'metrics': metric_table})

    # Evaluate accumulated predictions.
    click.echo('Evaluation')
    preds = evaluate(results, threshold=0.3, ontology_subgraph=ontology_subgraph)

    true_annots = {
        (pos, chromosome):list(set(df['go_id'].values))
        for (pos, chromosome), df in annots_test.groupby(['pos', 'seqname'])
    }

    test_h_precision_micro = Hprecision_micro(ontology_subgraph, preds, true_annots)
    click.echo('Hprecision_micro', test_h_precision_micro)
    wandb.run.summary['test_h_precision_micro'] = test_h_precision_micro

    test_h_recall_micro = Hrecall_micro(ontology_subgraph, preds, true_annots)
    click.echo('Hrecall_micro', test_h_recall_micro)
    wandb.run.summary['test_h_recall_micro'] = test_h_recall_micro


if __name__ == "__main__":
    model()
