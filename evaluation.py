import numpy as np
import pandas as pd
import os
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
# import matplotlib.pyplot as plt

from parsers.obo import parse_obo

from operator import itemgetter
import itertools

def find_root(graph, node=None):
    # Find the root of the graph
    if node is None:
        node = list(graph.nodes())[0]
    parents = list(graph.successors(node))
    if len(parents) == 0: return node
    else: return find_root(graph, parents[0])


def show_graph(graph, pos=None, with_labels=True, node_color='blue'):
    if pos is None:
        pos = {}
        root = find_root(graph)
        bfs = [(root, 0.0, 0.0, 2*np.pi)]
        pos[root] = np.array([0, 0])
        analyzed_nodes = [root]
        while bfs != []:
            aux = []
            for (node, r, sa, fa) in bfs:
                childrens = set(graph.predecessors(node))
                childrens = sorted(childrens - set(analyzed_nodes))
                n = float(len(childrens))
                new_bfs = [(children, r + 1.0, sa + i*(fa - sa)/n, sa + (i + 1)*(fa - sa)/n) for i, children in enumerate(childrens)]
                for (new_node, new_r, s, f) in new_bfs:
                    pos[new_node] = new_r * np.array([np.cos(s), np.sin(s)])
                analyzed_nodes += childrens
                aux += new_bfs
            bfs = aux

    nx.draw(graph, pos, with_labels=with_labels, node_color=node_color)
    plt.show()

    return pos


def check_sanity(graph, probs):
    '''
    Check if probs is coherent with graph's hierarchy
        graph: graph of the ontology
        probs: dictionary of GO terms and their probabilities
    '''
    ans = True
    for node in graph:
        parents = list(graph.successors(node))
        ans = ans and all([probs[node] <= probs[parent] for parent in parents])
        if not ans:
            return ans
    return ans


def pre_probs_parallelize(node, graph, prior_probs):
    parents = graph.successors(node)
    if len(list(parents)) == 0:
        raise ValueError('All this nodes have parents')
    parents_prior_probs = np.array([prior_probs[parent] for parent in parents])
    P_par_1 = prior_probs[node] * np.prod(parents_prior_probs)
    P_par_0 = 1 - P_par_1
    P_child_0 = (1 - prior_probs[node]) * np.prod(1 - parents_prior_probs)
    P_child_1 = 1 - P_child_0
    if P_par_0 > P_child_1:
        return P_par_1
    else:
        return P_child_1


def post_probs_parallelize(node, graph, pre_probs_node, post_probs):
    parents = list(graph.successors(node))
    min_posterior_probs_parents = np.amin([post_probs[parent] for parent in parents])
    return np.amin([pre_probs_node, min_posterior_probs_parents])


def posterior_correction(graph, prior_probs, root, graph_nodes):
    '''
    Given the graph of the ontology and the prior probabilities return the probabilities after hierarchical corrections.
    '''
    # preliminary_probs
    # pre_probs = [pre_probs_parallelize(node, graph, prior_probs) for node in graph_nodes]
    # pre_probs = dict(zip(graph_nodes, pre_probs))
    # pre_probs[root] = prior_probs[root]

    pre_probs = prior_probs
    pre_probs[root] = 1.0

    # # posterior_correction
    post_probs = {}
    post_probs[root] = pre_probs[root]
    node = root
    childrens = list(graph.predecessors(node))
    visited_nodes = set({root})
    while len(childrens) > 0:
        post_probs_aux = {node: post_probs_parallelize(node, graph, pre_probs[node], post_probs) for node in childrens}
        # post_probs = {**post_probs, **post_probs_aux}
        for p in post_probs_aux:
            # if p in post_probs:
            #     raise Exception('p in post_probs', p, post_probs)
            post_probs[p] = post_probs_aux[p]
        visited_nodes = visited_nodes.union(set(childrens))
        childrens = [set(graph.predecessors(children)) for children in childrens]
        childrens = list(set.union(*childrens))
        childrens = [children for children in childrens if set(graph.successors(children)).issubset(visited_nodes)]

    return post_probs


def evaluate(prediction, graph, thresholds):
    data_post = []
    GO_terms = list(prediction.columns)
    preds = {th:{} for th in thresholds}
    root = find_root(graph)
    graph_nodes = list(graph.nodes)
    graph_nodes.remove(root)
    for index, row in tqdm(list(prediction.iterrows())):
        probs = list(row)
        prior_probs = dict(zip(GO_terms, probs))
        post_probs = posterior_correction(graph, prior_probs, root, graph_nodes)
        _, probs = zip(*sorted(post_probs.items(), key=itemgetter(0)))
        data_post.append(probs)

        for th in thresholds:
            preds[th][index] = [node for node in GO_terms if post_probs[node] >= float(th)]

    data_post = np.array(data_post)

    post_results = pd.DataFrame(data=data_post, columns=GO_terms, index=prediction.index)

    return post_results, preds


def ancestors(graph, y_pred):
    # y_pred are the GO terms predicted for a single gene
    # return the set of ancestors of GO terms in y_pred in the ontology
    ans = set(y_pred)
    aux = set(y_pred)
    while len(aux) > 0:
        for node in aux:
            ans = ans.union(set(graph.successors(node)))
            aux = aux.union(set(graph.successors(node)))
            aux = aux - {node}
    return ans


def hmetrics(graph, Y_pred, Y_true):
    root = find_root(graph)
    res = {th:[] for th in Y_pred}
    for th in Y_pred:
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        # hPrec = 0
        # hRec = 0
        # hF1 = 0
        for gene, y_pred in Y_pred[th].items():
            y_pred.append(root) # ensure that root belog to y_pred
            # y_true = Y_true.get(gene, [root]) # ensure that root belog to y_true
            y_true = Y_true.get(gene, [])
            P = ancestors(graph, y_pred) # set(P) should be equal to set(y_pred)
            # if set(P) != set(y_pred):
            #     print('NOT', gene)
            T = ancestors(graph, y_true)
            P_inter_T = len(P.intersection(T))
            numerator += P_inter_T
            denominator2 += len(T)
            denominator1 += len(P)
            # hPrec += P_inter_T / len(P)
            # hRec += P_inter_T / len(T)
            # hF1 += 2 * P_inter_T / (len(P) + len(T))
        hprec = numerator / denominator1
        hrec = numerator / denominator2
        hf1 = (2 * hprec * hrec) / (hprec + hrec)
        # hPrec = hPrec / len(Y_pred)
        # hRec = hRec / len(Y_pred)
        # hF1 = hF1 / len(Y_pred)
        # res[th] = [hprec, hrec, hf1, hPrec, hRec, hF1]
        res[th] = [hprec, hrec, hf1]

    return res


def saveEvals(PATH, organism_id, ontology, ontology_graphs):
    results = pd.read_csv('./{}/{}_model_{}_{}.csv'.format(PATH, PATH, organism_id, ontology), sep='\t', dtype={'seqname':str}).sort_values(['seqname', 'pos']).set_index(['seqname', 'pos'])
    # results = pd.read_csv('results_model_celegans_cellular_component.csv', sep='\t', dtype={'seqname':str}).sort_values(['seqname', 'pos']).set_index(['seqname', 'pos'])
    # random_results = pd.read_csv('random_results_model_{}_{}.csv'.format(organism_id, ontology), sep='\t', dtype={'seqname':str}).set_index(['seqname', 'pos'])
    results = results.reindex(sorted(results.columns), axis=1)

    go_ids = results.columns.tolist()
    ontology_subgraph = ontology_graphs[ontology].subgraph(go_ids)

    thresholds = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95']
    post_results, preds = evaluate(results, ontology_subgraph, thresholds)
    if not os.path.exists('./{}'.format(PATH)):
        os.mkdir('./{}'.format(PATH))
    if not os.path.exists('./{}/post'.format(PATH)):
        os.mkdir('./{}/post'.format(PATH))

    post_results.to_csv('./{}/post/post_results_{}_{}.csv'.format(PATH, organism_id, ontology), sep='\t', index=True)

    # random_results = np.random.uniform(size=results.shape)*(results != 0)
    # random_results = pd.DataFrame(random_results, columns=results.columns, index=results.index)

    # flatten = np.array(results).flatten()
    # ind = np.argwhere(flatten != 0)
    # ind2 = np.random.permutation(ind)
    # flatten.flat[ind] = flatten[ind2]
    # random_permutation = flatten.reshape(-1, results.shape[1])
    # random_permutation = pd.DataFrame(random_permutation, columns=results.columns, index=results.index)

    random_results = np.random.uniform(size=results.shape)
    random_results = pd.DataFrame(random_results, columns=results.columns, index=results.index)

    def shuffle(X):
        [np.random.shuffle(x) for x in X]

    shuffledInRows = np.copy(results)
    shuffle(shuffledInRows)

    shuffledInColumns = np.copy(results)
    shuffle(shuffledInColumns.T)

    # random_permutation = pd.DataFrame(array, columns=results.columns, index=results.index)
    random_rows = pd.DataFrame(shuffledInRows, columns=results.columns, index=results.index)
    random_columns = pd.DataFrame(shuffledInColumns, columns=results.columns, index=results.index)

    random_post_results, random_preds = evaluate(random_results, ontology_subgraph, thresholds)
    # _, permutation_preds = evaluate(random_permutation, ontology_subgraph, thresholds)
    _, permutation_rows = evaluate(random_rows, ontology_subgraph, thresholds)
    _, permutation_columns = evaluate(random_columns, ontology_subgraph, thresholds)

    # random_post_results.to_csv('random_post_results.csv', sep='\t', index=True)

    annots_test = pd.read_csv('../datasets/processed/{}/{}/annots_test.csv'.format(organism_id, ontology), sep='\t', dtype={'seqname':str})
    true_annots = {(chromosome, pos):list(set(df['go_id'].values)) for (chromosome, pos), df in annots_test.groupby(['seqname', 'pos'])}

    metrics_results = hmetrics(ontology_subgraph, preds, true_annots)
    metrics_random = hmetrics(ontology_subgraph, random_preds, true_annots)
    # metrics_permutation = hmetrics(ontology_subgraph, permutation_preds, true_annots)
    metrics_rows = hmetrics(ontology_subgraph, permutation_rows, true_annots)
    metrics_columns = hmetrics(ontology_subgraph, permutation_columns, true_annots)

    for shuffle in ['row', 'column']:
        if shuffle == 'row':
            metrics_permutation = metrics_rows
        elif shuffle == 'column':
            metrics_permutation = metrics_columns

        evaluation = []
        for th in thresholds:
            metrics = list(itertools.chain.from_iterable([metrics_results[th], metrics_random[th], metrics_permutation[th]]))        
            evaluation.append(metrics)

        columns = [
            'prec', 'recall', 'f1',
            'rand_prec', 'rand_recall', 'rand_f1',
            'perm_prec', 'perm_recall', 'perm_f1',
            # 'precm', 'recallm', 'f1m', 'precM', 'recallM', 'f1M',
            # 'rand_precm', 'rand_recallm', 'rand_f1m', 'rand_precM', 'rand_recallM', 'rand_f1M',
            # 'perm_precm', 'perm_recallm', 'perm_f1m' 'perm_precM', 'perm_recallM', 'perm_f1M',
        ]
        evaluation = np.array(evaluation)
        evaluation = pd.DataFrame(evaluation, columns=columns)
        evaluation['threshold'] = thresholds

        columns = [
            'threshold',
            'prec', 'rand_prec', 'perm_prec',
            'recall', 'rand_recall', 'perm_recall',
            'f1', 'rand_f1', 'perm_f1',
            # 'precm', 'rand_precm', 'perm_precm', 'precM', 'rand_precM', 'perm_precM',
            # 'recallm', 'rand_recallm', 'perm_recallm', 'recallM', 'rand_recallM', 'perm_recallM',
            # 'f1m', 'rand_f1m', 'perm_f1m', 'f1M', 'rand_f1M', 'perm_f1M',
        ]

        evaluation = evaluation[columns]

        if not os.path.exists('./{}'.format(PATH)):
            os.mkdir('./{}'.format(PATH))
        if not os.path.exists('./{}/{}_metrics'.format(PATH, shuffle)):
            os.mkdir('./{}/{}_metrics'.format(PATH, shuffle))

        evaluation.to_csv('./{}/{}_metrics/metrics_{}_{}.csv'.format(PATH, shuffle, organism_id, ontology), sep='\t', index=False)


if __name__ == '__main__':
    ontology_path = '../datasets/raw/obo/go-basic.obo'
    # ontology_path = '/home/dsilvera/Drive/iibce/workspace/datasets/go-basic.obo'
    gos, ontology_gos, go_alt_ids, ontology_graphs = parse_obo(ontology_path)

    PATHS = ['results', 'complete']
    ORGANISMS_ID = ['scer', 'celegans', 'dmel', 'hg', 'mm']
    ONTOLOGIES = ['cellular_component', 'molecular_function', 'biological_process']

    PATHS = ['results']
    PATHS = ['results', 'test']

    # saveEvals('test', 'celegans', 'cellular_component', ontology_graphs)

    Parallel(n_jobs=-1, verbose=10)(delayed(saveEvals)(p[0], p[1], p[2], ontology_graphs) for p in itertools.product(PATHS, ORGANISMS_ID, ONTOLOGIES))
