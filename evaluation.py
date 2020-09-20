import numpy as np
import pandas as pd
import os
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from parsers.obo import parse_obo

from operator import itemgetter
import itertools

def find_root(graph, node=None):
    if node == None:
        node = list(graph.nodes())[0]
    parents = list(graph.successors(node))
    if len(parents) == 0: return node
    else: return find_root(graph, parents[0])


def show_graph(graph, pos=None, with_labels=True, node_color='blue'):
    if pos == None:
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
    root = find_root(graph)
    ans = True
    for node in graph:
        if node == root:
            continue
        parents = list(graph.successors(node))
        ans = ans and all([probs[node] <= probs[parent] for parent in parents])
    return ans


def posterior_correction(graph, prior_probs):
    # preliminary_probs
    root = find_root(graph)
    pre_probs = {}
    for node in graph:
        if node == root:
            pre_probs[node] = prior_probs[node]
        else:
            parents = graph.successors(node)
            if len(list(parents)) == 0:
                raise ValueError('All this nodes have parents')
            P_par_1 = prior_probs[node] * np.prod([prior_probs[parent] for parent in parents])
            P_par_0 = 1 - P_par_1
            P_child_0 = (1 - prior_probs[node]) * np.prod([1 - prior_probs[parent] for parent in parents])
            P_child_1 = 1 - P_child_0
            if P_par_0 > P_child_1: pre_probs[node] = P_par_1
            else: pre_probs[node] = P_child_1

    # posterior_correction
    post_probs = {}
    def posterior_correction_aux(graph, visited_nodes=set({}), node=root):
        if len(visited_nodes) < len(graph):
            if node == root:
                post_probs[node] = pre_probs[node]
            else:
                parents = list(graph.successors(node))
                min_posterior_probs_parents = np.amin([post_probs[parent] for parent in parents])
                post_probs[node] = np.amin([pre_probs[node], min_posterior_probs_parents])
            visited_nodes.add(node)
            childrens = list(graph.predecessors(node))
            for children in childrens:
                parents_children = set(graph.successors(children))
                if parents_children.issubset(visited_nodes):
                    posterior_correction_aux(graph, visited_nodes, node=children)
    
    posterior_correction_aux(graph)
    
    return post_probs


def pre_probs_parallelize(node, graph, prior_probs):
    parents = graph.successors(node)
    # No se porque, pero si sacas este if posterior_correction y posterior_correction_2 dejan de dar iguales
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


def posterior_correction_2(graph, prior_probs, root, graph_nodes):
    # preliminary_probs
    pre_probs = [pre_probs_parallelize(node, graph, prior_probs) for node in graph_nodes]
    pre_probs = dict(zip(graph_nodes, pre_probs))
    pre_probs[root] = prior_probs[root]

    # # posterior_correction
    post_probs = {}
    post_probs[root] = pre_probs[root]
    node = root
    childrens = list(graph.predecessors(node))
    visited_nodes = set({root})
    while len(childrens) > 0:
        post_probs_aux = [post_probs_parallelize(node, graph, pre_probs[node], post_probs) for node in childrens]
        post_probs_aux = dict(zip(childrens, post_probs_aux))

        post_probs = {**post_probs, **post_probs_aux}
        visited_nodes = visited_nodes.union(set(childrens))
        childrens = [set(graph.predecessors(children)) for children in childrens]
        childrens = list(set.union(*childrens))
        childrens = [children for children in childrens if set(graph.successors(children)).issubset(visited_nodes)]

    return post_probs


def posterior_correction_parallelize(graph, prior_probs, root, graph_nodes):
    # preliminary_probs
    pre_probs = Parallel(n_jobs=-1, verbose=10)(delayed(pre_probs_parallelize)(node, graph, prior_probs) for node in graph_nodes)
    pre_probs = dict(zip(graph_nodes, pre_probs))
    pre_probs[root] = prior_probs[root]

    # # posterior_correction
    post_probs = {}
    post_probs[root] = pre_probs[root]
    node = root
    childrens = list(graph.predecessors(node))
    visited_nodes = set({root})
    while len(childrens) > 0:
        post_probs_aux = Parallel(n_jobs=-1, verbose=10)(delayed(post_probs_parallelize)(node, graph, pre_probs[node], post_probs) for node in childrens)
        post_probs_aux = dict(zip(childrens, post_probs_aux))

        post_probs = {**post_probs, **post_probs_aux}
        visited_nodes = visited_nodes.union(set(childrens))
        childrens = [set(graph.predecessors(children)) for children in childrens]
        childrens = list(set.union(*childrens))
        childrens = [children for children in childrens if set(graph.successors(children)).issubset(visited_nodes)]

    return post_probs


def evaluate(prediction, graph, threshold=0.3):
    data_post = []
    GO_terms = list(prediction.columns)
    preds = {}
    pos = None
    root = find_root(graph)
    graph_nodes = list(graph.nodes)
    graph_nodes.remove(root)
    for index, row in tqdm(list(prediction.iterrows())):
        probs = list(row)
        prior_probs = dict(zip(GO_terms, probs))
        post_probs = posterior_correction_2(graph, prior_probs, root, graph_nodes)
        _, probs = zip(*sorted(post_probs.items(), key=itemgetter(0)))
        data_post.append(probs)

        preds[index] = [node for node in GO_terms if post_probs[node] > threshold]

        # # ckeck if old posterior_correction is equal to the new one
        # # si comentas los ifs de las lineas 67 y 101, entonces post_probs y post_probs_old no van a ser iguales
        # post_probs_old = posterior_correction(graph, prior_probs)
        # post_probs_values = np.array(list(post_probs.values()))
        # post_probs_old_values = np.array(list(post_probs_old.values()))
        # if (post_probs != post_probs_old):
        #     _, probs_old = zip(*sorted(post_probs_old.items(), key=lambda x:x[0]))
        #     print('NOT EQUAL', np.sum(post_probs_values == 0) != (len(post_probs_values) - 1), np.sum(post_probs_old_values == 0) != (len(post_probs_old_values) - 1), (np.array(probs_old) <= np.array(probs)).mean(), np.mean(probs_old))
        # else:
        #     print(np.sum(post_probs_values == 0) != (len(post_probs_values) - 1))


        #     if post_probs[node] > threshold:
        #         color_map.append('blue')
        #     else: color_map.append('green')
        # pos = show_graph(graph, pos=pos, with_labels=True, node_color=color_map)

        # print(index, check_sanity(graph, post_probs))

    data_post = np.array(data_post)

    post_results = pd.DataFrame(data=data_post, columns=prediction.columns, index=prediction.index)

    return post_results, preds


def evaluate2(prediction, graph, thresholds):
    data_post = []
    GO_terms = list(prediction.columns)
    preds = {th:{} for th in thresholds}
    pos = None
    root = find_root(graph)
    graph_nodes = list(graph.nodes)
    graph_nodes.remove(root)
    for index, row in tqdm(list(prediction.iterrows())):
        probs = list(row)
        prior_probs = dict(zip(GO_terms, probs))
        post_probs = posterior_correction_2(graph, prior_probs, root, graph_nodes)
        _, probs = zip(*sorted(post_probs.items(), key=itemgetter(0)))
        data_post.append(probs)

        for th in thresholds:
            preds[th][index] = [node for node in GO_terms if post_probs[node] > float(th)]

    data_post = np.array(data_post)

    post_results = pd.DataFrame(data=data_post, columns=prediction.columns, index=prediction.index)

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


def Hprecision_micro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    numerator = 0
    denominator = 0
    for gene, y_pred in Y_pred.items():
        y_true = Y_true.get(gene, [])
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        numerator += len(P.intersection(T))
        denominator += len(P)
    return numerator / denominator


def Hrecall_micro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    numerator = 0
    denominator = 0
    for gene, y_pred in Y_pred.items():
        y_true = Y_true.get(gene, [])
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        numerator += len(P.intersection(T))
        denominator += len(T)
    return numerator / denominator


def HF1_micro(graph, Y_pred, Y_true):
    hprec = Hprecision_micro(graph, Y_pred, Y_true)
    hrec = Hrecall_micro(graph, Y_pred, Y_true)
    return (2 * hprec * hrec) / (hprec + hrec)


def metrics_micro(graph, Y_pred, Y_true):
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    res = {th:[] for th in Y_pred}
    for th in Y_pred:
        for gene, y_pred in Y_pred[th].items():
            if gene in Y_true: y_true = Y_true[gene]
            else: y_true = []
            P = ancestors(graph, y_pred)
            T = ancestors(graph, y_true)
            numerator += len(P.intersection(T))
            denominator2 += len(T)
            denominator1 += len(P)
        hprec = numerator / denominator1
        hrec = numerator / denominator2
        hf1 = (2 * hprec * hrec) / (hprec + hrec)
        res[th] = [hprec, hrec, hf1]

    return res


def Hprecision_macro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    value = 0
    for gene, y_pred in Y_pred.items():
        y_true = Y_true.get(gene, [])
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        value += len(P.intersection(T)) / len(P)
    return value / len(Y_pred)


def Hrecall_macro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    value = 0
    for gene, y_pred in Y_pred.items():
        y_true = Y_true.get(gene, [])
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        value += len(P.intersection(T)) / len(T)
    return value / len(Y_pred)


def HF1_macro(graph, Y_pred, Y_true):
    value = 0
    for gene, y_pred in Y_pred.items():
        y_true = Y_true.get(gene, [])
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        P_inter_T = len(P.intersection(T))
        value += ( 2 * (P_inter_T**2) / (len(P) * len(T)) ) / ( (P_inter_T / len(P)) + (P_inter_T / len(T)) )
    return value / len(Y_pred)


if __name__ == '__main__':
    ontology_path = '../datasets/raw/obo/go-basic.obo'
    gos, ontology_gos, go_alt_ids, ontology_graphs = parse_obo(ontology_path)

    PATHS = ['results', 'complete']
    ORGANISMS_ID = ['scer', 'celegans', 'dmel', 'hg', 'mm']
    ONTOLOGIES = ['cellular_component', 'molecular_function', 'biological_process']

    def evaluation2(PATH, organism_id, ontology):
        print(organism_id, ontology)

        results = pd.read_csv('./{}/{}_model_{}_{}.csv'.format(PATH, PATH, organism_id, ontology), sep='\t', dtype={'seqname':str}).set_index(['pos', 'seqname'])
        # random_results = pd.read_csv('random_results_model_{}_{}.csv'.format(organism_id, ontology), sep='\t', dtype={'seqname':str}).set_index(['pos', 'seqname'])

        go_ids = results.columns.tolist()
        ontology_subgraph = ontology_graphs[ontology].subgraph(go_ids)

        # post_results, preds = evaluate(results, ontology_subgraph, threshold=0.3)
        # random_post_results, random_preds = evaluate(random_results, ontology_subgraph, threshold=0.3)

        random_results = np.random.uniform(size=results.shape)*(results != 0)
        random_results = pd.DataFrame(random_results, columns=results.columns, index=results.index)

        flatten = np.array(results).flatten()
        ind = np.argwhere(flatten != 0)
        ind2 = np.random.permutation(ind)
        flatten.flat[ind] = flatten[ind2]
        random_permutation = flatten.reshape(-1, results.shape[1])
        random_permutation = pd.DataFrame(random_permutation, columns=results.columns, index=results.index)

        thresholds = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95']

        post_results, preds = evaluate2(results, ontology_subgraph, thresholds)
        random_post_results, random_preds = evaluate2(random_results, ontology_subgraph, thresholds)
        permutation_post_results, permutation_preds = evaluate2(random_permutation, ontology_subgraph, thresholds)

        if not os.path.exists('./{}'.format(PATH)):
            os.mkdir('./{}'.format(PATH))
        if not os.path.exists('./{}/post'.format(PATH)):
            os.mkdir('./{}/post'.format(PATH))

        post_results.to_csv('./{}/post/post_results_{}_{}.csv'.format(PATH, organism_id, ontology), sep='\t', index=True)
        # random_post_results.to_csv('random_post_results.csv', sep='\t', index=True)

        annots_test = pd.read_csv('../datasets/processed/{}/{}/annots_test.csv'.format(organism_id, ontology), sep='\t', dtype={'seqname':str})
        true_annots = {(pos, chromosome):list(set(df['go_id'].values)) for (pos, chromosome), df in annots_test.groupby(['pos', 'seqname'])}

        # score = Hprecision_micro(ontology_subgraph, preds, true_annots)
        # random_score = Hprecision_micro(ontology_subgraph, random_preds, true_annots)
        # print('Hprecision_micro', score, random_score)
        # score = Hrecall_micro(ontology_subgraph, preds, true_annots)
        # random_score = Hrecall_micro(ontology_subgraph, random_preds, true_annots)
        # print('Hrecall_micro', score, random_score)

        metrics_results = metrics_micro(ontology_subgraph, preds, true_annots)
        metrics_random = metrics_micro(ontology_subgraph, random_preds, true_annots)
        metrics_permutation = metrics_micro(ontology_subgraph, permutation_preds, true_annots)

        evaluation = []
        for th in thresholds:
            metrics = list(itertools.chain.from_iterable([metrics_results[th], metrics_random[th], metrics_permutation[th]]))        
            evaluation.append(metrics)

        columns = [
            'prec', 'recall', 'f1',
            'rand_prec', 'rand_recall', 'rand_f1',
            'perm_prec', 'perm_recall', 'perm_f1'
        ]
        evaluation = np.array(evaluation)
        evaluation = pd.DataFrame(evaluation, columns=columns, index=thresholds)

        columns = ['prec', 'rand_prec', 'perm_prec',
                'recall', 'rand_recall', 'perm_recall',
                'f1', 'rand_f1', 'perm_f1',
                ]
        evaluation = evaluation[columns]

        if not os.path.exists('./{}'.format(PATH)):
            os.mkdir('./{}'.format(PATH))
        if not os.path.exists('./{}/metrics'.format(PATH)):
            os.mkdir('./{}/metrics'.format(PATH))

        evaluation.to_csv('./{}/metrics/metrics_{}_{}.csv'.format(PATH, organism_id, ontology), sep='\t', index=True)

    # evaluation2('results', 'mm', 'cellular_component')

    Parallel(n_jobs=-1, verbose=10)(delayed(evaluation2)(p[0], p[1], p[2]) for p in itertools.product(['complete'], ['mm'], ['biological_process']))
    # Parallel(n_jobs=-1, verbose=10)(delayed(evaluation2)(p[0], p[1], p[2]) for p in itertools.product(PATHS, ORGANISMS_ID, ONTOLOGIES))
