import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

from parsers.obo import parse_obo

import operator

ontology_path = '../datasets/raw/obo/go-basic.obo'
gos, ontology_gos, go_alt_ids, ontology_graphs = parse_obo(ontology_path)

organism_id = 'mm'
chromosome = '9'
ontology = 'biological_process'

# create example
data_path = '../datasets/processed/{}/{}/'.format(organism_id, chromosome)
df = pd.read_csv('{}/seq_score_{}.csv'.format(data_path, ontology), sep='\t')
GO_terms = list(df.columns)

pos_test = list(pd.read_csv('{}/test.csv'.format(data_path), sep='\t').pos)
print(pos_test)


data = np.random.uniform(size=(len(pos_test), len(df.columns)))
print(data.shape)
print(data)

df = pd.DataFrame(data=data, columns=GO_terms, index=pos_test)
data_path = '../datasets/results/{}/{}'.format(organism_id, chromosome)
df.to_csv('{}/results_{}.csv'.format(data_path, ontology), sep='\t', index=True)

data_path = '../datasets/results/{}/{}'.format(organism_id, chromosome)
results = pd.read_csv('{}/results_{}.csv'.format(data_path, ontology), sep='\t', index_col=0)
GO_terms = list(results.columns)
ontology_subgraph = ontology_graphs[ontology].subgraph(GO_terms)


def find_root(graph, node=None):
    if node == None:
        node = list(graph.nodes())[0]
    parents = list(graph.successors(node))
    if len(parents) == 0: return node
    else: return find_root(graph, parents[0])

def show_graph(graph, with_labels=True, node_color='blue'):
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


def evaluate(prediction): ### hacer que siempre vengan los terminos GO en orden alfabetico
    data_post = []
    threshold = 0.3
    GO_terms = list(prediction.columns)
    preds = {}
    for index, row in prediction.iterrows():
        probs = list(row)
        prior_probs = dict(zip(GO_terms, probs))
        post_probs = posterior_correction(ontology_subgraph, prior_probs)
        _, probs = zip(*sorted(post_probs.items(), key=lambda x:x[0]))
        data_post.append(probs)

        preds[index] = [node for node in post_probs if post_probs[node] > threshold]

        color_map = []
        for node in ontology_subgraph:
            if post_probs[node] > threshold:
                color_map.append('blue')
            else: color_map.append('green')
        show_graph(ontology_subgraph, with_labels=True, node_color=color_map)

        # print(index, check_sanity(ontology_subgraph, post_probs))

    data_post = np.array(data_post)

    df = pd.DataFrame(data=data_post, columns=prediction.columns, index=prediction.index)
    df.to_csv('{}/post_results_{}.csv'.format(data_path, ontology), sep='\t', index=True)


    return preds

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
        if gene in Y_true: y_true = Y_true[gene]
        else: y_true = []
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
        if gene in Y_true: y_true = Y_true[gene]
        else: y_true = []
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        numerator += len(P.intersection(T))
        denominator += len(T)
    return numerator / denominator

def Hprecision_macro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    value = 0
    for gene, y_pred in Y_pred.items():
        if gene in Y_true: y_true = Y_true[gene]
        else: y_true = []
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        value += len(P.intersection(T)) / len(P)
    return value / len(Y_pred)

def Hrecall_macro(graph, Y_pred, Y_true):
    # Y_pred and Y_true are dictionaries whose keys are genes and values are lists of GO terms
    value = 0
    for gene, y_pred in Y_pred.items():
        if gene in Y_true: y_true = Y_true[gene]
        else: y_true = []
        P = ancestors(graph, y_pred)
        T = ancestors(graph, y_true)
        value += len(P.intersection(T)) / len(T)
    return value / len(Y_pred)





preds = evaluate(results)


# print(preds)


genome = pd.read_csv('../datasets/preprocessed/{}/genome.csv'.format(organism_id), sep='\t')
# print(genome.shape)
seq_genome = genome[genome.seqname == chromosome]
seq_genome['pos'] = range(len(seq_genome)) # esto da un warning
# print(seq_genome.shape)

seq_genome = seq_genome[seq_genome.pos.isin(results.index)]
# print(seq_genome.shape)

if 'name' in seq_genome.columns: gene_identifier = 'name'
elif 'id' in seq_genome.columns: gene_identifier = 'id'
else: raise Exception

expanded_annots = pd.read_csv('../datasets/preprocessed/{}/expanded_annots_{}.csv'.format(organism_id, ontology), sep='\t')
print(expanded_annots.shape)
expanded_annots = expanded_annots[expanded_annots['gene_id'].isin(seq_genome[gene_identifier].values)]
expanded_annots = expanded_annots[expanded_annots['go_id'].isin(GO_terms)]
print(expanded_annots.shape)
expanded_annots['pos'] = expanded_annots['gene_id'].replace(seq_genome.set_index(gene_identifier)['pos'].to_dict())

true_annots = {pos:list(set(df['go_id'].values)) for pos, df in expanded_annots.groupby('pos')}

score = Hprecision_micro(ontology_subgraph, preds, true_annots)
print(score)
score = Hrecall_micro(ontology_subgraph, preds, true_annots)
print(score)


# show_graph(ontology_subgraph)


