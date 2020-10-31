import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

import parsers.obo as obo

from itertools import chain
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

ONTOLOGIES = ['biological_process', 'cellular_component', 'molecular_function']

def find_root(graph, node=None):
    # Find the root of the graph
    if node is None:
        node = list(graph.nodes())[0]
    parents = list(graph.successors(node))
    if len(parents) == 0: return node
    else: return find_root(graph, parents[0])

def reshape_lea(gene_pos, window_size):
    # Function to compute LEA vectorized
    n = len(gene_pos)
    w = window_size
    ans = np.zeros((n, 2 * w + 1))
    for i in range(2 * w + 1):
        ans[min(n, max(w - i, 0)):max(0, min(n + w - i, n)),i] = gene_pos[min(n, max(i - w, 0)):max(0, min(n, n + i - w))]
    return ans

def calculate_enrichment(gene_pos, window_size):
    # Compute LEA using gene_pos for window_size
    n = len(gene_pos)
    w = window_size
    if w < n - 1:
        m = min(2 * w + 1, n)
        h = m - (w + 1)
        window_sizes = np.repeat(m, n)
        window_sizes[:h] = np.arange(w + 1, m)
        window_sizes[-h:] = np.arange(m - 1, w, -1)
    else:
        window_sizes = np.repeat(n, n)

    lea_array = (reshape_lea(gene_pos, window_size).sum(axis=1) / window_sizes) / (gene_pos.mean())
    return lea_array

# def calculate_seq_lea_score(seq_genes, seq_annots, organism_id, window_sizes, th=100):
#     gene_identifier = get_gene_identifier(seq_genes)
#     chromosome = seq_genes.seqname.values[0]

#     # seq_genes = seq_genes.sort_values(['start', 'strand', 'size'], ascending=True) # correct order
#     # seq_genes['pos'] = range(len(seq_genes))
#     seq_len = len(seq_genes)

#     train_size = 0.8
#     gene_train, gene_test, pos_train, pos_test = train_test_split(seq_genes[gene_identifier].values, seq_genes.pos.values, train_size=train_size)

#     for ontology, seq_annots_ontology in seq_annots.groupby('ontology'):
#         seq_lea_ontology = {window_size: {} for window_size in window_sizes}
#         seq_score = {}
#         for go_id, seq_annots_go in seq_annots_ontology.groupby('go_id'):
#             seq_annots_go['pos'] = seq_annots_go['gene_id'].replace(seq_genes.set_index(gene_identifier)['pos'].to_dict())
#             go_annots_pos = seq_annots_go[['gene_id', 'pos']]

#             go_annots_pos_train = go_annots_pos[go_annots_pos['pos'].isin(pos_train)].drop_duplicates()
#             go_annots_pos_test = go_annots_pos[go_annots_pos['pos'].isin(pos_test)].drop_duplicates()

#             if (len(seq_annots_go) < th) or (len(go_annots_pos_train) == 0) or (len(go_annots_pos_test) == 0):
#                 continue

#             save_path = '../datasets/processed/'
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)

#             save_path = '{}/{}/'.format(save_path, organism_id)
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)

#             save_path = '{}/{}/'.format(save_path, chromosome)
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)

#             if not os.path.exists('{}/train.csv'.format(save_path)):
#                 train_data = {'pos':pos_train, 'gene_id':gene_train}
#                 train_df = pd.DataFrame(data=train_data)
#                 test_data = {'pos':pos_test, 'gene_id':gene_test}
#                 test_df = pd.DataFrame(data=test_data)

#                 train_df.to_csv('{}/train.csv'.format(save_path), sep='\t', index=False)
#                 test_df.to_csv('{}/test.csv'.format(save_path), sep='\t', index=False)

#             go_annots_pos_train.to_csv('{}/{}_train.csv'.format(save_path, go_id), sep='\t', index=False)
#             go_annots_pos_test.to_csv('{}/{}_test.csv'.format(save_path, go_id), sep='\t', index=False)


#             seq_score[go_id] = score_function(seq_len, pos_train)

#             mask_train = np.isin(range(seq_len), pos_train)
#             for ws in window_sizes:
#                 seq_lea_ontology[ws][go_id] = calculate_enrichment(mask_train, ws)

#         if len(seq_score) > 0:
#             seq_score = pd.DataFrame(data=seq_score)
#             seq_score.to_csv('{}/seq_score_{}.csv'.format(save_path, ontology), sep='\t', index=False)

#         for ws in window_sizes:
#             if len(seq_lea_ontology[ws]) > 0:
#                 seq_lea_ontology[ws] = pd.DataFrame(data=seq_lea_ontology[ws])
#                 seq_lea_ontology[ws].to_csv('{}/seq_lea_{}_{}.csv'.format(save_path, ws, ontology), sep='\t', index=False)


def calculate_seq_lea_score2_parallelize(go_id, go_annots_train, organism_id, ontology, save_path_ont, len_chromosomes, window_sizes):
    data = []
    for seqname, seq_annots in go_annots_train.groupby('seqname'):
        positions = list(seq_annots.pos.values)

        seq_score = score_function(len_chromosomes[seqname], positions)
        data_seq = {'pos': range(len_chromosomes[seqname]), 'seqname': seqname, 'score': seq_score}

        mask_train = np.isin(range(len_chromosomes[seqname]), positions)
        for ws in window_sizes:
            seq_lea = calculate_enrichment(mask_train, ws)
            data_seq['lea_{}'.format(ws)] = seq_lea
        data_seq = pd.DataFrame(data_seq)
        data.append(data_seq)
    data = pd.concat(data)
    columns = ['pos', 'seqname', 'score'] + ['lea_{}'.format(ws) for ws in window_sizes]
    data = data[columns]
    data.to_csv('{}/{}.csv'.format(save_path_ont, go_id), index=False, sep='\t')


def calculate_seq_lea_score2(genome, expanded_annots, organism_id, window_sizes, th=100):
    gene_identifier = get_gene_identifier(genome)
    train_size = 0.8

    save_path = '../datasets/processed/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = '{}/{}/'.format(save_path, organism_id)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    len_chromosomes = dict(genome.groupby('seqname').size())

    genome_train, genome_test, _, _ = train_test_split(genome, genome, train_size=train_size)
    genome_train.to_csv('{}/genome_train.csv'.format(save_path), index=False, sep='\t')
    genome_test.to_csv('{}/genome_test.csv'.format(save_path), index=False, sep='\t')

    MIN_LIST_SIZE_TRAIN = 40
    MIN_LIST_SIZE_TEST = 10

    for ontology, exp_annots_ontology in expanded_annots.groupby('ontology'):
        save_path_ont = '{}/{}/'.format(save_path, ontology)
        if not os.path.exists(save_path_ont):
            os.mkdir(save_path_ont)

        annots_train = exp_annots_ontology[exp_annots_ontology.gene_id.isin(genome_train[gene_identifier])]
        annots_test = exp_annots_ontology[exp_annots_ontology.gene_id.isin(genome_test[gene_identifier])]

        grouped_train = annots_train.groupby('go_id')
        grouped_test = annots_test.groupby('go_id')

        # grouped_train = np.array(grouped_train)[np.array(grouped_train.size() >= th*train_size)]
        grouped_train = np.array(grouped_train)[np.array(grouped_train.size() >= MIN_LIST_SIZE_TRAIN)]
        # grouped_test = np.array(grouped_test)[np.array(grouped_test.size() >= th*(1 - train_size))]
        grouped_test = np.array(grouped_test)[np.array(grouped_test.size() >= MIN_LIST_SIZE_TEST)]

        gos_train, _ = zip(*grouped_train)
        gos_test, _ = zip(*grouped_test)
        gos_train, gos_test = set(gos_train), set(gos_test)
        gos_inter = gos_train & gos_test
        grouped_train = dict((go_id, go_annots) for (go_id, go_annots) in grouped_train if go_id in gos_inter)
        grouped_test = dict((go_id, go_annots) for (go_id, go_annots) in grouped_test if go_id in gos_inter)

        print(ontology, len(gos_inter), len(gos_inter) / len(exp_annots_ontology.groupby('go_id')))

        annots_train = annots_train[annots_train['go_id'].isin(gos_inter)]
        annots_test = annots_test[annots_test['go_id'].isin(gos_inter)]
        annots_train.to_csv('{}/annots_train.csv'.format(save_path_ont), index=False, sep='\t')
        annots_test.to_csv('{}/annots_test.csv'.format(save_path_ont), index=False, sep='\t')

        Parallel(n_jobs=-1, verbose=10)(
            delayed(calculate_seq_lea_score2_parallelize)(go_id,
                                                          go_annots_train,
                                                          organism_id,
                                                          ontology,
                                                          save_path_ont,
                                                          len_chromosomes,
                                                          window_sizes)
            for go_id, go_annots_train in annots_train.groupby('go_id')
        )


def calculate_seq_lea_parallelize(go_id, go_annots_train, organism_id, ontology, save_path_ont, len_chromosomes, window_sizes):
    '''
    Compute LEA for go_id using annotations in go_annots_train for each window_size in window_sizes.
        go_id: GO term to compute LEA
        go_annots_train: annotations of go_id
        save_path_ont: path to save LEA
        len_chromosomes: dictionary of chromosomes and length of this chromosome in genome
        window_sizes: window_sizes for compute LEA
    '''
    data = []
    for seqname, seq_annots in go_annots_train.groupby('seqname'):
        positions = (seq_annots.reset_index(inplace=False)).pos.values

        # seq_score = score_function(len_chromosomes[seqname], positions)
        # data_seq = {'pos': range(len_chromosomes[seqname]), 'seqname': seqname, 'score': seq_score}
        data_seq = {'pos': range(len_chromosomes[seqname]), 'seqname': seqname}

        mask_train = np.isin(range(len_chromosomes[seqname]), positions)
        for ws in window_sizes:
            seq_lea = calculate_enrichment(mask_train, ws)
            data_seq['lea_{}'.format(ws)] = seq_lea
        data_seq = pd.DataFrame(data_seq)
        data.append(data_seq)
    data = pd.concat(data)
    # columns = ['pos', 'seqname', 'score'] + ['lea_{}'.format(ws) for ws in window_sizes]
    columns = ['pos', 'seqname'] + ['lea_{}'.format(ws) for ws in window_sizes]
    data = data[columns]
    data.to_csv('{}/{}.csv'.format(save_path_ont, go_id), sep='\t', index=False)


def calculate_seq_lea(genome, expanded_annots, organism_id, window_sizes, depths):
    '''
    Split genome and expanded_annots in train and test sets and compute LEA for GO terms with annotations in training set of 
    expanded_annots for each window_size in window_sizes.
        genome: genome for organism_id
        expanded_annots: hierarchical annotations
        window_sizes: window_sizes for compute LEA
    '''
    train_size = 0.8

    save_path = '../datasets/processed/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = '{}/{}/'.format(save_path, organism_id)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    len_chromosomes = dict(genome.groupby('seqname').size())

    genome_train, genome_test, _, _ = train_test_split(genome, genome, train_size=train_size)
    genome_train.to_csv('{}/genome_train.csv'.format(save_path), sep='\t', index=False)
    genome_test.to_csv('{}/genome_test.csv'.format(save_path), sep='\t', index=False)

    MIN_LIST_SIZE_TRAIN = 40
    MIN_LIST_SIZE_TEST = 10
    MAX_DEPTH = 1000

    for ontology, exp_annots_ontology in expanded_annots.groupby('ontology'):
        save_path_ont = '{}/{}/'.format(save_path, ontology)
        if not os.path.exists(save_path_ont):
            os.mkdir(save_path_ont)

        annots_train = exp_annots_ontology[exp_annots_ontology.index.isin(genome_train.index)]
        annots_test = exp_annots_ontology[exp_annots_ontology.index.isin(genome_test.index)]

        grouped_train = annots_train.groupby('go_id')
        grouped_test = annots_test.groupby('go_id')

        # dephs_train = np.array([depths[go] for go,_ in grouped_train])
        # dephs_test = np.array([depths[go] for go,_ in grouped_test])

        grouped_train = np.array(grouped_train)[
            np.array(grouped_train.size() >= MIN_LIST_SIZE_TRAIN)
            # np.array(dephs_train <= MAX_DEPTH)
        ]
        grouped_test = np.array(grouped_test)[
            np.array(grouped_test.size() >= MIN_LIST_SIZE_TEST)
            # np.array(dephs_train <= MAX_DEPTH)
        ]

        gos_train, _ = zip(*grouped_train)
        gos_test, _ = zip(*grouped_test)
        gos_train, gos_test = set(gos_train), set(gos_test)
        gos_inter = gos_train & gos_test
        grouped_train = dict((go_id, go_annots) for (go_id, go_annots) in grouped_train if go_id in gos_inter)
        grouped_test = dict((go_id, go_annots) for (go_id, go_annots) in grouped_test if go_id in gos_inter)

        print(ontology, len(gos_inter), len(gos_inter) / len(exp_annots_ontology.groupby('go_id')))

        annots_train = annots_train[annots_train['go_id'].isin(gos_inter)]
        annots_test = annots_test[annots_test['go_id'].isin(gos_inter)]
        annots_train.to_csv('{}/annots_train.csv'.format(save_path_ont), sep='\t', index=False)
        annots_test.to_csv('{}/annots_test.csv'.format(save_path_ont), sep='\t', index=False)

        Parallel(n_jobs=-1, verbose=10)(
            delayed(calculate_seq_lea_parallelize)(
                go_id,
                go_annots_train,
                organism_id,
                ontology,
                save_path_ont,
                len_chromosomes,
                window_sizes
            ) for go_id, go_annots_train in annots_train.groupby('go_id')
        )


# def score_function(num_genes_in_chromosome, positions):
# #     return score for every gen position
#     genes_positions_GO = np.array(positions).reshape(-1, 1)
#     distances = pairwise_distances(genes_positions_GO, genes_positions_GO, metric='l1')
#     distances = np.sort(distances, axis=1)
#     mean = distances.mean(axis=0)
#     # median = np.median(distances, axis=0)

#     genes_positions = np.arange(num_genes_in_chromosome).reshape(-1, 1)
#     distances = pairwise_distances(genes_positions, genes_positions_GO, metric='l1')
#     norma = np.linalg.norm(np.sort(distances, axis=1) - mean, axis=1)
#     # norma = np.linalg.norm(np.sort(distances, axis=1) - median, axis=1)

#     score = num_genes_in_chromosome / (num_genes_in_chromosome + norma)

#     return score

# def select_annots(annots, seq_genes):
#     gene_identifier = get_gene_identifier(seq_genes)
#     return annots[annots['gene_id'].isin(seq_genes[gene_identifier].values)]


if __name__ == '__main__':
    data_path = '../datasets/preprocessed/'
    ontology_path = '../datasets/raw/obo/go-basic.obo'
    window_sizes = [5, 10, 20, 50, 100]

    gos, ontology_gos, go_alt_ids, ontology_graphs = obo.parse_obo(ontology_path)

    for organism_id in os.listdir(data_path):
        directory = '{}/{}/'.format(data_path, organism_id)

        genome = pd.read_csv('{}/genome.csv'.format(directory), sep='\t')
        genome = genome.set_index(['pos', 'seqname'])
        expanded_annots = pd.read_csv('{}/expanded_annots.csv'.format(directory), sep='\t')
        expanded_annots = expanded_annots.set_index(['pos', 'seqname'])

        depths = {}
        # for ontology, antology_annots in expanded_annots.groupby(['ontology']):
        #     ontology_subgraph = ontology_graphs[ontology].subgraph(antology_annots.go_id.unique())
        #     root = find_root(ontology_subgraph)
        #     graph_distances = nx.shortest_path_length(ontology_subgraph,target=root)
        #     depths = {**depths, **graph_distances}
        
        # expanded_annots['depths'] = expanded_annots['go_id'].map(depths)

        calculate_seq_lea(genome,
                          expanded_annots,
                          organism_id,
                          window_sizes,
                          depths)

