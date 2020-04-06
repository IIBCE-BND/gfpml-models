import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

import matplotlib.pyplot as plt

from itertools import chain
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

ONTOLOGIES = ['biological_process', 'cellular_component', 'molecular_function']

def get_gene_identifier(genome):
    if 'name' in genome.columns: gene_identifier = 'name'
    elif 'id' in genome.columns: gene_identifier = 'id'
    return gene_identifier

def reshape_lea(gene_pos, window_size):
    n = len(gene_pos)
    w = window_size
    ans = np.zeros((n, 2 * w + 1))
    for i in range(2 * w + 1):
        ans[min(n, max(w - i, 0)):max(0, min(n + w - i, n)),i] = gene_pos[min(n, max(i - w, 0)):max(0, min(n, n + i - w))]
    return ans

def calculate_enrichment(gene_pos, window_size):
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


def calculate_seq_lea_parallelize(go_id, go_annots_train, organism_id, ontology, save_path_ont, len_chromosomes, window_sizes):
    data = []
    for seqname, seq_annots in go_annots_train.groupby('seqname'):
        positions = list(seq_annots.pos.values)
        data_seq = {'pos': range(len_chromosomes[seqname]), 'seqname': seqname}
        mask_train = np.isin(range(len_chromosomes[seqname]), positions)
        for ws in window_sizes:
            seq_lea = calculate_enrichment(mask_train, ws)
            data_seq['lea_{}'.format(ws)] = seq_lea
        data_seq = pd.DataFrame(data_seq)
        data.append(data_seq)
    data = pd.concat(data)
    columns = ['pos', 'seqname'] + ['lea_{}'.format(ws) for ws in window_sizes]
    data = data[columns]
    data.to_csv('{}/{}.csv'.format(save_path_ont, go_id), index=False, sep='\t')


def calculate_seq_lea(genome, expanded_annots, organism_id, window_sizes, th=100):
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
            delayed(calculate_seq_lea_parallelize)(go_id,
                                                          go_annots_train,
                                                          organism_id,
                                                          ontology,
                                                          save_path_ont,
                                                          len_chromosomes,
                                                          window_sizes)
            for go_id, go_annots_train in annots_train.groupby('go_id')
        )



def select_annots(annots, seq_genes):
    gene_identifier = get_gene_identifier(seq_genes)
    return annots[annots['gene_id'].isin(seq_genes[gene_identifier].values)]



if __name__ == '__main__':
    data_path = '../datasets/preprocessed/'
    window_sizes = [5, 10, 20, 50, 100]
    window_sizes = [5, 100]

    for organism_id in os.listdir(data_path):
        print(organism_id)
        directory = '{}/{}/'.format(data_path, organism_id)

        genome = pd.read_csv('{}/genome.csv'.format(directory), sep='\t')
        gene_identifier = get_gene_identifier(genome)

        expanded_annots = pd.read_csv('{}/expanded_annots.csv'.format(directory), sep='\t')
        expanded_annots = expanded_annots[expanded_annots.gene_id.isin(genome[gene_identifier])]

        calculate_seq_lea(genome,
                                expanded_annots,
                                organism_id,
                                window_sizes)



        ########## AFTER COMPUTE ##########
        ###################################

        # genome_train = pd.read_csv('../datasets/processed/{}/genome_train.csv'.format(organism_id), sep='\t')
        # genome_test = pd.read_csv('../datasets/processed/{}/genome_test.csv'.format(organism_id), sep='\t')
        # len_chromosomes = dict(genome.groupby('seqname').size())
        # # print(len_chromosomes)

        # for ontology in ONTOLOGIES:
        #     print(ontology)
        #     data_path = '../datasets/processed/{}/{}/'.format(organism_id, ontology)
        #     annots_train = pd.read_csv('{}/annots_train.csv'.format(data_path), sep='\t')
        #     annots_test = pd.read_csv('{}/annots_test.csv'.format(data_path), sep='\t')

        #     go_ids = annots_train['go_id'].unique()
        #     for go_id in go_ids:
        #         mask_train = genome_train[['pos', 'seqname']].set_index(['pos', 'seqname'])
        #         mask_test = genome_test[['pos', 'seqname']].set_index(['pos', 'seqname'])
        #         data = []
        #         for seqname in len_chromosomes:
        #             df = pd.read_csv('{}/{}/{}.csv'.format(data_path, seqname, go_id), sep='\t')
        #             data.append(df)

        #         data = pd.concat(data)
        #         data = data.set_index(['pos', 'seqname'])
        #         data_train = data[data.index.isin(mask_train.index)]
        #         data_test = data[data.index.isin(mask_test.index)]

        #         data_train.to_csv('{}/{}_train.csv'.format(data_path, go_id), sep='\t', index=True)
        #         data_test.to_csv('{}/{}_test.csv'.format(data_path, go_id), sep='\t', index=True)

