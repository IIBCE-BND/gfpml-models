import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import pandas as pd
import os
from joblib import Parallel, delayed

import parsers.annot as annot
import parsers.gtf as gtf
import parsers.obo as obo


ontology_path = '../datasets/raw/obo/go-basic.obo'

data_load = {
    'celegans': {
        'gtf': '../datasets/raw/gtf/Caenorhabditis_elegans.WBcel235.94.gtf',
        'gaf': '../datasets/raw/gaf/wb.gaf',
        'centromere': ''
    },
    'dmel': {
        'gtf': '../datasets/raw/gtf/Drosophila_melanogaster.BDGP6.94.gtf',
        'gaf': '../datasets/raw/gaf/fb.gaf',
        'centromere': ''
    },
    'hg': {
        'gtf': '../datasets/raw/gtf/Homo_sapiens.GRCh38.94.gtf',
        'gaf': '../datasets/raw/gaf/goa_human.gaf',
        'centromere': '../datasets/raw/centromeres/hg-centromere.csv'
    },
    'mm': {
        'gtf': '../datasets/raw/gtf/Mus_musculus.GRCm38.94.gtf',
        'gaf': '../datasets/raw/gaf/mgi.gaf',
        'centromere': ''
    },
    'scer': {
        'gtf': '../datasets/raw/gtf/Saccharomyces_cerevisiae.R64-1-1.94.gtf',
        'gaf': '../datasets/raw/gaf/sgd.gaf',
        'centromere': '../datasets/raw/centromeres/scer-centromere.csv'
    }
}


gos, ontology_gos, go_alt_ids, ontology_graphs = obo.parse_obo(ontology_path)

# generate genome, annotations and hierarchical annotations
for organism_id in data_load:
    print(organism_id)
    print(organism_id)

    genome = gtf.parse_gtf(data_load[organism_id]['gtf'], data_load[organism_id]['centromere'])
    annots = annot.parse_annot(data_load[organism_id]['gaf'], go_alt_ids)

    # 'name' is not in genome.columns for scer, instead we must use 'id'
    if 'name' in genome.columns: gene_identifier = 'name'
    elif 'id' in genome.columns: gene_identifier = 'id'

    annots = annots[annots['gene_id'].isin(genome[gene_identifier].values)]
    annots['pos'] = annots['gene_id'].replace(genome.set_index(gene_identifier)['pos'].to_dict())
    annots['seqname'] = annots['gene_id'].replace(genome.set_index(gene_identifier)['seqname'].to_dict())

    data_path = '../datasets/preprocessed/'
    directory = data_path + organism_id
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    expanded_annots = []
    for ontology in ontology_gos:
        print(ontology)
        annots_ontology = annots[annots['go_id'].isin(ontology_gos[ontology])]
        expanded_annots_ontology = annot.expand_annots(annots_ontology, ontology_graphs[ontology])
        expanded_annots_ontology = expanded_annots_ontology.sort_values(['seqname', 'pos', 'go_id'])
        expanded_annots_ontology.to_csv('{}/{}_{}.csv'.format(directory, 'expanded_annots', ontology),
                                        sep='\t',
                                        index=False)
        expanded_annots_ontology['ontology'] = ontology
        expanded_annots.append(expanded_annots_ontology)
    expanded_annots = pd.concat(expanded_annots).sort_values(['seqname', 'pos', 'go_id'])
    expanded_annots.to_csv('{}/{}.csv'.format(directory, 'expanded_annots'), sep='\t', index=False)

    genome = genome.sort_values(['seqname', 'pos'])
    genome.to_csv('{}/{}.csv'.format(directory, 'genome'), sep='\t', index=False)
    annots = annots.sort_values(['seqname', 'pos', 'go_id'])
    annots.to_csv('{}/{}.csv'.format(directory, 'annots'), sep='\t', index=False)
