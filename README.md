# gfpml-models

## preprocessing

Extract the genome and hierarchical annotations from raw data.

## processing

From the data generated by preprocessing, it generates the LEA curves for each organism and ontology.

Parameters

+ train_size = 0.8: It is a value in the interval [0,1] that represents the proportion of genes that will be used as a training sample of the models. LEA is calculated from the genes that are selected with this parameter.
+ MIN_LIST_SIZE_TRAIN = 40, MIN_LIST_SIZE_TEST = 10: They are integer values that determine the minimum size of genes that a GO term must have to calculate its LEA. The predictor model will only consider those GO terms that satisfy these minimum quantities.
+ window_sizes = [5, 100]: It is the list of possible window sizes fro LEA.

## model

Follow the implementation of the model described in [1](https://pdfs.semanticscholar.org/44f1/4625bbcd74f364810e87dd115d17b4445bb2.pdf) and [2](https://www.tandfonline.com/doi/pdf/10.1080/13102818.2018.1521302?needAccess=true) with some variations.

## evaluation

Implementation of hierarchical precision and hierarchical recall metrics. These metrics are those cited in the mentioned papers and are also used in the CAFA competitions.
