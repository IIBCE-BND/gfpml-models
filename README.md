# gfpml-models

## preprocessing

Extrae de los datos crudos el genoma y las anotaciones jerárquicas.
<!-- Extract from raw data genome and hierarchical annotations. -->

## processing

De los datos generados por preprocessing genera las curvas de lea y score para cada organismo y ontología.

Parámetos

+ train_size = 0.8. Es un valor en el intervalo [0,1] que representa la proporción de genes que serán utilizados como muestra de entrenamiento de los modelos. A partir de los genes que se seleccionan con este parámetro se calculan lea y el score.
+ MIN_LIST_SIZE_TRAIN = 40, MIN_LIST_SIZE_TEST = 10. Son valores enteros que determinan el tamaño mínino de genes que tiene que tener un término GO para calcular su curva lea y score. El modelo predictor sólo considerará aquellos términos GO que satisfagan estas cantidades mínimas.
+ window_sizes = [5, 100]. Es una lista de enteros de posibles tamaños de ventanas para lea.

## model

Sigue la implementación del model descripto en [1](https://pdfs.semanticscholar.org/44f1/4625bbcd74f364810e87dd115d17b4445bb2.pdf) y [2](https://www.tandfonline.com/doi/pdf/10.1080/13102818.2018.1521302?needAccess=true) con algunas variantes. El modelo utilizado es un SVM, pero puede ser modificado con facilidad.

## evaluation

Se implementan las métricas hierarchical precision y hierarchical recall. Estas métricas son las citadas en los paper mencionados y además son empleadas en las competencias CAFA.
