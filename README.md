# gfpml-models

## Preprocessing

Extrae de los datos crudos el genoma y las anotaciones jerárquicas.
<!-- Extract from raw data genome and hierarchical annotations. -->

## Processing

De los datos generados por preprocessing genera las curvas de lea y score para cada organismo y ontología.

Parámetos

+ train_size = 0.8. Es un valor en el intervalo [0,1] que representa la proporción de genes que serán utilizados como muestra de entrenamiento de los modelos. A partir de los genes que se seleccionan con este parámetro se calculan lea y el score.
+ MIN_LIST_SIZE_TRAIN = 40, MIN_LIST_SIZE_TEST = 10. Son valores enteros que determinan el tamaño mínino de genes que tiene que tener un término GO para calcular su curva lea y score. El modelo predictor sólo considerará aquellos términos GO que satisfagan estas cantidades mínimas.
+ window_sizes = [5, 100]. Es una lista de enteros de posibles tamaños de ventanas para lea.

## Training

Sigue la implementación del model descripto en [1](https://pdfs.semanticscholar.org/44f1/4625bbcd74f364810e87dd115d17b4445bb2.pdf) y [2](https://www.tandfonline.com/doi/pdf/10.1080/13102818.2018.1521302?needAccess=true) con algunas variantes. El modelo utilizado es un SVM, pero puede ser modificado con facilidad.

## Evaluation

Se implementan las métricas hierarchical precision y hierarchical recall. Estas métricas son las citadas en los paper mencionados y además son empleadas en las competencias CAFA.

## cluster.uy

### Acceso al cluster
1. Agregar a `~/.ssh/config` la siguiente configuración:
```
Host clusteruy
  HostName login.cluster.uy
  User ...
  IdentityFile ~/.ssh/bnd
  IdentitiesOnly yes
  ForwardAgent yes
```
2. Acceder a la máquina de login:
```
> ssh clusteruy
```
3. Una vez de haber ingresado a la máquina de login, moverse a la máquina de compilación:
```
> ssh compilacion
```

### Copiar código y datos al cluster:
1. Moverse a la carpeta correspondiente al proyecto:
```
> cd path/to/gfpml-models
```
2. Sync con cluster:
```
> rsync -e "ssh -i /home/usuario_local/.ssh/key_clusteruy" -r usuario@login.cluster.uy:/clusteruy/home/usuario/gfpml-models/ .
```

### Comandos útiles

Ejecutar comandos de processing y training:
```
> sbatch procesing.sh
> sbatch model.sh
```

Ver información de un job:
```
> scontrol show jobid -dd job_id
```

Ver estado de los jobs del usuario:
```
> squeue -u usuario
```