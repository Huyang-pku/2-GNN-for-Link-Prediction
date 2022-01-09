# 2-GNN-for-Link-Prediction
Fix some problems of Xiyuan's code:

1 Update node degree for every batch

2 Sample and save edges pairwisely

3 Add edge pooling in last layer

4 Compare concatenate pooling and element-wise product pooling


usage: main.reproduce

| dataset  | results              | results from Wang Xiyuan | results before |
|----------|----------------------|--------------------------|----------------|
| Celegans | 86.3                 |           87.8           |     83.1       |
|   Yeast  | 95.7                 |           91.5           |     95.8       |
|   Power  | 78.3                 |           52.1           |     76.9       |
|    PB    | 94.5                 |           94.2           |     93.5       |
|  Router  | 96.0                 |                          |     96.7       |
|   USAir  | 93.7                 |                          |     95.2       |
|    NS    | 98.6                 |                          |     97.9       |

TODO: 

Use optuna; 
Add predicting edge; 
Enlarge negative edge pool
