# MCNS

Implemetation of "MCNS", Understanding Negative Sampling in Graph Representation Learning.

## Introduction
MCNS is a novel negative sampling strategy for graph embedding learning, which approximates the positive distribution with self-contrast approximation and accelerates negative sampling by Metropolis-Hastings. The experimental results demonstrate its robustness and superiorities.

## Preparation
* Python 3.7
* Tensorflow 1.14.0


## Training
### Training on the existing datasets
You can use ```$ ./experiments/***.sh``` to train MCNS model on the recommendation task. For example, if you want to train on the Amazon dataset, you can run ```$ ./experiments/amazon.sh``` or ```python main.py --input data/amazon/``` to train MCNS model.

### Training on your own datasets
if you want to train MCNS on your own dataset, you should prepare the following four files:
* train.txt: Each line represents an edge ```<node1> <node2>```.
* valid.txt: the same format with train.txt
* test.txt: the same format with train.txt
* test_neg.txt: For each node, we select some unconnected nodes as negs for evaluation. For Amazon and Alibaba datasets, we select 500 negs, and all unconnected negs for ml-100k to evaluate hits@k and MRR.  


## Dataset
* ml-100k contains 943 users, 1,682 items and 100,000 edges.
* Amazon contains 192,403 users, 63,001 items and 1,689,188 edges.
* Alibaba contains 106,042 users, 53591 items and 907,470 edges.

## Performance 
The performance of MCNS on recommendation task with ML-100K, Amazon and Alibaba are tested:

  Metrics| ML-100k  | Amazon | Alibaba 
 ---- | ----- | ------  | ------  
 MRR  | 0.114 | 0.108  | 0.116 
 Hits@30  | 0.413 | 0.386  | 0.387 
