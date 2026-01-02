# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

This is the official code release of the following paper: 

Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. [Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning](https://arxiv.org/abs/2104.10353). SIGIR 2021.

<img src="https://github.com/Lee-zix/RE-GCN/blob/master/img/regcn.png" alt="regcn_architecture" width="700" class="center">

## Quick Start

### Environment variables & dependencies
```
conda create -n regcn python=3.7

conda activate regcn

pip install -r requirement.txt
```

### Process data
First, unzip and unpack the data files 
```
tar -zxvf data-release.tar.gz
```
For the three ICEWS datasets `ICEWS18`, `ICEWS14`, `ICEWS05-15`, go into the dataset folder in the `./data` directory and run the following command to construct the static graph.
```
cd ./data/<dataset>
python ent2word.py
```

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

0. Make dictionary to save models
```
mkdir models
```

1. Train models
```
cd src
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0
```

### Enhanced Training with Auxiliary Information Graphs (NEW)

本项目支持三种辅助信息图增强方案，可以显著提升时序知识图谱补全（TKGC）性能与可解释性。

#### 方案一：时间一致性破坏图（TIG - Temporal Inconsistency Graph）

检测实体在相邻时间片间的结构变化，用于调制时间门控机制。当存在 break edge 时，模型更倾向于遗忘历史状态。

```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --relation-prediction --use-tig --tig-threshold 0.5 --gpu 0
```

参数说明：
- `--use-tig`: 启用 TIG 模块
- `--tig-threshold`: break edge 检测阈值（Jaccard 距离），默认 0.5

#### 方案二：时间因果依赖图（TCDG - Temporal Causal Dependency Graph）

建模关系在时间上的因果触发链，对关系表示进行因果聚合增强。

```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --relation-prediction --use-tcdg --tcdg-max-delta 3 --gpu 0
```

参数说明：
- `--use-tcdg`: 启用 TCDG 模块
- `--tcdg-max-delta`: 因果聚合的最大时间差，默认 3

#### 方案三：时间约束图（TCG - Temporal Constraint Graph）

建模事实间的互斥约束，通过约束损失直接影响 embedding 学习。

```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --relation-prediction --use-tcg --tcg-margin 1.0 --tcg-weight 0.1 --gpu 0
```

参数说明：
- `--use-tcg`: 启用 TCG 模块
- `--tcg-margin`: 约束损失的 margin 值，默认 1.0
- `--tcg-weight`: 约束损失权重，默认 0.1

#### 组合使用所有增强方案

```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --relation-prediction --add-static-graph --use-tig --tig-threshold 0.5 --use-tcdg --tcdg-max-delta 3 --use-tcg --tcg-margin 1.0 --tcg-weight 0.1 --gpu 0
```

### Evaluate models
To generate the evaluation results of a pre-trained model, simply add the `--test` flag in the commands above. 

For example, the following command performs single-step inference and prints the evaluation results (with ground truth history).
```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test
```

The following command performs multi-step inference and prints the evaluation results (without ground truth history).
```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --multi-step --topk 0
```


### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other experiment set up according to Section 5.1.4 in the paper (https://arxiv.org/abs/2104.10353). 


## Citation
If you find the resource in this repository helpful, please cite
```
@article{li2021temporal,
  title={Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning},
  author={Li, Zixuan and Jin, Xiaolong and Li, Wei and Guan, Saiping and Guo, Jiafeng and Shen, Huawei and Wang, Yuanzhuo and Cheng, Xueqi},
  booktitle={SIGIR},
  year={2021}
}
```
