import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


class TemporalInconsistencyGraph:
    """
    时间一致性破坏图（Temporal Inconsistency Graph, TIG）
    用于检测实体在相邻时间片间的结构变化，生成 break edge
    """
    def __init__(self, num_ents, threshold=0.5):
        self.num_ents = num_ents
        self.threshold = threshold
        self.prev_entity_relations = None  # 存储上一时间片的实体关系集合
    
    def compute_break_edges(self, g_prev, g_curr, num_rels):
        """
        计算相邻时间片间的 break edges
        使用关系集合 Jaccard 距离来度量结构变化
        返回具有显著结构变化的实体索引列表
        """
        if g_prev is None:
            return torch.zeros(self.num_ents)
        
        # 获取当前和上一时间片的实体-关系集合
        prev_relations = self._get_entity_relations(g_prev, num_rels)
        curr_relations = self._get_entity_relations(g_curr, num_rels)
        
        break_scores = torch.zeros(self.num_ents)
        
        for ent in range(self.num_ents):
            prev_rels = prev_relations.get(ent, set())
            curr_rels = curr_relations.get(ent, set())
            
            if len(prev_rels) == 0 and len(curr_rels) == 0:
                continue
            
            # 计算 Jaccard 距离
            union = prev_rels | curr_rels
            intersection = prev_rels & curr_rels
            if len(union) > 0:
                jaccard_dist = 1 - len(intersection) / len(union)
                if jaccard_dist > self.threshold:
                    break_scores[ent] = jaccard_dist
        
        return break_scores
    
    def _get_entity_relations(self, g, num_rels):
        """获取每个实体的关系集合"""
        entity_relations = defaultdict(set)
        if hasattr(g, 'edata') and 'type' in g.edata:
            src, dst = g.edges()
            edge_types = g.edata['type']
            for s, d, r in zip(src.tolist(), dst.tolist(), edge_types.tolist()):
                entity_relations[s].add(r % num_rels)
                entity_relations[d].add(r % num_rels)
        return entity_relations


class TemporalCausalDependencyGraph(nn.Module):
    """
    时间因果依赖图（Temporal Causal Dependency Graph, TCDG）
    用于建模关系在时间上的因果触发链
    """
    def __init__(self, num_rels, h_dim, max_time_delta=3, top_k=10):
        super(TemporalCausalDependencyGraph, self).__init__()
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.max_time_delta = max_time_delta
        self.top_k = top_k
        
        # 可学习的因果强度矩阵（关系i在时间delta后触发关系j的强度）
        self.causal_weights = nn.Parameter(
            torch.zeros(max_time_delta, num_rels * 2, num_rels * 2)
        )
        nn.init.xavier_uniform_(self.causal_weights)
        
        # 注意力聚合层
        self.attention_layer = nn.Linear(h_dim, 1)
        
    def forward(self, rel_emb, history_rel_embs):
        """
        对关系表示进行因果聚合
        rel_emb: 当前时间片的关系表示 [num_rels*2, h_dim]
        history_rel_embs: 历史关系表示列表 [[num_rels*2, h_dim], ...]
        返回增强后的关系表示
        """
        if len(history_rel_embs) == 0:
            return rel_emb
        
        causal_aggregated = torch.zeros_like(rel_emb)
        
        for delta, hist_rel_emb in enumerate(reversed(history_rel_embs[-self.max_time_delta:])):
            if delta >= self.max_time_delta:
                break
            
            # 获取因果权重 [num_rels*2, num_rels*2]
            causal_weight = torch.softmax(self.causal_weights[delta], dim=0)
            
            # 因果聚合：每个关系从历史关系获取加权信息
            # [num_rels*2, num_rels*2] @ [num_rels*2, h_dim] -> [num_rels*2, h_dim]
            causal_info = torch.mm(causal_weight, hist_rel_emb)
            causal_aggregated = causal_aggregated + causal_info
        
        # 融合原始表示和因果聚合表示
        enhanced_rel_emb = rel_emb + causal_aggregated
        
        return enhanced_rel_emb


class TemporalConstraintGraph(nn.Module):
    """
    时间约束图（Temporal Constraint Graph, TCG）
    用于建模事实间的互斥约束和先后约束
    """
    def __init__(self, h_dim, margin=1.0):
        super(TemporalConstraintGraph, self).__init__()
        self.h_dim = h_dim
        self.margin = margin
        
        # 用于学习约束关系的投影层
        self.constraint_proj = nn.Linear(h_dim * 3, h_dim)
        
    def compute_constraint_loss(self, fact_embs, constraint_pairs):
        """
        计算约束损失
        fact_embs: 事实表示 [batch, h_dim*3]（head, rel, tail 拼接）
        constraint_pairs: 互斥约束对列表 [(fact_i_idx, fact_j_idx), ...]
        """
        if len(constraint_pairs) == 0:
            return torch.zeros(1, device=fact_embs.device)
        
        loss = torch.zeros(1, device=fact_embs.device)
        
        for i, j in constraint_pairs:
            if i < len(fact_embs) and j < len(fact_embs):
                fi = fact_embs[i]
                fj = fact_embs[j]
                # 互斥事实的表示应该尽量不同
                distance = torch.norm(fi - fj, p=2)
                loss = loss + torch.clamp(self.margin - distance, min=0)
        
        return loss / max(len(constraint_pairs), 1)
    
    def detect_mutex_constraints(self, triples, rel_mutex_pairs=None):
        """
        自动检测互斥约束对
        基于同一主语和同一关系类型但不同宾语的事实
        """
        if rel_mutex_pairs is None:
            rel_mutex_pairs = set()
        
        mutex_pairs = []
        
        # 按(主语, 关系)分组
        fact_groups = defaultdict(list)
        for idx, triple in enumerate(triples):
            h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
            fact_groups[(h, r)].append((idx, t))
        
        # 同一(主语,关系)下的不同宾语事实构成互斥
        for key, facts in fact_groups.items():
            if len(facts) > 1:
                for i in range(len(facts)):
                    for j in range(i + 1, len(facts)):
                        if facts[i][1] != facts[j][1]:  # 不同宾语
                            mutex_pairs.append((facts[i][0], facts[j][0]))
        
        return mutex_pairs


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False,
                 # 新增：三种辅助信息图增强参数
                 use_tig=False, tig_threshold=0.5,
                 use_tcdg=False, tcdg_max_delta=3,
                 use_tcg=False, tcg_margin=1.0, tcg_weight=0.1):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        # 新增：辅助信息图增强开关和参数
        self.use_tig = use_tig
        self.use_tcdg = use_tcdg
        self.use_tcg = use_tcg
        self.tcg_weight = tcg_weight

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError
        
        # ========== 方案一：时间一致性破坏图（TIG）==========
        if self.use_tig:
            self.tig = TemporalInconsistencyGraph(num_ents, threshold=tig_threshold)
            # TIG break embedding：用于调制时间门控
            self.break_emb = nn.Parameter(torch.Tensor(num_ents, h_dim))
            nn.init.zeros_(self.break_emb)
            # 融合 break signal 到时间门控的权重
            self.break_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.break_gate_weight, gain=nn.init.calculate_gain('sigmoid'))
        
        # ========== 方案二：时间因果依赖图（TCDG）==========
        if self.use_tcdg:
            self.tcdg = TemporalCausalDependencyGraph(
                num_rels=num_rels,
                h_dim=h_dim,
                max_time_delta=tcdg_max_delta
            )
            self.history_rel_embs = []  # 存储历史关系表示
        
        # ========== 方案三：时间约束图（TCG）==========
        if self.use_tcg:
            self.tcg = TemporalConstraintGraph(h_dim=h_dim, margin=tcg_margin) 

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []
        
        # TCDG: 清空历史关系表示列表（每次forward开始时重置）
        if self.use_tcdg:
            self.history_rel_embs = []
        
        prev_g = None  # TIG: 用于存储上一时间片的图

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            # ========== 方案二：TCDG 因果聚合 ==========
            if self.use_tcdg and len(self.history_rel_embs) > 0:
                # 对关系表示进行因果增强
                self.h_0 = self.tcdg(self.h_0, self.history_rel_embs)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            # 存储当前关系表示用于后续因果聚合
            if self.use_tcdg:
                self.history_rel_embs.append(self.h_0.detach().clone())
            
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            
            # ========== 方案一：TIG 时间门控调制 ==========
            if self.use_tig and prev_g is not None:
                # 计算 break scores
                break_scores = self.tig.compute_break_edges(prev_g, g, self.num_rels)
                if use_cuda:
                    break_scores = break_scores.cuda().to(self.gpu)
                
                # 生成 break embedding 调制信号
                # break_scores: [num_ents], break_emb: [num_ents, h_dim]
                break_signal = break_scores.unsqueeze(1) * self.break_emb  # [num_ents, h_dim]
                
                # 修改时间门控：当存在 break edge 时，倾向于遗忘历史状态
                time_weight = torch.sigmoid(
                    torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias +
                    torch.mm(break_signal, self.break_gate_weight)
                )
                # break_signal 强时，降低历史权重（更多使用当前状态）
                time_weight = time_weight * (1 - break_scores.unsqueeze(1) * 0.5)
            else:
                time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
            
            # TIG: 更新上一时间片的图
            if self.use_tig:
                prev_g = g
                
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_constraint = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        
        # ========== 方案三：TCG 约束损失 ==========
        if self.use_tcg:
            # 构建事实表示：head_emb + rel_emb + tail_emb
            head_embs = pre_emb[all_triples[:, 0]]  # [batch, h_dim]
            rel_embs = r_emb[all_triples[:, 1]]     # [batch, h_dim]
            tail_embs = pre_emb[all_triples[:, 2]]  # [batch, h_dim]
            fact_embs = torch.cat([head_embs, rel_embs, tail_embs], dim=1)  # [batch, h_dim*3]
            
            # 自动检测互斥约束对
            mutex_pairs = self.tcg.detect_mutex_constraints(all_triples)
            
            # 计算约束损失
            if len(mutex_pairs) > 0:
                loss_constraint = self.tcg.compute_constraint_loss(fact_embs, mutex_pairs)
                loss_constraint = self.tcg_weight * loss_constraint
        
        return loss_ent, loss_rel, loss_static, loss_constraint
