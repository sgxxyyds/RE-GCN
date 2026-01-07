import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

# from hisres_rgcn.layers import RGCNBlockLayer as RGCNLayer
from hisres_rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, CandRGCNLayer
from hisres_src.model import BaseRGCN
from hisres_src.decoder import *
from hisres_rgcn.utils import build_his_graph

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
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            # g.edata['r'] = init_rel_emb
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                x, r = layer(g, [], r)
            return g.ndata.pop('h'), r
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

class CandRGCN(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return CandRGCNLayer(self.h_dim, self.h_dim, self.num_rels, 100,
                            activation=F.rrelu, dropout=0.2, self_loop=True, skip_connect=False)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            g.edata['r'] = init_rel_emb[g.edata['type']]
            # g.edata['t'] = init_time_emb
            # x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], [])
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
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, timestamps=0):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
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
        self.sin = torch.sin
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.timestamps = num_times

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.r_linear = nn.Linear(2*h_dim, h_dim)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.time_emb = torch.nn.Parameter(torch.Tensor(self.timestamps, 32), requires_grad=True).float()
        torch.nn.init.normal_(self.time_emb)
                     
        self.time_linear = nn.Linear(2 * h_dim, h_dim)

        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.linear_pred_layer_s1 = nn.Linear(h_dim, h_dim)
        self.linear_g = nn.Linear(h_dim, h_dim)
        self.linear_pred_layer_o1 = nn.Linear(h_dim, h_dim)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.loss_sim = torch.nn.CosineEmbeddingLoss(margin=0.2)
        self.cand_layer_raw = CandRGCN(num_ents,
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

        self.cand_layer_inv = CandRGCN(num_ents,
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

        self.rgcn_2 = RGCNCell(num_ents,
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

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(2 * self.h_dim, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell_2 = nn.GRUCell(2 * self.h_dim, self.h_dim)
        self.entity_cell_2 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob_raw = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_inv = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)

            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def forward(self, g_list, gg_list, ggg_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
            self.h_gg = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            self.h_gg = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = self.h

        history_embs = []
        history_embs_gg = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            #* add time encoding like LogCL
            t2 = len(g_list)-i+1
            h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
            self.h =self.time_linear(torch.concat([self.h,h_t],dim=1))

            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                self.h_0 = self.r_linear(torch.cat((self.emb_rel, x_input), dim=1))
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h, current_r = self.rgcn.forward(g, self.h, self.h_0)
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            current_r = torch.cat((current_r, x_input), dim=1)
            self.h_0 = self.relation_cell_1(current_r, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            history_embs.append(self.h)

        for i, g in enumerate(gg_list):
            g = g.to(self.gpu)
            temp_e = self.h_gg[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                self.h_0_gg = self.r_linear(torch.cat((self.emb_rel, x_input), dim=1))
                self.h_0_gg = F.normalize(self.h_0_gg) if self.layer_norm else self.h_0_gg
            current_h, current_r = self.rgcn_2.forward(g, self.h_gg, self.h_0_gg)
            self.h_gg = self.entity_cell_2(current_h, self.h_gg)
            self.h_gg = F.normalize(self.h_gg) if self.layer_norm else self.h_gg
            current_r = torch.cat((current_r, x_input), dim=1)
            self.h_0_gg = self.relation_cell_2(current_r, self.h_0_gg)
            self.h_0_gg = F.normalize(self.h_0_gg) if self.layer_norm else self.h_0_gg
            history_embs_gg.append(self.h_gg)

        return history_embs, static_emb, self.h_0, gate_list, degree_list, history_embs_gg

    def get_loss(self, glist, gglist, ggglist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_spc = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        #* list the triples and the inverse triples
        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        triples = triples.to(self.gpu)
        inverse_triples = inverse_triples.to(self.gpu)
        evolve_embs, static_emb, r_emb, _, _, evolve_embs_gg = self.forward(glist, gglist, ggglist, static_graph, use_cuda)
        g_emb = evolve_embs[-1]
        gg_emb = evolve_embs_gg[-1]
        weight_g = torch.sigmoid(self.linear_g(g_emb))
        pre_emb = weight_g * g_emb + (1 - weight_g) * gg_emb
        #* list the raw_voc and inv_voc
        raw_voc, inv_voc = torch.chunk(entity_history_vocabulary, 2, 0)[0], torch.chunk(entity_history_vocabulary, 2, 0)[1]
        
        time_embs = 0
        #* raw
        his_graph_raw = build_his_graph(self.num_ents, self.num_rels, triples, raw_voc, self.gpu).to(self.gpu)
        emb_raw = F.normalize(self.cand_layer_raw(his_graph_raw, pre_emb, r_emb))
        #* inverse
        his_graph_inv = build_his_graph(self.num_ents, self.num_rels, inverse_triples, inv_voc, self.gpu, True).to(self.gpu)
        emb_inv = F.normalize(self.cand_layer_inv(his_graph_inv, pre_emb, r_emb))

        weight_s = torch.sigmoid(self.linear_pred_layer_s1(emb_raw))
        weight_o = torch.sigmoid(self.linear_pred_layer_o1(emb_inv))

        final_emb_raw = weight_s * emb_raw + (1 - weight_s) * pre_emb# + time_embs
        final_emb_inv = weight_o * emb_inv + (1 - weight_o) * pre_emb# + time_embs

        if self.entity_prediction:
            preds, _ = self.history_mode(final_emb_raw, r_emb, time_embs, triples)
            predo, _ = self.history_mode(final_emb_inv, r_emb, time_embs, inverse_triples, True)

            loss_ent += 0.5 * self.loss_e(preds, triples[:, 2]) + 0.5 * self.loss_e(predo, inverse_triples[:, 2])

        if self.relation_prediction:
            all_triples = torch.cat([triples, inverse_triples])
            all_triples = all_triples.to(self.gpu)
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            loss_rel += self.loss_r(score_re, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    # angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
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

        return loss_ent, loss_rel, loss_static, loss_spc


    def predict(self, glist, gglist, ggglist, num_rels, static_graph, triples, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_triples = triples[:, [2, 1, 0, 3]]
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat((triples, inverse_triples)).to(self.gpu)

            evolve_embs, _, r_emb, _, _, evolve_embs_gg = self.forward(glist, gglist, ggglist, static_graph, use_cuda)
            # pre_emb = evolve_embs[-1]
            g_emb = evolve_embs[-1]
            gg_emb = evolve_embs_gg[-1]
            weight_g = torch.sigmoid(self.linear_g(g_emb))

            pre_emb = weight_g * g_emb + (1 - weight_g) * gg_emb
            #* list the raw_voc and inv_voc
            raw_voc, inv_voc = torch.chunk(entity_history_vocabulary, 2, 0)[0], torch.chunk(entity_history_vocabulary, 2, 0)[1]

            time_embs = 0
            #* raw
            his_graph_raw = build_his_graph(self.num_ents, self.num_rels, triples, raw_voc, self.gpu).to(self.gpu)
            emb_raw = F.normalize(self.cand_layer_raw(his_graph_raw, pre_emb, r_emb))
            #* inverse
            his_graph_inv = build_his_graph(self.num_ents, self.num_rels, inverse_triples, inv_voc, self.gpu, True).to(self.gpu)
            emb_inv = F.normalize(self.cand_layer_inv(his_graph_inv, pre_emb, r_emb))

            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, all_triples, rel_history_vocabulary)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r

            weight_s = torch.sigmoid(self.linear_pred_layer_s1(emb_raw))
            weight_o = torch.sigmoid(self.linear_pred_layer_o1(emb_inv))

            final_emb_raw = weight_s * emb_raw + (1 - weight_s) * pre_emb# + time_embs
            final_emb_inv = weight_o * emb_inv + (1 - weight_o) * pre_emb# + time_embs
            preds, _ = self.history_mode(final_emb_raw, r_emb, time_embs, triples)
            predo, _ = self.history_mode(final_emb_inv, r_emb, time_embs, inverse_triples, True)

            score = torch.cat([preds, predo], dim=0)

            return all_triples, score, score_rel

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, inv=False):
        if inv:
            return self.decoder_ob_inv.forward(pre_emb, r_emb, time_embs, all_triples)
        else:
            return self.decoder_ob_raw.forward(pre_emb, r_emb, time_embs, all_triples)

    def rel_raw_mode(self, pre_emb, r_emb, all_triples):
        return self.rdecoder_re1.forward(pre_emb, r_emb, all_triples).view(-1, 2 * self.num_rels)

    def rel_history_mode(self, pre_emb, r_emb, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        return self.rdecoder_re2.forward(pre_emb, r_emb, all_triples, partial_embeding=global_index)







