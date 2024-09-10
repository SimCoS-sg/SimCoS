import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from torch.nn.utils.rnn import pad_sequence
from utils import bulid_KG, get_sub_graph

class RGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        regularizer=None,
        num_bases=-1,
        dropout=0.0,
        self_loop=False,
        ns_mode=False,
    ):
        super(RGCN, self).__init__()

        if num_bases == -1:
            num_bases = num_rels
        self.emb = nn.Embedding(num_nodes, h_dim, padding_idx=0)
        self.conv1 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer, num_bases, self_loop=self_loop
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer,
            num_bases,
            self_loop=self_loop,
        )
        self.dropout = nn.Dropout(dropout)
        self.ns_mode = ns_mode

    def forward(self, g):
        nids = g.ndata[dgl.NID]
        x = self.emb(nids)
        h = self.conv1(g, x, g.edata['relation'])
        h = self.dropout(F.relu(h))
        h = self.conv2(g, h, g.edata['relation'])
        # print(h.shape)
        return h 


class MyModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # self._kg = bulid_KG()
        self._logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._rgcn = RGCN(150112 + 1, 768, 768, 2 * 24)


    # def _get_sub_graph(self, eids, padding_idx=0):
    #     subgraphs = []
    #     # print(eids.device)
    #     kg = self._kg.to(eids.device)
    #     # print(kg.device)
    #     for eid in eids:
    #         non_zero_indices = (eid != padding_idx).nonzero().squeeze()
    #         eid_without_pad = eid[non_zero_indices]
    #         # print(eid_without_pad.device)
    #         subgraphs.append(self._sample_2hop(kg, eid_without_pad, 2))
    #     del kg
    #     return subgraphs

    # def _sample_2hop(self, g, center_node, num_neighbors):
    #     # 采样一跳邻居
    #     one_hop_subgraph = dgl.sampling.sample_neighbors(g, center_node, num_neighbors)
    #     one_hop_subgraph = dgl.edge_subgraph(g, one_hop_subgraph.edata['_ID'])
    #     two_hop_subgraph = dgl.sampling.sample_neighbors(g, one_hop_subgraph.ndata['_ID'], num_neighbors)
    #     two_hop_subgraph = dgl.edge_subgraph(g, two_hop_subgraph.edata['_ID'])
    #     return two_hop_subgraph

    def _get_sub_graph(self, head = None, edge_size = None, tail = None, relation = None, orign_ID = None, id_size = None):
        subgraphs = []
        # print(head, edge_size, tail, relation, orign_ID)
        for h, edge_num, t, r, id, id_num in zip(head, edge_size, tail, relation, orign_ID, id_size):
            # print(h, edge_num, t, r, id)
            subgraph = dgl.DGLGraph((h[:edge_num], t[:edge_num]))
            subgraph.edata['relation'] = r[:edge_num]
            subgraph.ndata['_ID'] = id[:id_num]
            subgraphs.append(subgraph)
        return subgraphs
    
    def _graph_encoder(self, subgraphs):
        rgcn_output = torch.stack([self._rgcn(subgraph).mean(dim=0) for subgraph in subgraphs], dim=0)
        return rgcn_output
        
    def _contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def _kl_loss(self, similarity, graph_similarity):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        # input should be a distribution in the log space
        input = F.log_softmax(similarity, dim=1)
        # Sample a batch of distributions. Usually this would come from the dataset
        target = F.softmax(graph_similarity * 10, dim=1)
        # print(input, target)
        return kl_loss(input, target) * 4
    
    def _weigh_loss(self, similarity, graph_similarity):
        caption_loss = self._kl_loss(similarity, graph_similarity)
        image_loss = self._kl_loss(similarity.t(), graph_similarity)
        return (caption_loss + image_loss) / 2.0
    
    def _clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self._contrastive_loss(similarity)
        image_loss = self._contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    
    # def _
    
    def _graph_similarity(self, eids, orign_ID):
        """
        计算多个图之间的相似度矩阵

        参数:
        - eids: 包含多个图的张量，每一行表示一个图，0表示padding

        返回:
        - similarity_matrix: 图之间的相似度矩阵
        """
        # 将张量转换为布尔张量，表示节点的存在
        
        # 计算Jaccard相似度
        def jaccard_similarity(graph1, graph2):
            non_zero_indices_1 = (graph1 != 0).nonzero().squeeze()
            non_zero_indices_2 = (graph2 != 0).nonzero().squeeze()
            combined = torch.cat((graph1[non_zero_indices_1], graph2[non_zero_indices_2]))
            union, counts = combined.unique(return_counts=True)
            intersection = union[counts > 1]
            return torch.numel(intersection) / torch.numel(union)

        # 计算图之间的相似度矩阵
        num_graphs = eids.size(0)
        eids_ = []
        for i in range(num_graphs):
            tensor_is_not_in = torch.isin(orign_ID[i], eids[i],invert=True)
            eids_.append(orign_ID[i][tensor_is_not_in])
        eids_ = pad_sequence(eids_, batch_first=True)
        # print(orign_ID)
        # print(eids_)
        # print(eids)
        similarity_matrix = torch.eye(eids.size(0), device=eids.device)
        alpha = 1
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                sim_1 = jaccard_similarity(eids[i], eids[j])
                sim_2 = jaccard_similarity(eids[i], eids_[j])
                sim_3 = jaccard_similarity(eids_[i], eids[j])
                sim_4 = jaccard_similarity(eids_[i], eids_[j])
                sim = alpha * sim_1 + (1 - alpha) * max(sim_2, sim_3, sim_4)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        return similarity_matrix
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        eids = None,
        head = None, edge_size = None, tail = None, relation = None, orign_ID = None, id_size = None
    ):
        
        transformer_outputs = super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask,
        decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        encoder_outputs=encoder_outputs,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
        # print(input_ids)
        # print()
        if eids is not None:
            # print(eid)
            
            # print(graph_similarity)
            # (4, 40, 768)
            # kg_e = self._rgcn.emb(eids)
            subgraphs = self._get_sub_graph(head, edge_size, tail, relation, orign_ID, id_size)
            # print(subgraphs)
            kg_features = self._graph_encoder(subgraphs)
            # print(kg_features.shape)
            # (4, 768)
            # kg_features = kg_e.mean(dim=1)
            # print(transformer_outputs.encoder_last_hidden_state.shape)
            # text_features = transformer_outputs.encoder_last_hidden_state[:, 0, :]

            text_features = transformer_outputs.encoder_last_hidden_state.mean(dim=1)

            kg_features = kg_features / kg_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit_scale = self._logit_scale.exp()

            logits_per_text = torch.matmul(text_features, kg_features.t()) * logit_scale
            logits_per_kg = logits_per_text.t()
            graph_similarity = self._graph_similarity(eids, orign_ID)
            # print(graph_similarity)
            contrastive_loss = None
            # print(cls_outputs.shape)
            # contrastive_loss = self._clip_loss(logits_per_text)
            # contrastive_loss = self._clip_loss(logits_per_text)
            # print(contrastive_loss)
            contrastive_loss = self._weigh_loss(logits_per_text, graph_similarity)
            # print(contrastive_loss)
            transformer_outputs.loss  = contrastive_loss + transformer_outputs.loss
            # print(contrastive_loss)
            # print(lm_loss)
        # print(eids)
        
        return transformer_outputs







    