import json
import argparse
from os import pread
import sys
import numpy as np
import jieba
import nltk
import torch
import dgl

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
from rouge import Rouge

def bleu(data):
    """
    compute rouge score
    Args:
        data (list of dict including reference and candidate):
    Returns:
            res (dict of list of scores): rouge score
    """

    res = {}
    for i in range(1, 5):
        res["sentence-bleu-%d"%i] = []
        res["corpus-bleu-%d"%i] = nltk.translate.bleu_score.corpus_bleu([[d["reference"].strip().split()] for d in data], [d["candidate"].strip().split() for d in data], weights=tuple([1./i for j in range(i)]))
    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        for i in range(1, 5):
            res["sentence-bleu-%d"%i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference], hypothesis=origin_candidate.strip().split(), weights=tuple([1./i for j in range(i)]))) 

    for key in res:
        if "sentence" in key:
            res[key] = np.mean(res[key])
        
    return res



def repetition_distinct(eval_data):
    result = {}
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, tmp_data in enumerate(eval_data):
            ngs = ["_".join(c) for c in ngrams(tmp_data["candidate"].strip().split(), i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    return result


def rouge(ipt, cand):
    rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
    item_name = ["f", "p", "r"]

    res = {}
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = []
    for k, (tmp_ipt, tmp_cand) in enumerate(zip(ipt, cand)):
        for tmp_ref in tmp_ipt.split("#"):
            # print(tmp_ref.strip())
            # print(" ".join(tmp_cand))

            # tmp_ref = tmp_ref.strip()
            # tmp_hyp = " ".join(tmp_cand).strip()

            tmp_ref = " ".join([w for w in "".join(tmp_ref.strip().split())])
            tmp_hyp = " ".join([w for w in "".join(tmp_cand.strip().split())])
            # print(tmp_ref)
            # print(tmp_hyp)
            try:
                tmp_res = Rouge().get_scores(refs=tmp_ref, hyps=tmp_hyp)[0]
                for name1 in rouge_name:
                    for name2 in item_name:
                        res["%s-%s"%(name1, name2)].append(tmp_res[name1][name2])
            except:
                continue
    for name1 in rouge_name:
        for name2 in item_name:                
            res["%s-%s"%(name1, name2)] = np.mean(res["%s-%s"%(name1, name2)])
    return {"coverage": res["rouge-l-r"]}


def LCS(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def Recon_LCS(x, y, exclusive=True):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = LCS(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    if len(recon_list):
        return "".join(recon_list).strip()
    else:
        return ""
    # return Ngrams(recon_list, exclusive=exclusive)
    # return recon_tuple


def lcs3_dp(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * (len(input_y)+1)) for i in range(len(input_x)+1)]
    maxlen = maxindex = 0
    for i in range(1, len(input_x)+1):
        for j in range(1, len(input_y)+1):
            if i == 0 or j == 0:  # 在边界上，自行+1
                    dp[i][j] = 0
            if input_x[i-1] == input_y[j-1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > maxlen:  # 随时更新最长长度和长度开始的位置
                    maxlen = dp[i][j]
                    maxindex = i - maxlen
                    # print('最长公共子串的长度是:%s' % maxlen)
                    # print('最长公共子串是:%s' % input_x[maxindex:maxindex + maxlen])
            else:
                dp[i][j] = 0
    # for dp_line in dp:
    #     print(dp_line)
    return input_x[maxindex:maxindex + maxlen]

def inversenum(a):
    num = 0
    all_num = 0
    for i in range(0,len(a)):
        for j in range(i,len(a)):
            if a[i] > a[j]:
                num += 1
            all_num += 1
    return num / float(all_num)

def find_all(sub,s):
	index_list = []
	index = s.find(sub)
	while index != -1:
		index_list.append(index)
		index = s.find(sub,index+1)
	
	if len(index_list) > 0:
		return index_list
	else:
		return -1

def order(ipt, cand, kw2id):
    num = []
    for k, (tmp_ipt, tmp_cand, tmp_kw2id) in enumerate(zip(ipt, cand, kw2id)):
        # all_pos = [[]]
        pos = []
        kw_list = list(tmp_kw2id.keys())
        kw_list.reverse()

        for tmp_ref in kw_list:
            tmp_ref = "".join(tmp_ref.strip().split())
            tmp_hyp = "".join(tmp_cand.strip().split())
            lcs = lcs3_dp(tmp_ref, tmp_hyp)
            if len(lcs)>1:
                pos.append(tmp_hyp.find(lcs))
            else:
                pos.append(-1)
        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])

        new_rank = [-1 for _ in idlist]
        for idl, ord in zip(idlist, orderlist):
            new_rank[idl] = tmp_kw2id[kw_list[ord]]
        num.append(1-inversenum(new_rank))

    return {"order": np.mean(num)}



def load_file(filename, pred=False):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if pred:
                data.append({"story": line.strip()})
            else:
                data.append(json.loads(line))
        f.close()
    return data

def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])


def compute(predictions, decoded_labels, return_dict=True):
    # golden_data = load_file(golden_file)
    # pred_data = load_file(pred_file, pred=True)

    # if len(golden_data) != len(pred_data):
    #     raise RuntimeError("Wrong Predictions")

    # ipt = ["#".join(g["outline"]) for g in golden_data]
    # truth = [g["story"] for g in golden_data]
    # pred = [p["story"] for p in pred_data]

    # kw2id = []
    # for i1, t1 in zip(ipt, truth):
    #     kw_list = i1.strip().split("#")
    #     pos = [t1.strip().find(kw.strip()) for kw in kw_list]

    #     idlist = list(range(len(pos)))
    #     orderlist = sorted(idlist, key=lambda x: pos[x])
    #     kw2id.append({})
    #     for idl, ord in zip(idlist, orderlist):
    #         kw2id[-1][kw_list[ord]] = idl


    eval_data = [{"reference": proline(g), "candidate": proline(p)} for g, p in zip(decoded_labels, predictions)]
    res = bleu(eval_data)
    res.update(repetition_distinct(eval_data))
    # res.update(rouge(ipt=ipt, cand=pred))
    # res.update(order(ipt=ipt, cand=pred, kw2id=kw2id))
    
    # for key in res:
    #     res[key] = "_"
    return res

with open("./conceptnet/all_entity.txt", 'r') as fr:
    Id2entity = [word.strip() for word in fr.readlines()]
Id2entity = ['<PAD_entity>'] + Id2entity
entity2Id  = {Id2entity[i]: i for i in range(len(Id2entity))}
with open("./conceptnet/relations.txt", 'r') as fr:
    Id2relation = [word.strip() for word in fr.readlines()]
relation2Id  = {Id2relation[i]: i for i in range(len(Id2relation))}

# 初始化一个空列表来存储读取的三元组
triples = []
with open('./conceptnet/chinese_triples.txt', 'r') as file:
    # 逐行读取文件内容
    lines = file.readlines()
    for line in lines:
        # 使用制表符分割每行内容，得到三个元素，并将它们转换为整数
        elements = line.strip().split('\t')
        triples.append((elements[0], elements[1], elements[2]))
# print(len(triples))
def get_entityID(input, max_length, padding_token):
    graph_input = []
    # print(len(input))
    for item in input:
        words_src = jieba.cut(item)
        entities_outline = list(set(Id2entity).intersection(set(words_src)))
        entities_ids = [entity2Id[entity] for entity in entities_outline]
        # print(len(entities_ids))
        
        # Padding or Truncating
        if len(entities_ids) < max_length:
            entities_ids = entities_ids + [padding_token] * (max_length - len(entities_ids))
        elif len(entities_ids) > max_length:
            entities_ids = entities_ids[:max_length]
        # print(len(entities_ids))
        graph_input.append(entities_ids)
    # print(len(graph_input))
    return graph_input

def bulid_KG():
    num_rels = len(Id2relation)
    # print(len(triples))
    # print(self._num_nodes)
    h = torch.tensor([entity2Id[i[0]] for i in triples])
    r = torch.tensor([relation2Id[i[1]] for i in triples])
    t = torch.tensor([entity2Id[i[2]] for i in triples])
    kg = dgl.graph((h, t))
    # print(kg)
    kg = dgl.add_reverse_edges(kg, ignore_bipartite=True, exclude_self=False)
    # print(kg)
    rev_r = r + num_rels
    kg.edata['relation'] = torch.cat((r, rev_r), dim=0)
    # print(kg)
    return kg

def pad_list(input, max_len, padding_idx=0):
    input = pad_sequence(input).T
    input = F.pad(input, (0, max_len - input.size(1)), value=padding_idx)
    return input

def get_sub_graph(kg, eids, padding_idx=0):
    max_len = 160
    subgraphs = []
    eids = torch.tensor(eids)
    for eid in eids:
        non_zero_indices = (eid != padding_idx).nonzero().squeeze()
        eid_without_pad = eid[non_zero_indices]
        subgraphs.append(sample_2hop(kg, eid_without_pad, 2))
    head = [g.adj_tensors('coo')[0] for g in subgraphs]
    edge_size = torch.tensor([s.size(0) for s in head])
    # print("head:", head)
    tail = [g.adj_tensors('coo')[1] for g in subgraphs]
    # print("tail:", head)
    relation = [g.edata['relation'] for g in subgraphs]
    # print("relation:", head)
    orign_ID = [g.ndata['_ID'] for g in subgraphs]
    id_size = torch.tensor([s.size(0) for s in orign_ID])
    # print("orign_ID:", head)
    return pad_list(head, 160), edge_size, pad_list(tail, 160), pad_list(relation, 160), pad_list(orign_ID, 160), id_size

def sample_2hop(g, center_node, num_neighbors):
    # 采样一跳邻居
    one_hop_subgraph = dgl.sampling.sample_neighbors(g, center_node, num_neighbors)
    one_hop_subgraph = dgl.edge_subgraph(g, one_hop_subgraph.edata['_ID'])
    two_hop_subgraph = dgl.sampling.sample_neighbors(g, one_hop_subgraph.ndata['_ID'], num_neighbors)
    two_hop_subgraph = dgl.edge_subgraph(g, two_hop_subgraph.edata['_ID'])
    return two_hop_subgraph


# def get_sub_graph(kg, eids, padding_idx=0):
#     last_non_padding_index = np.array([np.where(row != padding_idx)[0][-1] + 1 if np.any(row != padding_idx) else 0 for row in eids])
#     subgraphs = [dgl.khop_in_subgraph(kg, eid[:index], 2)[0] for index, eid in zip(last_non_padding_index, eids)]
#     return subgraphs
