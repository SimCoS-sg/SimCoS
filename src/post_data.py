import json
import random
import jieba
import sys
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from conceptnet.search import *


jieba.load_userdict("./conceptnet/all_entity.txt")
for special_tokens in ["<extra_id_%d>"%k for k in range(100)]:
    # print(special_tokens)
    jieba.add_word(special_tokens)
# print(list(jieba.cut("<extra_id_0>")))

def load_file(filename, pred=False):
    data = []
    with open(filename, "r",encoding='utf-8') as f:
        for line in f.readlines():
            if pred:
                data.append({"label": line.strip()})
            else:
                data.append(json.loads(line))
        f.close()
    return data

def write_txt_file_mask_storyline(filename,data, mask_p = 0.15):
    with open(filename+".source", "w+",encoding='utf-8') as fs:
        with open(filename+".target", "w+",encoding='utf-8') as ft:
            for i in data:
                str_title = i['title']
                for index in range(len(i['outline'])):
                    if random.random() < mask_p:
                        str_storyline = i['outline'][0:index] + ['<mask>'] + i['outline'][index + 1:]
                        fs.write(str_title+"#"+"#".join(str_storyline)+"\n")
                        ft.write(i['outline'][index]+"\n")
                    
def write_txt_file_mask_storyline_with_entity_label(filename, data, mask_p = 0.15):
    with open("./conceptnet/all_entity.txt", 'r') as fr:
        all_e = [word.strip() for word in fr.readlines()]
    with open(filename+".source", "w",encoding='utf-8') as fs, open(filename+".target", "w",encoding='utf-8') as ft:
        for i in tqdm(data):
            str_title = i['title']
            for index in range(len(i['outline'])):
                if random.random() < mask_p:
                    storyline = i['outline'][0:index] + ['<mask>'] + i['outline'][index + 1:]
                    str_storyline = str_title+"#"+"#".join(storyline)+"\n"
# label entities in storyline
#                     words = jieba.cut(str_storyline)
#                     str_storyline = ''.join([word if word not in all_e else "<entity>" + word + "</entity>" for word in words])
                    fs.write(str_storyline)
                    str_target = i['outline'][index]+"\n"
# label entities in target
                    words = jieba.cut(str_target)
                    str_target = ''.join([word if word not in all_e else "<entity>" + word + "</entity>" for word in words])
                    ft.write(str_target)                
                
def save_distribution(data, pic_name):
    plt.clf()
    plt.hist(data, bins=range(min(data), max(data)+2), align='left', rwidth=0.8)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of List')
    # plt.xticks(range(min(data), max(data)+1))
    plt.savefig(pic_name+'.png')

def analyse_route(data):
    with open("./conceptnet/all_entity.txt", 'r') as fr:
        all_e = [word.strip() for word in fr.readlines()]
    concept_net = loadConceptnet('/home/cis/zhu/LOT-LongLM/longlm/KEPLM/conceptnet/chineseconceptnet.csv')
    start = [cht2chs(strip(word).replace('_',''))for word in concept_net['start'].tolist()]
    end = [cht2chs(strip(word).replace('_','')) for word in concept_net['end'].tolist()]
    # e_str2id = {}
    # for i in len(all_e):
    #     e_str2id[all_e[i]] = i
    G = nx.Graph()
    G.add_nodes_from(all_e)
    for i in range(len(start)):
        G.add_edge(start[i], end[i], relation=concept_net['relation'][i])
    analyse_result = {}
    outline_entities_num = []
    story_entities_num = []
    one_hop_entities_num = []
    one_hop_result_num = []
    two_hop_entities_num = []
    two_hop_result_num = []
    subgraph_hop_num = []

    for i in tqdm(data):
        str_title = i['title']
        str_outline = str_title+"#"+"#".join(i['outline'])+"\n"
        str_story = i['story']
        words_outline = jieba.cut(str_outline)
        words_story = jieba.cut(str_story)
        entities_outline = list(set(all_e).intersection(set(words_outline)))
        entities_story = list(set(all_e).intersection(set(words_story)))

        shortest_paths_length_list = []
        for story_node in entities_story:
            shortest_path_length = sys.maxsize
            for outline_node in entities_outline:
                try:
                    shortest_path_length = min(shortest_path_length, nx.shortest_path_length(G, source=outline_node, target=story_node))
                except nx.NetworkXNoPath:
                    print('No path')
            # print(shortest_path_length)
            if(shortest_path_length < sys.maxsize and shortest_path_length > 0):
                shortest_paths_length_list.append(int(shortest_path_length))
        subgraph_hop_num.append(int(sum(shortest_paths_length_list)/len(shortest_paths_length_list)))

        one_hop_entities = []
        for node in entities_outline:
            one_hop_entities.extend(list(G.neighbors(node)))
        one_hop_set = set(one_hop_entities)

        story_set = set(entities_story)
        outline_set = set(entities_outline)

        one_hop_result_set = one_hop_set.intersection(story_set).difference(outline_set)

        two_hop_entities = []
        for node in one_hop_set:
            two_hop_entities.extend(list(G.neighbors(node)))

        two_hop_set = set(one_hop_entities + two_hop_entities)
        two_hop_result_set = two_hop_set.intersection(story_set).difference(outline_set)

        outline_entities_num.append(len(entities_outline))
        story_entities_num.append(len(entities_story))
        one_hop_entities_num.append(len(one_hop_entities))
        one_hop_result_num.append(len(one_hop_result_set)) 
        two_hop_entities_num.append(len(two_hop_entities))
        two_hop_result_num.append(len(two_hop_result_set))

    analyse_result['avg_outline_entities_num'] = sum(outline_entities_num)/len(outline_entities_num)
    analyse_result['avg_story_entities_num'] = sum(story_entities_num)/len(story_entities_num)
    analyse_result['avg_one_hop_entities_num'] = sum(one_hop_entities_num)/len(one_hop_entities_num)
    analyse_result['avg_one_hop_result_num'] = sum(one_hop_result_num)/len(one_hop_result_num)
    analyse_result['avg_two_hop_entities_num'] = sum(two_hop_entities_num)/len(two_hop_entities_num)
    analyse_result['avg_two_hop_result_num'] = sum(two_hop_result_num)/len(two_hop_result_num)
    analyse_result['avg_subgraph_hop_num'] = sum(subgraph_hop_num)/len(subgraph_hop_num)

    save_distribution(outline_entities_num, 'outline_entities_num')
    save_distribution(story_entities_num, 'story_entities_num')
    save_distribution(one_hop_entities_num, 'one_hop_entities_num')
    save_distribution(one_hop_result_num, 'one_hop_result_num')
    save_distribution(two_hop_entities_num, 'two_hop_entities_num')
    save_distribution(two_hop_result_num, 'two_hop_result_num')
    save_distribution(subgraph_hop_num, 'subgraph_hop_num')

    print(analyse_result)


def entityID_dataset(input_file, output_file, processed_col = "src"):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    with open("./conceptnet/all_entity.txt", 'r') as fr:
        Id2entity = [word.strip() for word in fr.readlines()]
    entity2Id  = {Id2entity[i]: i for i in range(len(Id2entity))}
    for item in data:
        words_src = jieba.cut(item["src"])
        # print(list(words_src))
        entities_outline = list(set(Id2entity).intersection(set(words_src)))
        item['eid'] = [entity2Id[entity] for entity in entities_outline]

    with open(output_file, 'w', encoding='utf-8') as output:
        for item in data:
            json.dump(item, output, ensure_ascii=False)
            output.write('\n')  # 写入一个换行符，使每个 JSON 对象独占一行
    



if __name__ == '__main__':

    # data = load_file("./train.jsonl")
    # analyse_route(data)
    # read the data from json file

#     write_txt_file_target_text("./result_train.txt",data)
    # convert data to txt file 

#     trans_txt2json_file("result.txt", "train.jsonl")
#     # extract the story from jsonl file and save as txt file 

#    mask one of storyline
    # write_txt_file_mask_storyline_with_entity_label("./post_train",data)
    
    # save the data to source file 
#     write_txt_file_target("./post_train.target",data)
#     save the data to target file 
