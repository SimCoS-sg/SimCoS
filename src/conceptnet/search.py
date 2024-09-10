import pandas as pd
import json
import opencc

# relation template
template = {
    '/r/RelatedTo': '和{}相关',
    '/r/FormOf': '的形式为{}',
    '/r/IsA': '是{}',
    '/r/PartOf': '是{}的一部分',
    '/r/HasA': '具有{}',
    '/r/UsedFor': '用来{}',
    '/r/CapableOf': '可以{}',
    '/r/AtLocation': '在{}',
    '/r/Causes': '导致{}',
    '/r/HasSubevent': ',接下来,{}',
    '/r/HasFirstSubevent': '，紧接着，{}',
    '/r/HasLastSubevent': '的最后一步是{}',
    '/r/HasPrerequisite': '的前提为{}',
    '/r/HasProperty': '具有{}的属性',
    '/r/MotivatedByGoal': '受到{}的驱动',
    '/r/ObstructedBy': '受到{}的影响',
    '/r/Desires': '想要{}',
    '/r/CreatedBy': '被{}创造',
    '/r/Synonym': '和{}同义',
    '/r/Antonym': '和{}反义',
    '/r/DistinctFrom': '和{}相区别',
    '/r/DerivedFrom': '由{}导致',
    '/r/SymbolOf': '象征着{}',
    '/r/DefinedAs': '定义为{}',
    '/r/MannerOf': '',
    '/r/LocatedNear': '和{}相邻',
    '/r/HasContext': '的背景是{}',
    '/r/SimilarTo': '和{}相似',
    '/r/EtymologicallyRelatedTo': '',
    '/r/EtymologicallyDerivedFrom': '',
    '/r/CausesDesire': '',
    '/r/MadeOf': '由{}制成',
    '/r/ReceivesAction': '',
    '/r/ExternalURL': ''
}

def loadConceptnet(file = 'chineseconceptnet.csv'):
    print('load conceptnet data')
    data = pd.read_csv(file, delimiter='\t')
    data.columns = ['uri', 'relation', 'start', 'end', 'json']
    print('delete entity')
    data = data[data['start'].apply(lambda row: row.find('zh') > 0) & data['end'].apply(lambda row: row.find('zh') > 0)]
    data.index = range(data.shape[0])
    weights = data['json'].apply(lambda row: json.loads(row)['weight'])
    data.pop('json')
    data.insert(4, 'weights', weights)
    return data

t2s = opencc.OpenCC('t2s.json')
s2t = opencc.OpenCC('s2t.json')
# 繁体转简体
def cht2chs(line):
    return t2s.convert(line)
# 简体转繁体
def chs2cht(line):
    return s2t.convert(line)

def search(data, words, n=20):
    result = data[data['start'].str.contains(chs2cht(words))]
    topK_result = result.sort_values("weights", ascending=False).head(n)
    return topK_result

def getEntity(data):
    start = [cht2chs(strip(word).replace('_',''))for word in data['start'].tolist()]
    start = list(set(start))
    print('start entity size: ')
    print(len(start))
    end = [cht2chs(strip(word).replace('_','')) for word in data['end'].tolist()]
    end = list(set(end))
    print('end entity size: ')
    print(len(end))
    all_e = list(set(start + end))
    print('all entity size: ')
    print(len(all_e))  
    with open('start_entity.txt', 'w') as fw:
        for word in start:
            fw.write(word + '\n')
    with open('end_entity.txt', 'w') as fw:
        for word in start:
            fw.write(word + '\n')   
    with open('all_entity.txt', 'w') as fw:
        for word in all_e:
            fw.write(word + '\n')
    relation = [cht2chs(strip(word, 2).replace('_','')) for word in data['relation'].tolist()]
    relation = list(set(relation))
    print('relation size: ')
    print(len(relation))
    with open('relations.txt', 'w') as fw:
        for word in set(relation):
            fw.write(word + '\n')
    

def strip(str, index=3):
    return str.split('/')[index]

def getTriple(data):
    start = [cht2chs(strip(word).replace('_',''))for word in data['start'].tolist()]
    end = [cht2chs(strip(word).replace('_','')) for word in data['end'].tolist()]
    relation = [cht2chs(strip(word, 2).replace('_','')) for word in data['relation'].tolist()]
    triples = list(zip(start, relation, end))
    with open('chinese_triples.txt', 'w') as file:
        for triple in triples:
            file.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
    return triples

if __name__ == "__main__":
    data = loadConceptnet()
    # triples = getTriple(data)

    getEntity(data)
#     topK_result = search(data, "meddle", 20)
#     for i in topK_result.index:
#         i = topK_result.loc[i]
#         if len(template[i['relation']]) > 0:
#             fanti = strip(i['start']) + template[i['relation']].format(strip(i['end']))
#             print(cht2chs(fanti))

        
        
        