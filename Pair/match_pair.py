from functools import reduce
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from scipy import spatial

model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
def cal_similarity(text1, text2):
    embedding_1 = model.encode(str(text1))
    embedding_2 = model.encode(str(text2))
    cosine_similarity = 1 - spatial.distance.cosine(embedding_1, embedding_2)
    return cosine_similarity

def deleteDuplicate(li):
    func = lambda x, y: x if y in x else x + [y]
    li = reduce(func, [[], ] + li)
    return li

def match_pair(m_lib,f_lib):
    for each in m_lib:
        each['op_type'] = each['op_type'].replace("*","")
    for each in f_lib:
        each['op_type'] = each['op_type'].replace("*","")
    df_malloc = pd.DataFrame(m_lib)
    df_free = pd.DataFrame(f_lib)
    pair_list = []
    for each in m_lib:
#         print(each)
        op_type = each['op_type']
#         print(op_type)
        free_df = df_free.query('op_type == "%s"'%op_type)
#         print(free_df)
        for each_free in free_df.iterrows():
            tmp_dict = {}
            tmp_dict["malloc_api"] = each['apiname']
            tmp_dict["malloc_QA"] = each['object']
            tmp_dict["malloc_type"] = each['op_type']
            tmp_dict["malloc_index"] = each['op_index']
            tmp_dict["free_api"] = each_free[1]["apiname"]
            tmp_dict["free_QA"] = each_free[1]["object"]
            tmp_dict["free_type"] = each_free[1]['op_type']
            tmp_dict["free_index"] = each_free[1]["op_index"]
            # tmp_dict['score'] = 
            pair_list.append(tmp_dict)
    pair_list = deleteDuplicate(pair_list)
    return pair_list

def get_max_pair(tmp_pairs):
    in_dict = dict()
    out_list = list()
    for pair in tmp_pairs:
        malloc_api = pair['malloc_api']
        malloc_obj = pair['malloc_QA']
        free_obj = pair['free_QA']
        score = cal_similarity(malloc_obj, free_obj)
        if malloc_api in in_dict.keys():
            
            if score > in_dict[malloc_api]['score']:
                in_dict[malloc_api]['score'] = score
                in_dict[malloc_api] = pair
        else:
            pair['score'] = score
            in_dict[malloc_api] = pair
    for key in in_dict.keys():
        out_dict = dict()
        out_dict = in_dict[key]
        out_dict['malloc_api'] = key
        out_list.append(out_dict)
    return out_list

in_path = 'mf_op_5.json'
out_path = 'pairs.csv'
all_ops = json.load(open(in_path,'r'))
data_format = {}
for each in all_ops:
    if each['lib'] not in data_format.keys():
        data_format[each['lib']] = {}
        data_format[each['lib']]['malloc'] = []
        data_format[each['lib']]['free'] = []
    tmp_dict = {}
    tmp_dict['apiname'] = each['apiname']
    tmp_dict['op_index'] = each['op_index']
    tmp_dict['op_type'] = each['op_type']
    tmp_dict['object'] = each['object']
    # tmp_dict['score'] = each['similarity']
    if each['type'] == 'malloc':
        data_format[each['lib']]['malloc'].append(tmp_dict)
    elif each['type'] == 'free':
         data_format[each['lib']]['free'].append(tmp_dict)
         
pairs = []


for lib in data_format.keys():
    tmp_pair = match_pair(data_format[lib]['malloc'], data_format[lib]['free'])
    # print(tmp_pair)
    tmp_pair = get_max_pair(tmp_pair)
    # exit(1)
    for each in tmp_pair:
        each['lib'] = lib
    pairs.extend(tmp_pair)
df = pd.DataFrame(pairs)
df.to_csv(out_path,index=False)