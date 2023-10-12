import json
import os
import re
from sentence_transformers import SentenceTransformer
from scipy import spatial
import pandas as pd
from functools import reduce
import re

def read_json(in_path):
    # in_list = list()
    out_list = list()
    with open(in_path, 'r') as f:
        tmp_list = f.readlines()
    for line in tmp_list:
        line = line.strip('\n')
        line_json = json.loads(line)
        out_list.append(line_json)
    return out_list

def split_camel_case(name):
    return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()


model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

type_black_list = ["int","bool","char","float","size_t","unsigned int","ints","const double", "enum","","dbus_int32_t*","dbus_bool_t","zip_uint64_t"]

def deleteDuplicate(li):
    func = lambda x, y: x if y in x else x + [y]
    li = reduce(func, [[], ] + li)
    return li

def cal_similarity(text1, text2):
    embedding_1 = model.encode(str(text1))
    embedding_2 = model.encode(str(text2))
    cosine_similarity = 1 - spatial.distance.cosine(embedding_1, embedding_2)
    return cosine_similarity

def read_from_json(filename,path):
    data = []
    with open(os.path.join(path,filename), 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def write2json(filename, data,path):
    with open(os.path.join(path,filename), 'w') as f:
        for line in data:
            f.write(json.dumps(line))
            f.write('\n')

def get_malloc_free(struct_json,lib_dict):
    malloc_ = []
    free_= []
    for each in struct_json:
        if each['apiname'] in lib_dict.keys():
            # each['link'] = lib_dict[each['apiname']]['link']
            each['type'] = lib_dict[each['apiname']]['type']
            each['object'] = lib_dict[each['apiname']]['object']
            lret_count = each['return_type'].strip().count("*")
            each['return_type'] = each['return_type'].strip().replace("*","").strip()+"*"*(lret_count)
            for tmp in each['parameters']:
                lparam_count = tmp['parameter'].strip().count("*") + tmp['type'].strip().count("*")
                tmp['parameter'] = tmp['parameter'].strip().replace("*","").strip()
                tmp['type'] = tmp['type'].replace("*","").strip()+"*"*(lparam_count)
                if tmp['type'] in type_black_list:
                    if tmp['type'] == "":
                        tmp['type'] = "void"
                    else:
                        tmp['type'] = ""
                        tmp['parameter'] = ""
            if each['type'] == "malloc":
                malloc_.append(each)
            elif each['type'] == "free":
                free_.append(each)
    return malloc_, free_

def get_lib_dict(in_path):
    in_list = read_json(in_path)
    out_dict = dict()
    for line in in_list:
        lib = line['lib']
        apiname = line['apiname']
        obj = line['obj']
        api_type = line['type']
        info_dict = dict()
        info_dict['type'] = api_type
        info_dict['object'] = obj

        if lib not in out_dict.keys():
            api_dict = dict()
            # info_dict = dict()
            api_dict[apiname] = info_dict
            out_dict[lib] = api_dict
        else:
            out_dict[lib][apiname] = info_dict
    return out_dict

      

def get_op_index(lib_dict,lib):
    struct_json = read_from_json("struct_%s.json"%lib, "/home/jhliu/api_pair/data/struct/")
    malloc_, free_ = get_malloc_free(struct_json, lib_dict[lib])
    # print(malloc_)
    for each in malloc_:
        for param in each['parameters']:
            if param['parameter'] == '' and param['type'] == '':
                param['similarity'] = 0
            else:
                tmp_type = " ".join(split_camel_case(param['type'])).replace("Const", "").replace("*","").replace("const","")
                tmp_type = " ".join(tmp_type.split("_"))
                param['similarity'] = cal_similarity(each['object'],tmp_type+" "+param['parameter'])
        each['return_type'] = each['return_type'].replace('XMLCALL','').replace("(","").strip()
        if each['return_type'] in type_black_list:
            each['return_similarity'] = 0
        else:
            tmp_type = " ".join(split_camel_case(each['return_type'])).replace("Const", "").replace("*","").replace("const","")
            tmp_type = " ".join(tmp_type.split("_"))
            each['return_similarity'] = cal_similarity(each['object'], tmp_type)#+" "+each['apiname'])
        
        if each['type'] == 'free':
            each['return_similarity'] = 0
        max_sim_index  = max(enumerate(each['parameters']), key=lambda x: x[1]['similarity'])[0]

        if each['return_similarity'] > each['parameters'][max_sim_index]['similarity']:
            each['op_index'] = -1
            each['op_type'] = each['return_type']
        else:
            each['op_index'] = max_sim_index
            each['op_type'] = each['parameters'][max_sim_index]['type']
    for each in free_:
        for param in each['parameters']:
            if param['parameter'] == '' and param['type'] == '':
                param['similarity'] = 0
                each['op_type'] = each['return_type']
            else:
                tmp_type = " ".join(split_camel_case(param['type'])).replace("Const", "").replace("*","").replace("const","")
                tmp_type = " ".join(tmp_type.split("_"))
                param['similarity'] = cal_similarity(each['object'],tmp_type+" "+param['parameter'])
        if each['return_type'] in type_black_list:
            each['return_similarity'] = 0
        else:
            tmp_type = " ".join(split_camel_case(each['return_type'])).replace("Const", "").replace("*","").replace("const","")
            tmp_type = " ".join(tmp_type.split("_"))
            each['return_similarity'] = cal_similarity(each['object'], tmp_type)#+" "+each['apiname'])
        each['return_type'] = each['return_type'].replace('XMLCALL','').replace("(","").strip()
        if each['type'] == 'free':
            each['return_similarity'] = 0
        max_sim_index  = max(enumerate(each['parameters']), key=lambda x: x[1]['similarity'])[0]

        if each['return_similarity'] > each['parameters'][max_sim_index]['similarity']:
            each['op_index'] = -1
            each['op_type'] = each['return_type']
        else:
            each['op_index'] = max_sim_index
            each['op_type'] = each['parameters'][max_sim_index]['type']
    return malloc_, free_

if __name__ == "__main__":
    in_path = "../QA/libpcap_re/final_api"
    out_path = 'mf_op_5.json'
    
    
    lib_dict = get_lib_dict(in_path)
    lib_list = lib_dict.keys()
    all_ops = []
    for lib in lib_list:
        print(lib)
        malloc_, free_ = get_op_index(lib_dict,lib)
        malloc_ = deleteDuplicate(malloc_)
        free_ = deleteDuplicate(free_)
        all_ops.extend(malloc_)
        all_ops.extend(free_)
    with open(out_path,'w') as f:
        json.dump(all_ops,f)