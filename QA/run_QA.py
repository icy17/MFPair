import csv
import json
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import collections
from transformers.pipelines.pt_utils import KeyPairDataset
# from pyinflect import getAllInflections, getInflection
# import nltk
import math
import spacy
import os
import time
from transformers import pipeline
import sys
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForQuestionAnswering
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import copy

nlp = spacy.load('en_core_web_sm')

name_list_fruit = ['apple', 'banana', 'orange', 'grape','pear', 'mango', 'lemon', 'peach', 'apricot', 'coconut']
name_list_name = ['Mary', 'Bob', 'Alex', 'Alice', 'Tom', 'Tim', 'Amber', 'Jim', 'Annie', 'Ali']

def gen_csv(in_list, out_csv):
    in_content = list()
    with open(in_list, 'r') as f:
        in_content = f.readlines()
    list_info = list()
    for data in in_content:
        data = data.strip('\n')
        json_data = json.loads(data)
        # json_data['apiname'] = json_data['apiname'].strip('\n')
        list_info.append(json_data)
    header = list(list_info[0].keys())
    with open(out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=header) # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()  # 写入列名
        writer.writerows(list_info) # 写入数据
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

def get_api_list(api_path):
    api_struct = read_json(api_path)
    api_list = list()
    for struct in api_struct:
        api_name = struct['apiname']
        if api_name == '':
            continue
        api_list.append(api_name)
    return api_list
def gen_all_api(gt_api_dict, no_operator_path):
    no_operator_api = read_json(no_operator_path)

    for api in no_operator_api:
        gt_api_dict[api['api']] = api['type']
    return gt_api_dict


def if_api_related(sentence, api_list):
    sentence = sentence.lower()
    flag = False
    for api in api_list:
        if len(api) == 0:
            continue
        api = api.lower()
        if sentence.find(api) != -1:
            # print('sentence: %s, can find api: %s' % (sentence, api))
            flag = True
            break
    return flag

def get_token_sentence(sentence, api_list):
    out_dict = dict()
    res = if_api_related(sentence, api_list)
    api_num = 0
    token_num = 0
    if res or sentence.find('_') != -1:
        doc = nlp(sentence)
        # doc = sentence.split(' ')
        # print('nlp tokens::')
        # print(doc)
        token_num = 0
        replace_list = list()
        for token in doc:
            token = str(token)
            # print('nlp tokens::')
            # print(token)
            # print(api_list)
            if get_right_api(token, api_list) != '':
            # if if_api_related(token, api_list):
                # print('related!')
                # if 
                if token in replace_list:
                    continue
                if api_num >= len(name_list):
                    out_dict = dict()
                    return out_dict
                # api = get_right_api(token, api_list)

                api_token = name_list[api_num]
                api_num += 1
                out_dict[api_token] = str(token)
                replace_list.append(token)
            else:
                if token.find('_') != -1:
                    if token in replace_list:
                        continue
                    if token_num >= len(name_list_fruit):
                        out_dict = dict()
                        return out_dict
                    token_rep = name_list_fruit[token_num]
                    token_num += 1
                    out_dict[token_rep] = str(token)
                    replace_list.append(token)
    return out_dict


def replace_srl(sentence, api_list):
    sentence = sentence.lower()
    token_dict = get_token_sentence(sentence, api_list)
    doc = nlp(sentence)
    final_sentence = ''
    if len(token_dict) != 0:
        for token in doc:
            tmp_token = str(token)
            for key in token_dict.keys():
                if token_dict[key] == str(token):
                    tmp_token = key
                    break
            final_sentence += str(tmp_token)
            final_sentence += ' '
            # sentence = sentence.replace(token_dict[key], key)
    if final_sentence == '':
        final_sentence = sentence
    # if final_sentence.find('_') != -1:
        
    #     # print('in replace')
    #     # print(final_sentence)
    #     doc = nlp(final_sentence)
    #     final_sentence2 = ''
    #     for token in doc:
    #         tmp_token = str(token)
    #         if tmp_token.find('_') != -1:
    #             tmp_token = 'symbol'
    #         final_sentence2 += str(tmp_token)
    #         final_sentence2 += ' '
    #     return final_sentence2, token_dict
    # else:
    return final_sentence, token_dict

def get_vb(word, verb_tag, orig_verb):
    res_tmp = word
    # orig_verb = get_vb(word, 'VB')
    if word == 'deallocates' or word == 'deallocated':
        orig_verb = 'deallocate'
    # if word == 'deallocate' or word == 'deallocates':
    #     res_tmp = 'deallocating'
    word = nlp(word)
    word = word[0].lemma_
    if verb_tag == 'VN':
        if orig_verb == 'allocate':
            res = 'allocation'
        elif orig_verb == 'create':
            res = 'creation'
        elif orig_verb == 'construct':
            res = 'construction'
        elif orig_verb == 'initialize':
            res = 'initialization'
        elif orig_verb == 'free':
            res = 'free'
        elif orig_verb == 'release':
            res = 'release'
        elif orig_verb == 'deallocate':
            res = 'deallocation'
        else:
            res = orig_verb
    else:
        res = getInflection(word , tag=verb_tag)
        # print('orig: ' + word)
        if res == None:
            return orig_verb
        res = res[0]
        # print(res)n_qa
    return res

def get_question(question_list, first_token, keyword):
    orig_verb = get_vb(keyword, 'VB', '')
    keyword_right = get_vb(keyword, question_list[1], orig_verb)
    question = first_token + question_list[0] + keyword_right + question_list[2]
    return question

def question_answer(question, text):
    text = text.replace('()', '')
    text = text.replace('( )', '')
    text = text.replace('(s)', '')
    text = text.replace('(3)', '')
    # print(text)
    if len(name_list) != 0:
        text, token_dict = replace_srl(text, api_list)
    
    context = text
    res = qa_model(question = question, context = context)
    answer = res['answer'].lower()
    score = res['score']
    doc = nlp(answer)
    answer_text = ''
    if len(name_list) != 0:
        for token in doc:
            token = str(token)
            for key in token_dict.keys():
                # print('in debug::')
                # print(answer)
                # print(key)
                # print(token_dict[key])
                if token.strip() == key.lower():
                    # print('if find')
                    token = token.replace(key.lower(), token_dict[key], 1).strip()
            answer_text += token
            answer_text += ' '
        answer = answer_text
    if len(name_list) == 0:
        answer = answer.replace('_ ', '_')
        answer = answer.replace(' _', '_')
                # print(answer + '\n\n')
    return answer, score, context

def question_answer1(question, text):
    text = text.replace('()', '')
    text = text.replace('( )', '')
    text = text.replace('(s)', '')
    text = text.replace('(3)', '')
    # print(text)
    if len(name_list) != 0:
        text, token_dict = replace_srl(text, api_list)
    # print('after replace')
    # print(text)
    # print(text)
    # print(token_dict)

    #tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    #number of tokens in segment A - question
    num_seg_a = sep_idx+1

    #number of tokens in segment B - text
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    assert len(segment_ids) == len(input_ids)
        
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #reconstructing the answer

    m = torch.nn.Softmax(dim=1)
    answer_start_score = m(output.start_logits)
    answer_end_score = m(output.end_logits)
    answer_start = torch.argmax(answer_start_score)
    answer_end = torch.argmax(answer_end_score)
    answer_start_score = answer_start_score.detach().numpy().flatten().tolist()
    answer_end_score = answer_end_score.detach().numpy().flatten().tolist()
    # print(answer_start_score)
    # print(answer_end_score)
    # answer_start = torch.argmax(output.start_logits)
    # answer_end = torch.argmax(output.end_logits)
    # max_list = [1] * len(start_list)
    # max_entropy = get_result_max(torch.Tensor(max_list))
    # get start/end scores:
    # answer_start_score = output.start_logits.detach().numpy().flatten().tolist()[answer_start]
    # answer_end_score = output.end_logits.detach().numpy().flatten().tolist()[answer_end]
    # print(tokens)
    # print(output.end_logits.detach().numpy().flatten().tolist())
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    else:
        return None, 0, text
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
        return None, 0, text
    print("\nAnswer:\n{}".format(answer.capitalize()))
    # tobe test
    # tmp_text = text.replace(' ', '')
    # tmp_answer = answer.replace(' ', '')
    # if tmp_text.lower().find(tmp_answer.lower()) == -1:
    #     return answer, 0
    if len(name_list) != 0:
        for key in token_dict.keys():
            # print('in debug::')
            # print(answer)
            # print(key)
            # print(token_dict[key])
            if answer.find(key.lower()) != -1:
                # print('if find')
                answer = answer.replace(key.lower(), token_dict[key], 1)
    if len(name_list) == 0:
        answer = answer.replace('_ ', '_')
        answer = answer.replace(' _', '_')
                # print(answer + '\n\n')
    return answer, min(answer_start_score[answer_start], answer_end_score[answer_end]), text

# def parse_result(answer, right_answer, score, threshold):
#     if score < threshold:
#         answer = '0'
#     if right_answer == answer:
#         return 'TP'
#     find_list = right_answer.split(' ')
#     for str in find_list:
#         match_api = get_right_api(str, api_list)
#         if match_api == right_answer:
#             return 'TP'
#         else:
#             continue
#     if right_answer == '1':
#         return 'TP'
#     else:
#         if right_answer != answer and answer != '0':
#             return 'FP'
#         elif right_answer != '0' and answer == '0':
#             return 'FN'
# precision: 分子：测出来正确的1+API(TP，分母：所有测出来的1+API(TP + FP)
# recall:分子:测出来正确的1+API(TP), 分母: 测试集中的所有1+API(all)
def parse_result(answer, right_answer, score, threshold, api_type, question):
    # operator1 = operator1.strip()
    # threshold = 0.5
    res_type = ''
    if question.find('allocating') != -1:
        res_type = 'malloc'
    elif question.find('freeing') != -1:
        res_type = 'free'
    # print(res_type)
    # print(api_type)
    answer = answer.strip()
    right_answer = right_answer.lower()
    answer = answer.lower()
    # print(threshold)
    # print(score)
    if score < threshold:
        answer = '0'
    
    if answer == '0':
        if right_answer != '0':
            return 'FN'
        return 'N'
    if api_type != 'all' and api_type != res_type:
        return 'Wrong-1'
    # print('pass type')
    # if right_answer == answer:
    #     return 'TN'
    
    # find_list = answer.split(' ')
    # for str in find_list:
    #     match_api = get_right_api(str, api_list)
    #     if match_api == right_answer:
    #         return 'TP'
    #     else:
    #         continue
    if right_answer.find('[[or]]') != -1:
        right_answers = right_answer.split('[[or]]')
        flag = False
        for right_answer in right_answers:
            right_answer = right_answer.strip()
            if right_answer != '0' and right_answer != '1':
                answer = answer.replace(' ', '')
                flag = True
                right_answer = right_answer.replace(' ', '')
                if answer.find(right_answer) != -1 or right_answer.find(answer) != -1:
                    return 'TP'
            # if right_answer == '1':
            #     flag = True
            #     answer = answer.replace(' ', '')
            #     operator1 = operator1.replace(' ', '')
            #     if answer.find(operator1) != -1 or operator1.find(answer) != -1:
            #         return 'TP'
        if flag:
            return 'Wrong-1'
        return 'Wrong'

    else:
        if right_answer != '0' and right_answer != '1':
            answer = answer.replace(' ', '')
            right_answer = right_answer.replace(' ', '')
            if answer.find(right_answer) != -1 or right_answer.find(answer) != -1:
                return 'TP'
            else:
                return 'Wrong-1'
        # if right_answer == '1':
        #     answer = answer.replace(' ', '')
        #     operator1 = operator1.replace(' ', '')
        #     if answer.find(operator1) != -1 or operator1.find(answer) != -1:
        #         return 'TP'
        #     else:
        #         return 'Wrong-1'
        else:
            return 'Wrong'

def gen_final_files(out_dir, threshold_list, gt_path):
    qa_path = out_dir + '-all_result'
    qa_result = read_json(qa_path)
    gt_lists = read_json(gt_path)

    for threshold in threshold_list:
        out_final_dir = out_dir + str(threshold)
        out_path_wrong = out_final_dir + '-all_wrong'
        out_path_fn = out_final_dir + '-all_fn'
        out_path_tp = out_final_dir + '-all_TP'
        if os.path.exists(out_path_wrong):
            os.remove(out_path_wrong)
        if os.path.exists(out_path_fn):
            os.remove(out_path_fn)
        if os.path.exists(out_path_tp):
            os.remove(out_path_tp)
        # print(out_path_wrong)
        # out_path = ''
        for result_json in qa_result:
            index = qa_result.index(result_json)
            # if index > 479:
            #     index -= 8
            # if index > 471 and index < 479:
            #     continue
            gt_json = gt_lists[index]
            sentence = gt_json['sentence']
            right_answer = gt_json['QA-API']
            if right_answer == '':
                continue
            # operator1 = str(gt_json['operator1'])
            result_list = list()
            res_dict = result_json
            api_type = gt_json['API-type']
            # question= result_json['']
            for item in result_json['info']:
                re = parse_result(item['answer'], right_answer, item['score'], threshold, api_type, item['question'])
                # print(re)

                result_list.append(re)
            # exit(1)
            if 'Wrong-1' in result_list:
                res_dict['result'] = 'Wrong-1'
            elif 'Wrong' in result_list:
                res_dict['result'] = 'Wrong'
            elif 'TP' in result_list:
                res_dict['result'] = 'TP'
            elif 'FN' in result_list:
                res_dict['result'] = 'FN'
            else:
                res_dict['result'] = 'others'
            # res_dict['replace'] = gt_json['replace']
            res_dict['sentence'] = sentence
            res_dict['QA-API'] = right_answer
            if res_dict['result'] == 'Wrong-1':
                with open(out_path_wrong, 'a') as f:
                    f.write(json.dumps(res_dict))
                    f.write('\n')
                with open(out_path_fn, 'a') as f:
                    f.write(json.dumps(res_dict))
                    f.write('\n')
            if res_dict['result'] == 'TP':
                # if lib == 'libxml2':
                #     libxml2_tp += 1
                # elif lib == 'zlib':
                #     zlib_tp += 1
                # elif lib == 'libzip':
                #     libzip_tp += 1
                # all_tp += 1
                with open(out_path_tp, 'a') as f:
                    f.write(json.dumps(res_dict))
                    f.write('\n')
            # elif res_dict['result'] == 'FP':
            #     if lib == 'libxml2':
            #         libxml2_fp += 1
            #     elif lib == 'zlib':
            #         zlib_fp += 1
            #     elif lib == 'libzip':
            #         libzip_fp += 1
            #     all_fp += 1
            #     with open(out_path_fp, 'a') as f:
            #         f.write(json.dumps(res_dict))
            #         f.write('\n')
            # elif res_dict['result'] == 'FN':
            #     if lib == 'libxml2':
            #         libxml2_fn += 1
            #     elif lib == 'zlib':
            #         zlib_fn += 1
            #     elif lib == 'libzip':
            #         libzip_fn += 1
            #     all_fn += 1
            #     with open(out_path_fn, 'a') as f:
            #         f.write(json.dumps(res_dict))
            #         f.write('\n')
            elif res_dict['result'] == 'Wrong':
                # if lib == 'libxml2':
                #     libxml2_wrong += 1
                # elif lib == 'zlib':
                #     zlib_wrong += 1
                # elif lib == 'libzip':
                #     libzip_wrong += 1
                # all_wrong += 1
                with open(out_path_wrong, 'a') as f:
                    f.write(json.dumps(res_dict))
                    f.write('\n')
            elif res_dict['result'] == 'FN':
                with open(out_path_fn, 'a') as f:
                    f.write(json.dumps(res_dict))
                    f.write('\n')
            # with open(out_path_all, 'a') as f:
            #         f.write(json.dumps(res_dict))
            #         f.write('\n')
            
def cal_F1(precision, recall):
    if precision + recall == 0:
        F1  = 0
    else:
        F1 = 2*(precision * recall) / (precision + recall)
    return F1


def generate_log(out_dir, threshold_list):
    libs = ['libzip', 'zlib', 'libevent', 'libgnutls', 'ldap', 'ffmpeg', 'libpcap', 'libdbus', 'libexpat', 'libmysql']
    model_tp = 0
    model_wrong = 1000
    model_f1 = 0
    all_recall = read_json(out_all_recall)
    model_desc = ''
    for threshold in threshold_list:
        out_final_dir = out_dir + str(threshold)
        all_result = out_dir + '-all_result'
        all_tp_path = out_final_dir + '-all_TP'
        all_wrong_path = out_final_dir + '-all_wrong'
        out_log_path = out_final_dir + '-all_data_log'
        api_prefix = out_final_dir.split('/') [-1]
        api_out = '/api-out'
        out_api = out_api_dir + api_out
        api_score_log = out_api_dir + '/api_score_log'
        if not os.path.exists(out_api):
            os.mkdir(out_api)
        api_res = parse_api(out_api+ '/' + api_prefix, all_result, in_path, threshold)
        api_res['info'] = api_prefix
        with open(api_score_log, 'a') as f:
            f.write(json.dumps(api_res))
            f.write('\n')
        if os.path.exists(out_log_path):
            os.remove(out_log_path)
        if not os.path.exists(all_tp_path):
            all_tp = []
        else:
            all_tp = read_json(all_tp_path)
        if not os.path.exists(all_wrong_path):
            all_wrong = []
        else:

            all_wrong = read_json(all_wrong_path)
        
        all_tp_num = 0
        all_wrong_num = 0
        all_num = 0
        all_recall_num = 0
        res_str = ''
        for lib in libs:
            lib_tp = 0
            lib_wrong = 0
            lib_recall = 0
            # lib_all = 0
            for tp in all_tp:
                if tp['lib'] == lib:
                    lib_tp += 1
            for wrong in all_wrong:
                if wrong['lib'] == lib:
                    lib_wrong += 1
            for recall in all_recall:
                if recall['lib'] == lib:
                    lib_recall += 1
            all_tp_num += lib_tp
            all_wrong_num += lib_wrong
            all_recall_num += lib_recall
            lib_recall_final = lib_tp / lib_recall
            if (lib_tp + lib_wrong) == 0:
                lib_precision = 0
            else:
                lib_precision = lib_tp / (lib_tp + lib_wrong)
            res_str = res_str + 'Lib: %s\nPrecision: %s  Recall: %s\n' % (lib, str(lib_precision), str(lib_recall_final))
            res_str  = res_str + 'FP: %s  TP: %s  All operator: %s\n' % (str(lib_wrong), str(lib_tp), str(lib_recall))
        all_recall_final = all_tp_num / all_recall_num
        
        if (all_tp_num + all_wrong_num) == 0:
            all_precision = 0
        else:
            all_precision = all_tp_num / (all_tp_num + all_wrong_num)
        all_f1 = cal_F1(all_precision, all_recall_final)
        all_dict = dict()
        res_str = res_str + '\n\nAll: \nPrecision: %s  Recall: %s  F1: %s\n' % (str(all_precision), str(all_recall_final), str(all_f1))
        res_str  = res_str + 'FP: %s  TP: %s  All operator: %s\n' % (str(all_wrong_num), str(all_tp_num), str(all_recall_num))
        with open(out_log_path, 'a') as f:
            f.write(res_str + '\n')
        all_dict['threshold'] = threshold
        all_dict['f1'] = all_f1
        all_dict['precision'] = all_precision
        all_dict['recall'] = all_recall_final
        all_dict['replace'] = out_dir.strip().split('/')[-1]
        with open(out_all_log, 'a') as f:
            f.write(json.dumps(all_dict))
            f.write('\n')
        if all_f1 > model_f1:
            model_tp = all_tp_num
            model_wrong = all_wrong_num
            model_desc = out_final_dir
            model_f1 = all_f1
        # if all_tp_num > model_tp:
        #     model_tp = all_tp_num
        #     model_wrong = all_wrong_num
        #     model_desc = out_final_dir
    if (model_tp + model_wrong) == 0:
        model_precision = 0
    else:
        model_precision = model_tp / (model_tp + model_wrong)
    model_recall = model_tp / all_recall_num
        
    return model_precision, model_recall, model_desc

def match_api_list(operator, api_list):
    # print(api_list)
    # exit(1)
    operator = operator.replace('(3)', '')
    operator = operator.replace('(3', '')
    operator = operator.replace('(z', '')
    operator = operator.lower()
    operator = operator.replace(',', '')
    # doc = nlp(operator)
    doc = operator.split(' ')
    # doc = list()
    # if operator.find(',') != -1:
    #     doc = operator.split(',')
    # if operator.find(' or ')!= -1:
    #     doc = operator.split(' or ')
    # if operator.find('/')!= -1:
    #     doc = operator.split('/')
    # list1 = operator.split(',')
    # for item in list1:
    #     list2 = item.split(' or ')
    #     for api in list2:
    #         doc.append(api)
    # return doc
    match_list = list()
    out_match_list = list()
    out_missing_list = list()
    for api in api_list:
        match_list.append(api.lower())
    for token in doc:
        token = str(token).strip('()')
        token = token.strip()
        if token in match_list:
            out_match_list.append(token)
        else:
            # if token in api_list:
            if token.find('dbus') != -1:
                if '_' + token in match_list:
                    out_match_list.append(token)
            out_missing_list.append(token)
    return out_match_list, out_missing_list

    # for api in api_list:

def get_info_list(info_list):
    info_dict = dict()
    out_list = list()
    for info in info_list:
        question = info['question']
        sentence = info['sentence']
        key = sentence + question
        if key not in info_dict.keys():
            info_dict[key] = info
    for key in info_dict.keys():
        out_list.append(info_dict[key])

def parse_api(out_dir, all_path, gt_path, threshold):
    print(out_dir)
    print(all_path)
    print(gt_path)
    # target_libs = ['libpcap', 'libdbus', 'ffmpeg']
    out_dict_path = out_dir + '-api_res'
    out_fn_path = out_dir + '-fn_res'
    gt_list = read_json(gt_path)
    # api_list
    res_list = read_json(all_path)
    api_dict = dict()
    gt_api = list()
    missing_dict = dict()
    for line in gt_list:
        # if line['lib'] not in target_libs:
        #     continue
        api = line['QA-API']
        tmp_list, missing_api = match_api_list(api, api_list)
        # for item in missing_api:
        #     missing_dict[item] = 1
        # for item in missing_api:
        #     with open('./miss_api', 'a') as f:
        #         f.write(item + '\n')
        api_type = line['API-type']
        if api_type == '':
            print(line)
        for api in tmp_list:
            if api != '' and api in api_list:
                api_dict[api] = api_type
                gt_api.append(api)
    # api_d/ict = gen_all_api(api_dict, no_operator_path)
            # if api not in gt_api:
            #     gt_api.append(api)
        # api_list.append(api)
    # api_list = list(set(api_list))
    # for key in missing_dict.keys():
    #     with open('./miss_api', 'a') as f:
    #             f.write(key + '\n')
    api_dict_res = dict()
    api_type_dict = dict()
    api_sentence_dict = dict()
    api_info_dict = dict()
    print('in parse api before first parse')
    print(threshold)
    # print()
    i = 0
    for line in res_list:
        i += 1
        print(i)
        # if line['lib'] not in target_libs:
        #     continue
        info_list = line['info']
        right_answer = line['QA-API']
        # if i == 148:
        print(len(info_list))
        for info in info_list:
            # if i == 148:
            res_type = ''
            # print(info)
            question = info['question']
            answer = info['answer']
            sentence = info['sentence']
            # get_api
            answer_api = list()
            for key in api_list:
                index = answer.find(key)
                match_api = ''
                if index != -1:
                    if len(answer) > len(key) + index and (answer[index + len(key)].isalpha() or answer[index + len(key)]== '_' or answer[index + len(key)].isdigit()):
                        match_api = ''
                    elif index != 0 and (answer[index-1].isalpha() or answer[index-1]== '_' or answer[index-1].isdigit()):
                        match_api = ''
                    else:
                        match_api = key
                if match_api != '' and key != '':
                    answer_api.append(match_api)
            print(answer_api)
                    # answer_list = answer.split(' ')
            # print(answer_api)
            # exit(1)
                # for answer_item in answer_list:
                #     api = get_right_api(answer_item, gt_api)
                #     if api != '':
                #         answer_api.append(key)
            # answer_api, missing_api = match_api_list(answer, api_list)
            # end
            score = info['score']
            if score > threshold:
                if question.find('allocating') != -1:
                    res_type = 'malloc'
                elif question.find('freeing') != -1:
                    res_type = 'free'
                # print('before answer_api')
                # print(len(info_list))
                if res_type == '':
                    continue
                for api in answer_api:
                    # api_dict = dict()
                    # api_dict['sentence'] = sentence
                    # api_dict['type'] = res_type
                    if res_type == '':
                        print(info)
                    if api in api_dict_res.keys():
                        if res_type == api_dict_res[api] or api_dict_res[api] == 'all':
                            api_sentence_dict[api + '-' + res_type].append(sentence)
                            api_info_dict[api + '-' + res_type].extend(info_list)
                            api_type_dict[api + '-' + res_type] = max(score, api_type_dict[api + '-' + res_type])
                            # api_sentence_dict[api + '-' + res_type]['sentence'].append(sentence)
                        else:
                            api_dict_res[api]= 'all'
                            tmp_list = list()
                            tmp_list.append(sentence)
                            tmp_list2 = list()
                            tmp_list2.extend(info_list)
                            api_sentence_dict[api + '-' + res_type] = tmp_list
                            api_info_dict[api + '-' + res_type] = tmp_list2
                            api_type_dict[api + '-' + res_type] = score
                            # api_sentence_dict[api + '-' + res_type]['sentence'].append(sentence)
                    else:
                        api_dict_res[api] = res_type
                        tmp_list = list()
                        tmp_list.append(sentence)
                        tmp_list2 = list()
                        tmp_list2.extend(info_list)
                        api_sentence_dict[api + '-' + res_type] = tmp_list
                        api_info_dict[api + '-' + res_type] = tmp_list2
                        api_type_dict[api + '-' + res_type] = score
            else:
                continue
    out_list = list()
    # calculate FP
    fp = 0
    tp = 0
    tp_dict = dict()
    fn_list = list()
    # for key in 
    print('in parse api before final')
    for key in api_dict_res.keys():
        tmp_dict = dict()
        tmp_dict['api'] = key
        res_type = api_dict_res[key]
        item_dict =dict()
        item_dict['api'] = key
        item_dict['res_type'] = api_dict_res[key]
        # tmp_dict['api'] = key
        if key not in api_dict.keys():
            item_dict['res'] = 'FP'
            if res_type == 'all':
                fp += 2
                item_dict['info'] = api_info_dict[key + '-' + 'malloc']
                item_dict['info'].extend(api_info_dict[key + '-' + 'free'])
            else:
                fp += 1
                item_dict['info'] = api_info_dict[key + '-' + res_type]
            out_list.append(item_dict)
            continue
        if api_dict_res[key] == 'all':
            if api_type_dict[key + '-free'] > api_type_dict[key + '-malloc']:
                api_dict_res[key] = 'free'
            else:
                api_dict_res[key] = 'malloc'
        if api_dict_res[key] == api_dict[key]:
            tp_dict[key] = api_dict[key]
            tp += 1
            tmp_dict['res'] = 'TP'
            tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['result_type'] =  api_dict_res[key]
            print(api_info_dict[key + '-' + api_dict[key]])
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict[key]]
            out_list.append(tmp_dict)

        elif api_dict_res[key] == 'all':
            tp += 1
            # tp_dict[key] = api_dict[key]
            tmp_dict['res'] = 'TP'
            # print(api_dict)
            tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['result_type'] =  api_dict_res[key]
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict[key]]
            out_list.append(tmp_dict)
            fp += 1
            wrong_dict = dict()
            wrong_dict['api'] = key
            wrong_dict['res'] = 'FP'
            if api_dict[key] == 'free':
                wrong_type = 'malloc'
            else:
                wrong_type = 'free'
            wrong_dict['sentences'] = api_sentence_dict[key + '-' + wrong_type]
            wrong_dict['result_type'] = wrong_type
            wrong_dict['info'] = api_info_dict[key + '-' + wrong_type]
            out_list.append(wrong_dict)
        else:
            fp += 1
            tmp_dict['res'] = 'FP'
            tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict_res[key]]
            tmp_dict['result_type'] =  api_dict_res[key]
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict_res[key]]
            out_list.append(tmp_dict)
    fn_list  = list()
    for key in api_dict.keys():
        if key not in api_dict_res.keys():
            fn_list.append(key)
    hit_fn = list()
    # for line in res_list:
    #     info = line['info']
    #     for fn_api in fn_list:
    #         if line['QA-API'].lower().find(fn_api) != -1:
    #             # with open()
    #             out_dict = dict()
    #             out_dict['miss_api'] = fn_api
    #             out_dict['sentence'] = line['sentence']
    #             out_dict['info'] = info
    #             hit_fn.append(fn_api)
    #             with open(out_fn_path, 'a') as f:
    #                 f.write(json.dumps(out_dict))
    #                 f.write('\n')
                # break
    for api in fn_list:
        if api not in hit_fn:
            out_dict = dict()
            out_dict['miss_api'] = api
            out_dict['sentence'] = ''
            out_dict['info'] = 'No'
            with open(out_fn_path, 'a') as f:
                f.write(json.dumps(out_dict))
                f.write('\n')
    # calculate FN
    fn = len(api_dict) - tp
    # precision
    precision = tp / (tp + fp)
    # recall
    recall = tp / len(api_dict)
    # f1
    f1 = cal_F1(precision, recall)
    # write
    for item in out_list:
        with open(out_dict_path, 'a') as f:
            f.write(json.dumps(item))
            f.write('\n')
    all_tp = len(api_dict)
    final_dict = dict()
    final_dict['FP'] = fp
    final_dict['fn'] = fn
    final_dict['precision'] = precision
    final_dict['recall'] = recall
    final_dict['f1'] = f1
    
    print(f'FP: {str(fp)}')
    print(f'FN: {str(fn)}')
    print(f'Precision: {str(precision)}')
    print(f'Recall: {str(recall)}')
    print(f'F1: {str(f1)}')
    print(f'All: {str(all_tp)}')
    return final_dict



# def get_fn(recall_path, tp_path):
#     all_tp = read_json(tp_path)
#     all_recall = read_json(recall_path)
#     for tp_json in all_tp:

def generate_threshold(begin, end, span):
    threshold_list = list()
    target = begin
    while 1:
        if target > end:
            break
        threshold_list.append(target)
        target += span
    if end not in threshold_list:
        threshold_list.append(end)
    return threshold_list
def get_right_api(sentence, api_list):
    max_api = ''
    max_len = 0
    for api in api_list:
        if sentence.find(api) != -1:
            length = len(api)
            if length > max_len:
                max_len = length
                max_api = api
    index = sentence.find(max_api)
    if len(sentence) > len(max_api) + index and (sentence[index + len(max_api)].isalpha() or sentence[index + len(max_api)]== '_' or sentence[index + len(max_api)].isdigit()):
        return ''
    return max_api

def parse_sentence(sentence_list):
    token_list = list()
    out_list = list()
    id = 0
    i = 0
    length = len(sentence_list)
    for sentence_json in sentence_list:
        id += 1
        print(f'{str(id)}/{str(length)}')
        
        text = sentence_json['sentence']
        # if text.lower() != 'free a gnutls_srp_client_credentials_t structure':
        #     continue
        # print(text)
        # exit(1)
        text = text.replace('()', '')
        text = text.replace('( )', '')
        text = text.replace('(s)', '')
        text = text.replace('(3)', '')
        # print(text)
        if len(name_list) != 0:
            text, token_dict = replace_srl(text, api_list)
            token_list.append(token_dict)
        out_dict = dict()
        out_dict = sentence_json
        out_dict['sentence'] = text
        # print(text)
        # exit(1)
        # out_dict['id'] = str(id)
        # out_dict['token'] = token
        out_list.append(out_dict)
    return token_list, out_list
    
def parse_back(sentence_list, token_list):
    out_list = list()
    answer_text = ''
    i = 0
    # print(token_list)
    
    for sentence_json in sentence_list:
        i += 1
        print(f'{str(i)}/{str(len(sentence_list))}')
        answer = sentence_json['answer'].lower()
        list_index = sentence_list.index(sentence_json)
        # print(list_index)
        # print(sentence_json)
        # print(token_list)
        token_dict = token_list[int(list_index / 4)]
        # print(token_dict)
        doc = nlp(answer)
        answer_text = ''
        if len(name_list) != 0:
            for token in doc:
                token = str(token)
                for key in token_dict.keys():
                    # print('in debug::')
                    # print(answer)
                    # print(key)
                    # print(token_dict[key])
                    if token.strip() == key.lower():
                        # print('if find')
                        token = token.replace(key.lower(), token_dict[key], 1).strip()
                answer_text += token
                answer_text += ' '
            answer = answer_text
        if len(name_list) == 0:
            answer = answer.replace('_ ', '_')
            answer = answer.replace(' _', '_')
        out_dict = dict()
        out_dict = sentence_json
        out_dict['answer'] = answer
        out_list.append(out_dict)

    return out_list

def get_validation_data(sentence_list):
    out_list_res = list()
    out_list_all = list()
    question_list = ['Who performs allocating?', 'Who performs freeing?', 'What has been allocated?', 'What has been freed?']
    for sentence_json in sentence_list:
        for question in question_list:
            out_dict = dict()
            out_all_dict =dict()
            out_all_dict = copy.deepcopy(sentence_json)
            out_dict['question'] = question
            out_dict['context'] = sentence_json['sentence']
            out_all_dict['question'] = question
            out_all_dict['context'] = sentence_json['sentence']
            # sentence_json['context'] = 
            # out_dict['id'] = sentence_json['id']
            out_list_res.append(out_dict)
            out_list_all.append(out_all_dict)
    return out_list_res, out_list_all
max_length = 384
stride = 128
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

n_best = 20
max_answer_length = 30
predicted_answers = []
def test_orig_model(validation_data_list, trained_checkpoint):
    # metric = evaluate.load("squad")
    # small_eval_set = raw_datasets["validation"].select(range(100))
    small_eval_set = validation_data_list
    out_list = list()
    # print(small_eval_set)
    # exit(1)
    # trained_checkpoint = "distilbert-base-cased-distilled-squad"
    # trained_checkpoint = model_name
    print(small_eval_set)
    eval_set = small_eval_set.map(
        preprocess_validation_examples,
        batched=True,
        # batch_size=32,
        remove_columns=validation_data_list.column_names,
    )
    print(eval_set)
    eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
    # print(eval_set_for_model[0])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eval_set_for_model.set_format("torch", device=device)

    
    dataloader = DataLoader(eval_set_for_model, batch_size=128)
    # batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

    # print(batch)
    trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
        device
    )
    # outputs
    print('wait for run')
    start_logits = list()
    end_logits = list()
    with torch.no_grad():
        for batch in dataloader:
            # batch=batch
            outputs = trained_model(**batch)
            # print(type(outputs))
            start_logits_s = softmax(outputs.start_logits, dim = 1)
            end_logits_s = softmax(outputs.end_logits, dim = 1)
            # print(start_logit)
            start_logits.extend(start_logits_s.cpu().numpy().tolist())
            end_logits.extend(end_logits_s.cpu().numpy().tolist())
    example_to_features = collections.defaultdict(list)
    
    for idx, feature in enumerate(eval_set):
        # print(idx)
        # print(feature)
        example_to_features[feature["example_id"]].append(idx)
    for example in small_eval_set:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            
            offsets = eval_set["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    answers.append(
                        {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": min(start_logit[start_index], end_logit[end_index]),
                        }
                    )
        
        best_answer = max(answers, key=lambda x: x["logit_score"])
        example['answer'] = best_answer["text"]
        example['score'] = best_answer["logit_score"]
        out_list.append(example)
    return out_list
    #     predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
    # theoretical_answers = [
    #     {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
    # ]
    # print(predicted_answers[0])
    # print(theoretical_answers[0])
    # # res = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    # print(res)

def gen_final_list(res_list,orig_list):
    out_list = list()
    first_line = ''
    num = 0
    info_list = list()
    out_dict = dict()
    i = 0
    length = len(res_list)
    for line in res_list:
        
        list_index = i
        i += 1
        print(f'{str(i)}/{str(length)}')
        del line['start']
        del line['end']
        # print(orig_list[list_index])
        # exit(1)
        # out_dict = copy.deepcopy(orig_list[list_index])
        line['question'] = orig_list[list_index]['question']
        line['context'] = orig_list[list_index]['context']
        out_dict['lib'] = orig_list[list_index]['lib']
        out_dict['apiname'] = orig_list[list_index]['apiname']
        info_dict = line
        out_dict['sentence'] = orig_list[list_index]['sentence']
        if 'close_api' in out_dict.keys():
            out_dict['close_api'] = orig_list[list_index]['close_api']
        if first_line == '':
            first_line = orig_list[list_index]['sentence']
            num = 1
            info_list.append(line)
        else:
            if num == 3 and orig_list[list_index]['sentence'] == first_line:
                num = 0
                info_list.append(line)
                out_dict['info'] = info_list
                out_list.append(out_dict)
                out_dict = dict()
                info_list = list()
                first_line = ''
            elif orig_list[list_index]['sentence'] == first_line and num < 3:
                num += 1
                info_list.append(info_dict)
    return out_list


if __name__ == '__main__':
    flag = 'run'
    if len(sys.argv) != 3:
        print('Usage: python3 ./run_QA.py <model_name> <out_dir>\n Example: python3 ./MF_identify.py deepset/roberta-base-squad2 ./libzip_re')
        out_dir = './libpcap_re'
        model_names = ['deepset/roberta-base-squad2']
        exit(1)
    else:
        out_dir = sys.argv[2] + '/'
        model_names = [sys.argv[1]]
    # Change: in_path is input sentence
    in_path = './libpcap'
    # Change: api_path contain all API name
    api_path = './API-list'
    # Change: hugging face token:
    access_token = 'Your hugging face token'

    out_api_dir = out_dir
    out_all_log = out_dir + 'all_score_log'
    out_all_recall = out_dir + 'all_recall'

    time_begin = time.time()
    name_lists = [2]
    name_list = list()
    
    
    question_list = ['Who performs allocating?', 'Who performs freeing?', 'What has been allocated?', 'What has been freed?']
    target_libs = ['libzip']
    
    first_list = ['', '']
    
    print('waiting for bert...')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # print(out_dir)
    api_list = get_api_list(api_path)
    sentence_list = read_json(in_path)
    if os.path.exists(out_all_recall):
            os.remove(out_all_recall)
    model_desc = ''
    out_log = out_dir + 'all_log'
    final_desc = ''
    final_f1 = 0
    final_precision = 0
    final_recall = 0
    for model_name in model_names:
        # if flag == '' or flag == 'run':
        #     qa_model = pipeline("question-answering", model=model_name, tokenizer=model_name, use_auth_token=access_token, device=0)
        time_model_begin = time.time()
        # if model_name.find('/') != -1:
        model_name = model_name.replace('/', '')
        out_dir1 = out_dir + '/' + model_name + '/'
        model_precision = 0
        model_recall = 0
        model_nums = 0
        model_f1 = 0
        model_log = out_dir1 + '/model_log'
        if not os.path.exists(out_dir1):
            os.mkdir(out_dir1)
        # print(out_dir1)
        
        out_dir2 = out_dir1
        if not os.path.exists(out_dir2):
            os.mkdir(out_dir2)
        # print(out_dir2)
        # exit(1)
        for name_num in name_lists:
            
            if name_num == 1:
                name_list = []
                name_str = 'orig'
            elif name_num == 2:
                name_list = name_list_name
                name_str = 'human'
            elif name_num == 3:
                name_list = name_list_fruit
                name_str = 'fruit'
            elif name_num == 4:
                name_list = list(set(['B', 'C', 'D', 'E','F','G','H','I','J','K','L', 'M','N', 'O', 'P', 'Q','R','S','T','U','V','W','X','Y','Z']))
                name_str = 'char'
            else:
                exit(1)
            out_dir_final = out_dir2 + '/' + name_str
            out_result = out_dir_final + '-all_result'
            # print(model_names[0])
            # exit(1)
            # if len(name_list) != 0:
            token_list, sentence_list = parse_sentence(sentence_list)
            
            class MyDataset(Dataset):
                def __init__(self, sentence_list):
                    self.data_list = sentence_list
                    self.length = len(sentence_list)
                def __len__(self):
                    return self.length
                def __getitem__(self, i):
                    return self.data_list[i]
            
            tokenizer = AutoTokenizer.from_pretrained(model_names[0])
            sentence_list, all_list = get_validation_data(sentence_list)
            # print(all_list)
            # print(sentence_list)
            # exit(1)
            df = pd.DataFrame(sentence_list)
            dataset = MyDataset(sentence_list)
            # print(dataset)
            qa_model = pipeline("question-answering", model=model_names[0], tokenizer=model_names[0], use_auth_token=access_token, device=0)
            
            print(len(dataset))
            res_list = list()
            # res_list = tqdm(qa_model(dataset, batch_size=128), total=len(dataset))
            for out in tqdm(qa_model(dataset, batch_size=128), total=len(dataset)):
                # print(type(out))
                # print(out)
                res_list.append(out)
                # break
            print('run finished')
            sentence_list1 = res_list
            if len(name_list) != 0:
                sentence_list1 = parse_back(res_list, token_list)
            res_list = gen_final_list(sentence_list1, all_list)
            for res_dict in res_list:
                with open(out_result, 'a') as f:
                        f.write(json.dumps(res_dict))
                        f.write('\n')
            time_end = time.time()
            print('Time: ' + str(time_end - time_begin))
            
            cp_cmd = 'cp ' + out_result + ' ' + out_dir + '/orig-all_result'
            os.system(cp_cmd)
            gen_csv(out_dir + '/orig-all_result', out_dir + '/orig-all_result.csv')
