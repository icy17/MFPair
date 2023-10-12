import json
import os
import sys
import csv
import copy

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
        api_list.append(api_name.lower())
    return api_list

def generate_threshold(begin, end, span):
    threshold_list = list()
    target = begin
    while 1:
        if target > end:
            break
        
        threshold_list.append(target)
        target += span
    # if end not in threshold_list:
    #     threshold_list.append(end)
    return threshold_list

def if_no_operator(info_list, object_threshold, operator_threshold):
    res_list = list()
    type_list = list()
    max_score = 0
    max_type = ''
    for info in info_list:
        
        question = info['question']
        score = info['score']
        # if operator:
        if question.find('been allocated?') != -1 and score > object_threshold:
            # if score > max_score:
            #     max_score = score
            #     max_type = 'malloc'
           type_list.append('malloc')
        if question.find('been freed?') != -1 and  score > object_threshold:
            # if score > max_score:
            #     max_score = score
            #     max_type = 'free'
           type_list.append('free')
    # type_list.append(max_type)
    operator_type = list()
    for info in info_list:
        question = info['question']
        score = info['score']
        # if 'malloc' in type_list:
        if question.find('performs allocating?') != -1:
            if score > operator_threshold:
                operator_type.append('malloc')
            else:
                if 'malloc' in type_list:
                    res_list.append('malloc')
        # if 'free' in type_list:
        if question.find('performs freeing?') != -1:
            if score > operator_threshold:
                operator_type.append('free')
            else:
                if 'free' in type_list:
                    res_list.append('free')
    return operator_type, res_list

# generate operator sentence list and no operator sentence list:
def split_operator(res_list, object_threshold, operator_threshold):
    no_operator_list = list()
    operator_list = list()
    type_list = ['malloc', 'free']
    for res_json in res_list:
        if res_json['lib'] not in target_libs:
            continue
        out_dict = dict()
        out_dict = res_json
        info_list = res_json['info']
        sentence = info_list[0]['context']
        # if sentence == 'Create a new SSL bufferevent to send its data over an SSL * on a socket':
        #     print(sentence)
        #     # exit(1)
        # else:
        #     continue
        operator_type, res_type = if_no_operator(info_list, object_threshold, operator_threshold)
        # print(operator_type)
        # print(res_type)
        len_no_op = len(res_type)
        res_json['operator_type'] = list()
        res_json['noop_type'] = list()
        if len(operator_type) != 0:
            res_json['operator_type'] = operator_type
            operator_list.append(res_json)
        if len(res_type) != 0:
            res_json['noop_type'] = res_type
            no_operator_list.append(res_json)
        # res_json['noop_type'] = res_type
        # if len_no_op == 2:
        #     res_json['operator_type'] = res_type
        #     no_operator_list.append(res_json)
        # elif len_no_op == 0:
        #     res_json['operator_type'] = res_type
        #     operator_list.append(res_json)
        # else:
        #     res_json['operator_type'] = res_type
        #     no_operator_list.append(res_json)
        #     tmp_json = dict()
        #     tmp_json = res_json
        #     operator_type = list()
        #     for api_type in type_list:
        #         if api_type not in res_type:
        #             operator_type.append(api_type)
        #     tmp_json['operator_type'] = operator_type
        # operator_list.append(res_json)
    return operator_list, no_operator_list

def match_api_list(operator, api_list):
    # print(api_list)
    # exit(1)
    operator = operator.replace('(3)', '')
    operator = operator.replace('()', '')
    operator = operator.replace('(3', '')
    operator = operator.replace('(z', '')
    operator = operator.lower()
    operator = operator.replace(',', ' ')
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
            # if token.find('dbus') != -1:
            #     if '_' + token in match_list:
            #         out_match_list.append(token)
            out_missing_list.append(token)
    return out_match_list, out_missing_list

def get_gt_op_dict(res_list, gt_api_list):
    api_dict = dict()
    for line in res_list:
        if line['lib'] not in target_libs:
            continue
        api = line['apiname'].lower().replace('()', '').strip()
        # if 'API-type(f)' not in line.keys():
        #     continue
        api_type = line['GT-op']
        if api_type == '' or api_type == '/' or api_type == '?':
            # print(line)
            continue
            # exit(1)
        if api == '':
            continue
        if api not in gt_api_list:
            continue
        if api in api_dict.keys():
            if api_dict[api] != api_type:
                print(line)
                print(api)
                print('format gt api wrong!')
                print('173')
                # exit(1)
        api_dict[api] = api_type
    return api_dict

def get_format_gt_dict(res_list, gt_api_list):
    api_dict = dict()
    for line in res_list:
        if line['lib'] not in target_libs:
            continue
        api = line['Format-API'].lower().replace('()', '').strip()
        if 'API-type(f)' not in line.keys():
            continue
        api_type = line['API-type(f)']
        if api_type == '':
            # print(line)
            continue
            # exit(1)
        if api == '':
            continue
        if api not in gt_api_list:
            continue
        if api in api_dict.keys():
            if api_dict[api] != api_type:
                print(line)
                print(api)
                print('format gt api wrong!')
                print('173')
                # exit(1)
        api_dict[api] = api_type
    return api_dict

def get_gt_dict(gt_list, gt_api_list):
    api_dict = dict()
    gt_api = list()
    missing_dict = dict()
    for line in gt_list:
        if line['lib'] not in target_libs:
            continue
        api = line['QA-API']
        if api == '':
            continue
        tmp_list, missing_api = match_api_list(api, gt_api_list)
        for item in missing_api:
            missing_dict[item] = 1
        # for item in missing_api:
        #     with open('./miss_api', 'a') as f:
        #         f.write(item + '\n')
        api_type = line['API-type']
        if api_type == '':
            # continue
            print(line)
        for api in tmp_list:
            if api != '':
                if api in api_dict.keys():
                    if api_dict[api] != api_type:
                        print(api)
                        print(api_type)
                        print('qa gt api wrong!')
                        # exit(1)
                api_dict[api] = api_type
                gt_api.append(api)
    return api_dict

# return all
def parse_qa_api(operator_list, gt_api_list, operator_threshold):
    # api_dict = dict()
    
    
    api_dict_res = dict()
    api_type_dict = dict()
    api_sentence_dict = dict()
    api_info_dict = dict()
    # print('in parse api before first parse')
    # print(operator_threshold)
    # print()
    i = 0
    for line in operator_list:
        i += 1
        # print(i)
        if line['lib'] not in target_libs:
            continue
        info_list = line['info']
        # right_answer = line['QA-API']
        # if i == 148:
        # print(len(info_list))
        for info in info_list:
            # if i == 148:
            # print(info)
            # exit(1)
            res_type = ''
            question = info['question']
            answer = info['answer'].lower()
            if 'context' in info.keys():
                sentence = info['context']
            else:
                sentence = info['sentence']
            # get_api
            answer_api = list()
            for key in gt_api_list:
                key = key.lower()
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
            # print(answer_api)
            # exit(1)
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
            if score > operator_threshold:
                if question.find('allocating') != -1:
                    res_type = 'malloc'
                elif question.find('freeing') != -1:
                    res_type = 'free'
                # print('before answer_api')
                # print(len(info_list))
                if res_type == '':
                    continue
                # print(res_type)
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
        
    # print(api_dict_res)
    for key in api_dict_res.keys():
        tmp_dict = dict()
        tmp_dict['api'] = key
        if api_dict_res[key] == 'all':
            if api_type_dict[key + '-free'] > api_type_dict[key + '-malloc']:
                api_dict_res[key] = 'free'
            else:
                api_dict_res[key] = 'malloc'
    return api_dict_res, api_info_dict

def parse_format_api(no_op_list, gt_api_list):
    # api_dict = dict()
    api_res_dict = dict()
    api_info_dict = dict()
    for line in no_op_list:
        # if line['Format-API'] == '':
        #     continue
        # print(line)
        # print('in for')
        if line['lib'] not in target_libs:
            continue
        apiname = line['apiname'].lower().replace('()', '').strip()
        
        if apiname == '':
            continue
        if 'close_api' in line.keys() and (apiname.find('[') != -1 or line['lib'] == 'ldap'):
            # print(apiname)
            # TODO: apiname=close api
            apiname = line['close_api']
            # continue
        # print('before continue')
        # print(apiname)
        if apiname == '':
            continue
        if apiname not in gt_api_list:
            # print('not in at api')
            continue
        # print(line)
        info_list = line['info']
        type_list = line['noop_type']
        # if apiname == 'av_dict_set':
        #     print(info_list[0]['context'])
        #     print(type_list)
        # if len(type_list) == 2:
        #     continue
        # if len(type_list) != 0:
        #     print(f'api: {apiname}, no operator: {str(type_list)}')
        # line['no_operator'] = type_list
        # parse_changed_list.append(line)
        # if apiname in api_res_dict.keys():
        #     continue
        for api_type in type_list:
            tmp_dict = dict()
            if apiname not in api_res_dict.keys():
                api_res_dict[apiname] = api_type
                tmp_list = list()
                tmp_list.extend(info_list)
                api_info_dict[apiname + '-' + api_type] = tmp_list
            elif api_type != api_res_dict[apiname]:
                if api_res_dict[apiname] != 'all':
                    api_res_dict[apiname] = 'all'
                    tmp_list = list()
                    tmp_list.extend(info_list)
                    api_info_dict[apiname + '-' + api_type] = tmp_list
                else:
                    api_info_dict[apiname + '-' + api_type].extend(info_list)
            else:
                api_info_dict[apiname + '-' + api_type].extend(info_list)
            # print(api_res_dict)
            # print(api_info_dict)
    return api_res_dict, api_info_dict

def parse_result(final_api_dict, gt_list, threshold_dict, out_dir, out_log_path):
    out_list = list()
    return 

# parse_notably(qa_api_dict, qa_info_dict, format_api_dict, format_info_dict, gt_all_dict)
def parse_notably(qa_api_dict, qa_info_dict, format_api_dict, format_info_dict):
    api_dict = dict()
    api_info_dict = dict()
    type_dict = dict()
    for key in qa_api_dict.keys():
        api_dict[key] = qa_api_dict[key]
        api_info_dict[key + '-' + api_dict[key]] = qa_info_dict[key + '-' + qa_api_dict[key]]
        type_dict[key] = 'QA'
    # qa_num = len(api_dict)
    for key in format_api_dict.keys():
        if key in api_dict.keys():
            continue
        api_dict[key] = format_api_dict[key]
        api_type = format_api_dict[key]
        if api_type == 'all':
            api_info_dict[key + '-free'] = format_info_dict[key + '-free']
            api_info_dict[key + '-malloc'] = format_info_dict[key + '-malloc']
        else:
            api_info_dict[key + '-' + api_dict[key]] = format_info_dict[key + '-' + format_api_dict[key]]
        type_dict[key] = 'format'
    return api_dict, api_info_dict, type_dict

def write_noop(op_list, noop_list, prefix, op_log, threshold_dict):
    op_path = prefix + '-op'
    noop_path = prefix + '-noop'
    for op in op_list:
        with open(op_path, 'a') as f:
            f.write(json.dumps(op))
            f.write('\n')
    for noop in noop_list:
        with open(noop_path, 'a') as f:
            f.write(json.dumps(noop))
            f.write('\n')
    operator_len = len(op_list)
    no_len = len(noop_list)
    threshold_dict['operator_num'] = operator_len
    threshold_dict['no-operator-num'] = no_len
    with open(op_log, 'a') as f:
        f.write(json.dumps(threshold_dict))
        f.write('\n')

def cal_F1(precision, recall):
    if precision + recall == 0:
        F1  = 0
    else:
        F1 = 2*(precision * recall) / (precision + recall)
    return F1

def write_api_result(api_dict_res, gt_dict, api_info_dict, out_dir_prefix, threshold_dict, out_log_path):
    all_res_path = out_dir_prefix + '-all_result'
    fn_path = out_dir_prefix + '-fn_result'
    fp = 0
    tp = 0
    precision = 0
    recall = 0
    tp_dict = dict()
    fn_list = list()
    out_list = list()
    api_dict = gt_dict
    # print(api_dict_res)
    # cal FP:
    for key in api_dict_res.keys():
        tmp_dict = dict()
        tmp_dict['api'] = key
        res_type = api_dict_res[key]
        item_dict =dict()
        item_dict['api'] = key
        item_dict['res_type'] = api_dict_res[key]
        # if api_dict_res[key] == 'all':
        #     if api_type_dict[key + '-free'] > api_type_dict[key + '-malloc']:
        #         api_dict_res[key] = 'free'
        #     else:
        #         api_dict_res[key] = 'malloc'
        if key not in api_dict.keys():
            item_dict['res'] = 'FP'
            if res_type == 'all':
                fp += 2
                item_dict['info'] = api_info_dict[key + '-' + 'malloc']
                item_dict['info'].extend(api_info_dict[key + '-' + 'free'])
                tmp_dict = copy.deepcopy(item_dict)
                tmp_dict['res_type'] = 'free'
                item_dict['res_type'] = 'malloc'
                out_list.append(tmp_dict)
            else:
                fp += 1
                item_dict['info'] = api_info_dict[key + '-' + res_type]
            out_list.append(item_dict)
            continue
        if api_dict_res[key] == api_dict[key]:
            tp_dict[key] = api_dict[key]
            tp += 1
            tmp_dict['res'] = 'TP'
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['res_type'] =  api_dict_res[key]
            # print(api_info_dict[key + '-' + api_dict[key]])
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict[key]]
            out_list.append(tmp_dict)

        elif api_dict_res[key] == 'all':
            tp += 1
            # tp_dict[key] = api_dict[key]
            tmp_dict['res'] = 'TP'
            # print(api_dict)
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['res_type'] =  api_dict[key]
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
            # wrong_dict['sentences'] = api_sentence_dict[key + '-' + wrong_type]
            wrong_dict['res_type'] = wrong_type
            wrong_dict['info'] = api_info_dict[key + '-' + wrong_type]
            out_list.append(wrong_dict)
        else:
            fp += 1
            tmp_dict['res'] = 'FP'
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict_res[key]]
            tmp_dict['res_type'] =  api_dict_res[key]
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict_res[key]]
            out_list.append(tmp_dict)
    # cal FN:
    for key in api_dict.keys():
        if key not in api_dict_res.keys():
            fn_list.append(key)
        else:
            if api_dict_res[key] != 'all' and api_dict_res[key] != api_dict[key]:
                fn_list.append(key)
    for item in out_list:
        with open(all_res_path, 'a') as f:
            f.write(json.dumps(item))
            f.write('\n')
    for api in fn_list:
        # if api not in hit_fn:
        out_dict = dict()
        out_dict['miss_api'] = api
        out_dict['sentence'] = ''
        out_dict['info'] = 'FN'
        with open(fn_path, 'a') as f:
            f.write(json.dumps(out_dict))
            f.write('\n')
    # cal score:
    fn = len(api_dict) - tp
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if len(api_dict) == 0:
        recall = 0
    else:
        
        recall = tp / len(api_dict)
    f1 = cal_F1(precision, recall)
    threshold_dict['Precision'] = precision
    threshold_dict['Recall'] = recall
    threshold_dict['FN'] = fn
    threshold_dict['FP'] = fp
    threshold_dict['F1'] = f1
    threshold_dict['TP'] = tp
    threshold_dict['All API'] = len(api_dict)
    with open(out_log_path, 'a') as f:
        f.write(json.dumps(threshold_dict))
        f.write('\n')
    if os.path.exists(all_res_path):
        
        gen_csv(all_res_path, all_res_path + '.csv')
        
def write_final_result(api_dict_res, gt_dict, api_info_dict, type_dict, out_dir_prefix, threshold_dict, out_log_path):
    all_res_path = out_dir_prefix + '-all_result'
    fn_path = out_dir_prefix + '-fn_result'
    fp = 0
    tp = 0
    precision = 0
    recall = 0
    tp_dict = dict()
    fn_list = list()
    out_list = list()
    api_dict = gt_dict
    qa_tp = 0
    format_tp = 0
    # print(api_dict_res)
    # cal FP:
    for key in api_dict_res.keys():
        tmp_dict = dict()
        tmp_dict['api'] = key
        tmp_dict['method'] = type_dict[key]
        res_type = api_dict_res[key]
        item_dict =dict()
        item_dict['api'] = key
        item_dict['res_type'] = api_dict_res[key]
        item_dict['method'] = type_dict[key]
        # if api_dict_res[key] == 'all':
        #     if api_type_dict[key + '-free'] > api_type_dict[key + '-malloc']:
        #         api_dict_res[key] = 'free'
        #     else:
        #         api_dict_res[key] = 'malloc'
        if key not in api_dict.keys():
            item_dict['res'] = 'FP'
            if res_type == 'all':
                fp += 2
                item_dict['info'] = api_info_dict[key + '-' + 'malloc']
                item_dict['info'].extend(api_info_dict[key + '-' + 'free'])
                tmp_dict = copy.deepcopy(item_dict)
                tmp_dict['res_type'] = 'free'
                item_dict['res_type'] = 'malloc'
                out_list.append(tmp_dict)
            else:
                fp += 1
                item_dict['info'] = api_info_dict[key + '-' + res_type]
            out_list.append(item_dict)
            continue
        if api_dict_res[key] == api_dict[key]:
            tp_dict[key] = api_dict[key]
            tp += 1
            if type_dict[key] == 'QA':
                qa_tp += 1
            elif type_dict[key] == 'format':
                format_tp += 1
            tmp_dict['res'] = 'TP'
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['res_type'] =  api_dict_res[key]
            # print(api_info_dict[key + '-' + api_dict[key]])
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict[key]]
            out_list.append(tmp_dict)

        elif api_dict_res[key] == 'all':
            tp += 1
            if type_dict[key] == 'QA':
                qa_tp += 1
            elif type_dict[key] == 'format':
                format_tp += 1
            # tp_dict[key] = api_dict[key]
            tmp_dict['res'] = 'TP'
            # print(api_dict)
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict[key]]
            tmp_dict['res_type'] =  api_dict[key]
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict[key]]
            out_list.append(tmp_dict)
            fp += 1
            wrong_dict = dict()
            wrong_dict['api'] = key
            wrong_dict['res'] = 'FP'
            wrong_dict['method'] = type_dict[key]
            if api_dict[key] == 'free':
                wrong_type = 'malloc'
            else:
                wrong_type = 'free'
            # wrong_dict['sentences'] = api_sentence_dict[key + '-' + wrong_type]
            wrong_dict['res_type'] = wrong_type
            wrong_dict['info'] = api_info_dict[key + '-' + wrong_type]
            out_list.append(wrong_dict)
        else:
            fp += 1
            tmp_dict['res'] = 'FP'
            # tmp_dict['sentences'] = api_sentence_dict[key + '-' + api_dict_res[key]]
            tmp_dict['res_type'] =  api_dict_res[key]
            tmp_dict['info'] = api_info_dict[key + '-' + api_dict_res[key]]
            out_list.append(tmp_dict)
    # cal FN:
    for key in api_dict.keys():
        if key not in api_dict_res.keys():
            fn_list.append(key)
    for item in out_list:
        with open(all_res_path, 'a') as f:
            f.write(json.dumps(item))
            f.write('\n')
    for api in fn_list:
        # if api not in hit_fn:
        out_dict = dict()
        out_dict['miss_api'] = api
        out_dict['sentence'] = ''
        out_dict['info'] = 'FN'
        with open(fn_path, 'a') as f:
            f.write(json.dumps(out_dict))
            f.write('\n')
    # cal score:
    fn = len(api_dict) - tp
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = tp / len(api_dict)
    f1 = cal_F1(precision, recall)
    threshold_dict['Precision'] = precision
    threshold_dict['Recall'] = recall
    threshold_dict['FN'] = fn
    threshold_dict['FP'] = fp
    threshold_dict['QA-TP'] = qa_tp
    threshold_dict['Format-TP'] = format_tp
    threshold_dict['All-tp'] = tp
    threshold_dict['F1'] = f1
    
    threshold_dict['All API'] = len(api_dict)
    with open(out_log_path, 'a') as f:
        f.write(json.dumps(threshold_dict))
        f.write('\n')
    if os.path.exists(all_res_path):
        gen_csv(all_res_path, all_res_path + '.csv')

def combine_gt_dict(qa_gt_dict, format_gt_dict, op_dict):
    final_api_dict = dict()
    for key in qa_gt_dict.keys():
        if key in final_api_dict.keys():
            if final_api_dict[key] != qa_gt_dict[key]:
                print('qa gt wrong?')
                # exit(1)
        else:
            final_api_dict[key] = qa_gt_dict[key]
    for key in format_gt_dict.keys():
        if key in final_api_dict.keys():
            if final_api_dict[key] != format_gt_dict[key]:
                print('format gt wrong?')
                # exit(1)
        else:
            final_api_dict[key] = format_gt_dict[key]
    for key in op_dict.keys():
        if key in final_api_dict.keys():
            if final_api_dict[key] != op_dict[key]:
                print('gt op wrong?')
                final_api_dict[key] = op_dict[key]
                # exit(1)
        else:
            final_api_dict[key] = op_dict[key]
    return final_api_dict

def get_api_dict(api_path):
    out_dict = dict()
    in_list = read_json(api_path)
    for line in in_list:
        lib = line['lib']
        if lib in out_dict.keys():
            out_dict[lib].append(line['apiname'])
        else:
            out_dict[lib] = [line['apiname']]
    return out_dict

def get_api_lib(apiname, api_dict):
    for key in api_dict.keys():
        if apiname == '':
            return ''
        else:
            if apiname in api_dict[key]:
                return key
    return ''

def get_obj(api, api_type, info_dict):
    question_dict = {'free': 'What has been freed?', 'malloc': 'What has been allocated?'}
    api_key = api + '-' + api_type
    info = info_dict[api_key]
    question = question_dict[api_type]
    for item in info:
        if item['question'] == question:
            obj = item['answer']
            return obj
    # return info


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 ./MF_identify.py <in_dir>')
        in_dir = './libzip_re'
        exit(1)
    else:
        in_dir = sys.argv[1] + '/'
        if not os.path.exists(in_dir):
            print(f'in dir {in_dir} not exists!')
            exit(1)
    # Change: gt_api_path file contain all api.
    gt_api_path = './API-list'
    # Change: Analyse target
    target_libs = ['libpcap']
    
    res_path = in_dir + 'orig-all_result'
    
    res_list = read_json(res_path)
    gt_api_list = list(set(get_api_list(gt_api_path)))
    api_dict = get_api_dict(gt_api_path)
    for operator_threshold in [0.2]:
        for object_threshold in [0.5]:
            print(operator_threshold)
            print(object_threshold)
            operator_threshold_str = format(operator_threshold, '.3f')
            object_threshold_str = format(object_threshold, '.3f')
            
            file_name = '/' + operator_threshold_str + '-' + object_threshold_str
            threshold_dict = dict()
            tmp_list = dict()
            threshold_dict['operator_threshold'] = operator_threshold_str
            threshold_dict['object_threshold'] = object_threshold_str
            tmp_list = threshold_dict
            operator_list, no_operator_list = split_operator(res_list, object_threshold, operator_threshold)
            threshold_dict = tmp_list
            qa_api_dict, qa_info_dict = parse_qa_api(operator_list,gt_api_list, operator_threshold)
            threshold_dict = tmp_list
            
            format_api_dict, format_info_dict = parse_format_api(no_operator_list, gt_api_list)
            threshold_dict = tmp_list
            final_api_dict, final_info_dict, type_dict = parse_notably(qa_api_dict, qa_info_dict, format_api_dict, format_info_dict)
            out_path = in_dir + 'final_api'
            for key in final_api_dict:
                if final_api_dict[key] == 'all':
                    out_dict = dict()
                    out_dict['type'] = 'free'
                    out_dict['apiname'] = key
                
                    info = get_obj(key, 'free', final_info_dict)
                    lib = get_api_lib(key, api_dict)
                    if lib == '':
                        print('error! lib is empty!')
                        exit(1)
                    out_dict['lib'] = lib
                    out_dict['obj'] = info
                    # print(info)
                    with open(out_path, 'a') as f:
                        f.write(json.dumps(out_dict))
                        f.write('\n')
                    out_dict['type'] = 'malloc'
                    info = get_obj(key, 'malloc', final_info_dict)
                    out_dict['lib'] = lib
                    out_dict['obj'] = info
                    with open(out_path, 'a') as f:
                        f.write(json.dumps(out_dict))
                        f.write('\n')
                else:
                    out_dict = dict()
                    # out_dict[key] = 
                    out_dict['type'] = final_api_dict[key]
                    out_dict['apiname'] = key
                    info = get_obj(key, final_api_dict[key], final_info_dict)
                    lib = get_api_lib(key, api_dict)
                    if lib == '':
                        print('error! lib is empty!')
                        exit(1)
                    out_dict['lib'] = lib
                    out_dict['obj'] = info
                    
                    with open(out_path, 'a') as f:
                        f.write(json.dumps(out_dict))
                        f.write('\n')
           