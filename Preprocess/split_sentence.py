import json
import spacy
import re
import string
nlp = spacy.load('en_core_web_sm')

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



def get_close_api(struct_path, doc, sentence):
    # doc = doc.strip('\n')
    doc = doc.replace('\n', ' ')
    index = doc.find(sentence)
    doc_small = doc[0: index + len(sentence) + 1]
    print(doc_small)
    doc_nlp = nlp(doc_small)
    # print(sentence)
    close_api = ''
    close_sentence = ''
    for sentence_small in doc_nlp.sents:
        # print(sentence_small)
        sentence_list = str(sentence_small).split('.')
        # print(sentence_list)
        for sentence_final in sentence_list:
            # print(sentence_final)
            if len(sentence_final.strip('\n')) == 0:
                continue
            # print('sentence:')
            sentence_final = sentence_final.strip(' ')
            # print(sentence_final)
            sentence_final = sentence_final.replace('(', '')
            sentence_final = sentence_final.replace(')', '')
            words = sentence_final.split(' ')
            
            word = words[0].strip(' ')
            # print('first word::')
            # print(word)
            # for api in api_list:
            if word in api_list:
                close_api = word
                close_sentence = sentence_final
                # break
                # break
    # print('1')
    return close_api, close_sentence
            # print(sentence_final)
    
    # print(doc_small)



def keyword_search(malloc_keyword, free_keyword, sentence_list, out_path, struct_path):
    # nlp = spacy.load('en_core_web_sm')
    num = len(sentence_list)
    i = 0
    for sentence_json in sentence_list:
        i += 1
        print(str(i) + '/' + str(num))
        sentence_type = sentence_json['sentence_type']
        sentence_orig = sentence_json['sentence']
        sentence = sentence_json['sentence'].lower()
        doc = nlp(sentence)
        
        list1=[]
        for token in doc:
            # token=token.lemma_   #词干化
            token = str(token)
            if token not in string.punctuation: #去除所有标点
                list1.append(token)
        malloc_keyword_list = list()
        malloc_orig_keyword_list = list()
        free_keyword_list = list()
        free_orig_keyword_list = list()
        type = ''
        close_api = ''
        close_sentence = ''
        # print('new sentence')
        for malloc_dict in malloc_keyword:
            for keyword in malloc_dict.keys():
                for search_word in malloc_dict[keyword]:
                    if search_word in list1:
                        malloc_keyword_list.append(keyword)
                        malloc_orig_keyword_list.append(search_word)
                        type = 'malloc'
                        close_api, close_sentence= get_close_api(struct_path, sentence_json['doc'], sentence_orig)
                        
                        # print('new sentence')
                        # print('malloc:')
                        # print(keyword_list)
                        break
        for free_dict in free_keyword:
            for keyword in free_dict.keys():
                for search_word in free_dict[keyword]:
                    if search_word in list1:
                        free_keyword_list.append(keyword)
                        free_orig_keyword_list.append(search_word)
                        close_api, close_sentence= get_close_api(struct_path, sentence_json['doc'], sentence_orig)
                        if type == 'malloc' or type == 'all':
                            type = 'all'
                            # print('all:')
                            # print(keyword_list)
                        else:
                            type = 'free'
                            # print('free:')
                            # print(keyword_list)
                        break
        if type != '':
            keywords_dict = dict()
            orig_keywords_dict = dict()
            keywords_dict['malloc'] = malloc_keyword_list
            keywords_dict['free'] = free_keyword_list
            orig_keywords_dict['malloc'] = malloc_orig_keyword_list
            orig_keywords_dict['free'] = free_orig_keyword_list
            sentence_json['type'] = type
            sentence_json['keywords'] = keywords_dict
            sentence_json['close_api'] = close_api
            sentence_json['close_sentence'] = close_sentence
            sentence_json['orig_keywords'] = orig_keywords_dict
            sentence_json['sentence_type'] = sentence_type
            del sentence_json['doc']
            with open(out_path, 'a') as f:
                f.write(json.dumps(sentence_json))
                f.write('\n')
        else:
            with open(out_path + 'no', 'a') as f:
                f.write(json.dumps(sentence_json))
                f.write('\n')
                # f.write(sentence_orig + '\n')
            # print(sentence_json)
        # if len(close_api) != 0:
        #     print(close_api)
        #     break

def get_first_desc(doc):
    sentence = doc['desc']
    sentence = sentence.strip(' ')
    sentence = sentence.strip('\n')
    sentences = sentence.split('.')
    first_one = ''
    for sentence in sentences:
        if len(sentence) != 0:
            first_one = sentence
            break
    return first_one


def split_sentence(in_list, out_sentence):
    # nlp = spacy.load('en_core_web_sm')
    for content in in_list:
        apiname = content['apiname']
        url = content['link']
        lib = content['lib']
        doc = content['doc']['all_doc']
        if doc != '' and doc[0] == '[':
            doc = doc.replace('[','', 1)
            # doc[0] = ' '
        if content['doc']['desc'] != '' and content['doc']['desc'][0] == '[':
            content['doc']['desc'] = content['doc']['desc'].replace('[', ' ', 1)
        # doc = doc.replace('[','', 1)
        # doc = doc.replace(']', '')
        # re_h = re.compile('</?\w+[^>]*>')
        if content['doc']['return'] != '':
            # content['doc']['return'] = re_h.sub('', content['doc']['return'])
            content['doc']['return'] = content['doc']['return'].replace('\n', ' ')
        if content['doc']['parameter'] != '':
            # content['doc']['parameter'] = re_h.sub('', content['doc']['parameter'])
            content['doc']['parameter'] = content['doc']['parameter'].replace('\n', ' ')
        if content['doc']['desc'] != '':
            # content['doc']['parameter'] = re_h.sub('', content['doc']['parameter'])
            content['doc']['desc'] = content['doc']['desc'].replace('\n', ' ')
        # doc = re_h.sub('', doc)
        doc = doc.replace('\n', ' ')
        content['doc']['all_doc'] = doc
        if len(doc) == 0 or doc == None:
            continue
        first_sentence = get_first_desc(content['doc'])
        # print(first_sentence)
        # exit(1)
        # print(doc)
        doc_nlp = nlp(doc)
        for sentence_nlp in doc_nlp.sents:
            out_dict = dict()
            sentence_nlp = str(sentence_nlp)
            sentences = sentence_nlp.split('.')
            for sentence in sentences:
                if sentence == None or len(sentence) == 0:
                    continue
            # print(sentence)
            # sentence = sentence.replace('</tt>', '')
            # sentence = sentence.replace('<p>', '')
            # sentence = sentence.replace('<em>', '')
            # sentence = sentence.replace('</em>', '')
                if type(apiname) != type(str):
                    out_dict['apiname'] = apiname
                else:
                    out_dict['apiname'] = apiname.strip('\n')
                out_dict['link'] = url
                out_dict['lib'] = lib
                out_dict['sentence'] = str(sentence)
                out_dict['doc'] = content['doc']['desc']
                out_dict['sentence_type'] = ''
                out_dict['first_sentence'] = 0
                # print(content['doc']['return'])
                # print('matched sentence:::')
                # print(sentence)
                # if sentence.find('Upon successful') != -1:
                #     print(list(sentence))
                #     print(list(content['doc']['return']))
                #     print(content['doc']['return'].find(sentence))
                if content['doc']['parameter'].find(sentence) != -1:
                    out_dict['sentence_type'] = 'parameter'
                elif content['doc']['return'].find(sentence) != -1:
                    out_dict['sentence_type'] = 'return'
                elif content['doc']['desc'].find(sentence) != -1:
                    out_dict['sentence_type'] = 'desc'
                    # print(first_sentence)
                    # print(sentence)
                    # exit(1)
                    if sentence.strip(' ') == first_sentence:
                        out_dict['first_sentence'] = 1
                tmp = sentence.strip(' ')
                if len(tmp) == 0 or len(sentence.strip('\n')) == 0 or len(sentence.strip('\t')) == 0:
                    continue
                with open(out_sentence,'a') as f:
                    f.write(json.dumps(out_dict))
                    f.write('\n')
            


if __name__ == '__main__':
    in_dir = '../example_data/'
    out_dir = '../QA/'
    libs = ['libpcap']
    
    
    for in_lib in libs:
        out_sentence = out_dir + in_lib
        # struct_path = in_dir + '/struct_' + in_lib + '.json'
        in_list = read_json(in_dir + in_lib + '.json')
        # in_list = read_json(in_dir + in_lib)
        final_out = out_sentence + '-keywords'
        # struct_json = read_json(struct_path)
        api_list = list()
        for struct in in_list:
            api_name = struct['apiname']
            api_list.append(api_name)
        split_sentence(in_list, out_sentence)
        



