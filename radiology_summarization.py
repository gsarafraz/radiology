from preprocessing.Preprocessing import Preprocessing
import pandas as pd
import sys
import numpy as np
import nltk
from nltk.util import ngrams
import sys


data_path = './data/'
preprocessing = Preprocessing(data_folder = data_path)

diseases = list(pd.read_csv(data_path + 'diseases_third_version.csv')['Term'])
base_types_diseases = list(pd.read_csv(data_path + 'diseases_third_version.csv')['baseType'])
body_parts = list(pd.read_csv(data_path + 'body_parts_second_version.csv')['Term'])
diseases = list([preprocessing.complete_normalization(x) for x in diseases])[1:]
base_types_diseases = list([preprocessing.complete_normalization(x) for x in base_types_diseases])[1:]
body_parts = list(set([preprocessing.complete_normalization(x) for x in body_parts]))[1:]

diseases_df = pd.DataFrame({'term':diseases, 'baseType': base_types_diseases})
body_parts_df = pd.DataFrame({'term':body_parts})

diseases_df_tmp = pd.DataFrame()

for index, row in diseases_df.iterrows():
    term = row['term']
    baseType = row['baseType']
    words = term.split()
    for word in words:
        diseases_df_tmp2 = {'term': [term.replace(word, word + 'های'), term.replace(word, word + 'ی'), term.replace(word, word + 'ها')], 'baseType': [baseType for i in range(3)]}
        diseases_df_tmp2 = pd.DataFrame(data=diseases_df_tmp2)
        diseases_df_tmp = pd.concat([diseases_df_tmp, diseases_df_tmp2], axis=0)

diseases_df = pd.concat([diseases_df_tmp, diseases_df], axis=0)

def count_subwords(word):
    return len(word.split())

diseases_df['num_subwords'] = diseases_df['term'].apply(lambda x: count_subwords(x))
body_parts_df['num_subwords'] = body_parts_df['term'].apply(lambda x: count_subwords(x))

max_subwords = max(body_parts_df['num_subwords'].max(), diseases_df['num_subwords'].max())



neg_verbs = ['نشد', 'نمیشود','نمی‌شود','نمی شود', 'ندارند','نبود','نگردد','نباشد', 'نگردید', 'نمیتواند', 'نمی‌تواند','ندارد','نمی تواند','نیست','نمی']

def find_polarity(sent):
    
    for verb in neg_verbs:
        if(verb in sent):
            return 'منفی'
    return 'مثبت'


def find_shared_body_parts_and_diseases(sent):
    
    shared_body_parts = []
    shared_disease = []
    shared_disease_bt = []
    count_subwords = max_subwords
    
    while(count_subwords > 0):
        
        n_grams = ngrams(sequence = sent.split(), n = count_subwords)
        grams_list = []
        for grams in n_grams:
            grams_list.append(' '.join(grams))
    
        grams_list = list(set(grams_list))
        
        shared_bodyparts_tmp  = list(body_parts_df[(body_parts_df['num_subwords']  == count_subwords) & (body_parts_df['term'].isin(grams_list))]['term'])
        shared_diseases_tmp_bt = list(diseases_df[(diseases_df['num_subwords'] == count_subwords) & (diseases_df['term'].isin(grams_list))]['baseType'])
        shared_diseases_tmp = list(diseases_df[(diseases_df['num_subwords'] == count_subwords) & (diseases_df['term'].isin(grams_list))]['term'])
        
        for word in shared_bodyparts_tmp:
            sent = sent.replace(word, '')

for word in shared_diseases_tmp:
    sent = sent.replace(word, '')
        
        shared_body_parts += shared_bodyparts_tmp
        shared_disease += shared_diseases_tmp
        shared_disease_bt += shared_diseases_tmp_bt
        count_subwords -= 1
    
    return shared_body_parts, shared_disease, shared_disease_bt

def find_prev_tag(sentenceslist, sentindx):
    
    tag = None
    sentenceslist = sentenceslist[:sentindx]
    for i in range(1,sentindx+1):
        sent = sentenceslist[-1*i]
        if('tag' in sent):
            intersect = list(set(sent.split()).intersection(body_parts))
            if(len(intersect) > 0):
                tag = intersect[0]
            break
    return tag


def bodypart_disease_alignment(sentenceslist):
    all_results = []
    
    for sentindx, sent in enumerate(sentenceslist):
        result = []
        shared_bodyparts, shared_diseases, shared_diseases_bt = find_shared_body_parts_and_diseases(sent)
        polarity = find_polarity(sent)
        
        if(len(shared_diseases) != 0):
            
            if(len(shared_bodyparts) == 0):
                tag = find_prev_tag(sentenceslist, sentindx)
                for index, disease in enumerate(shared_diseases):
                    result.append({'عارضه':shared_diseases_bt[index],'عضو':tag, 'رخداد یا عدم رخداد': polarity})
        
            else:
                index_bps = [sent.index(x) for x in shared_bodyparts]
                index_diseases = [sent.index(x) for x in shared_diseases]
                
                for index, d_index in enumerate(index_diseases):
                    
                    tmp_arr = np.abs(np.array(index_bps) - d_index)
                    bp_min_index = np.argmin(tmp_arr)
                    result.append({'عارضه':shared_diseases_bt[index],'عضو':shared_bodyparts[bp_min_index], 'رخداد یا عدم رخداد': polarity})
    
    if(len(result) > 0):
        all_results.append(result)
    return all_results



def print_result(report):
    count = 1
    all_results = ''
    for element in report:
        all_results += str(count) + ") " + str(element) + '\n'
        count += 1
    all_results = all_results.replace('[','').replace(']','').replace('{','').replace('}','').replace('\'','').replace(',', ',   ')
    return all_results

def summarize_radiology_report(report):
    
    report = preprocessing.cleaner(report)
    report = preprocessing.complete_normalization(report)
    report = preprocessing.sentence_tokenizer(report)
    
    print(print_result(bodypart_disease_alignment(report)))


if __name__ == "__main__":
    
    report = sys.argv[1]
    summarize_radiology_report(report)