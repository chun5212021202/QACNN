import glob
import re
from nltk.stem.snowball import SnowballStemmer
import json
import numpy as np


snowball = SnowballStemmer("english")
re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')



def processLine(line):
    
    line = line.lower()
    line = re_alphanumeric.sub('', line)
    line = re_multispace.sub(' ', line)
    
    return line

def processWord(word):
    word = word.lower()
    word = re_alphanumeric.sub('', word)
    word = re_multispace.sub(' ', word)
    word = snowball.stem(word)
    return word

def listSplitPlot(fd):
    lines = []
    while True:
        line = fd.readline()
        if line == '':
            break
        line = line.strip('\r\n')
        if len(line) > 1:
            line = processLine(line)
            line = line.split()
            lines.append(line)
    return lines


def compute():
    a=0
    b=0

    train = 0
    val = 0
    test = 0

    register = json.load(open('../word_vec/register.json'))
    QA = json.load(open('../raw_data/qa.json'))
    

    QA_train_vec = []
    QA_val_vec = []
    QA_test_vec = []
    with open('../word_vec/glove.42B.300d.json') as glove_file:
        glove_dict = json.load(glove_file)

        for qa in QA:
            qa_vec = {}
            qa_vec['qid'] = qa['qid']
            qa_vec['correct_index'] = qa['correct_index']
            qa_vec['imdb_key'] = qa['imdb_key']
            qa_vec['video_clips'] = qa['video_clips']

            question = processLine(qa['question']).split()
            question_vec = []
            for word in question:
                a+=1
                word_stem = snowball.stem(word)
                if word in glove_dict.keys():
                    b+=1
                    word_vec = glove_dict[word]
                    question_vec.append(word_vec)
                elif word_stem in glove_dict.keys():
                    b+=1
                    word_vec = glove_dict[word_stem]
                    question_vec.append(word_vec)                         
                elif word in register.keys():
                    word_vec = register[word]
                    question_vec.append(word_vec)
                else:
                    question_vec.append([0]*300)

            qa_vec['question'] = question_vec


            option_5 = [ processLine(X).split() for X in qa['answers'] ]
            option_5_vec = []
            for option in option_5:
                option_vec = []
                for word in option:
                    a+=1
                    if word in glove_dict.keys():
                        b+=1
                        word_vec = glove_dict[word]
                        option_vec.append(word_vec)
                    elif word in register.keys():
                        word_vec = register[word]
                        option_vec.append(word_vec)
                    else:
                        option_vec.append([0]*300)
                option_5_vec.append(option_vec)

            qa_vec['answers'] = option_5_vec

            if 'train' in qa['qid']:
                QA_train_vec.append(qa_vec)
                train+=1
            elif 'val' in qa['qid']:
                QA_val_vec.append(qa_vec)
                val+=1
            elif 'test' in qa['qid']:
                QA_test_vec.append(qa_vec)
                test+=1
            else:
                sys.exit("[ERROR]")         

    json.dump(QA_train_vec, open('../output_data/question/qa.train.json','w'))
    json.dump(QA_val_vec, open('../output_data/question/qa.val.json','w'))
    json.dump(QA_test_vec, open('../output_data/question/qa.test.json','w'))
    print('all: ',a,'  glove: ',b)
    print('train: ',train,'  val: ',val,'  test: ',test)

compute()
