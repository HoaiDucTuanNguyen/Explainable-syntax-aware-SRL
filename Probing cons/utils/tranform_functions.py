import joblib
import argparse
import os
import re
import json
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from statistics import median
from sklearn.model_selection import train_test_split
SEED = 42

def mask_word_in_sentence(sentence, predicate, ner_tag, pos_tag, index_word_to_mask):
    
    # => array of [new sentence, ner_tag, pos_tag]
    return True

    
def coNLL_ner_pos_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """
    This function transforms the data present in coNLL_data/. 
    Raw data is in BIO tagged format with the POS and NER tags separated by space.
    The transformation function converts the each raw data file into two separate tsv files,
    one for POS tagging task and another for NER task. Following transformed files are written at wrtDir

    - NER transformed tsv file.
    - NER label map joblib file.
    - POS transformed tsv file.
    - POS label map joblib file.

    For using this transform function, set ``transform_func`` : **snips_intent_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """

    
    f = open(os.path.join(dataDir, readFile))

    nerW = open(os.path.join(wrtDir, 'ner_{}.tsv'.format(readFile.split('.')[0])), 'w')
    posW = open(os.path.join(wrtDir, 'pos_{}.tsv'.format(readFile.split('.')[0])), 'w')

    labelMapNer = {}
    labelMapPos = {}

    sentence = []
    senLens = []
    labelNer = []
    labelPos = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    
    for i, line in enumerate(tqdm(f)):
        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                # NOTE: CODE HERE
                
                nerW.write("{}\t{}\t{}\n".format(uid, labelNer, sentence))
                posW.write("{}\t{}\t{}\n".format(uid, labelPos, sentence))
                senLens.append(len(sentence))
                #print("len of sentence :", len(sentence))

                sentence = []
                labelNer = []
                labelPos = []
                uid += 1
            continue
            
        sentence.append(wordSplit[0])
        labelPos.append(wordSplit[-2])
        labelNer.append(wordSplit[-1])
        # NOTE: CODE HERE
        if isTrainFile:
            if wordSplit[-1] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
            if wordSplit[-2] not in labelMapPos:
                labelMapPos[wordSplit[-2]] = len(labelMapPos)
    
    print("NER File Written at {}".format(wrtDir))
    print("POS File Written at {}".format(wrtDir))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "ner_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapPos != {} and isTrainFile:
        print("Created POS label map from train file {}".format(readFile))
        print(labelMapPos)
        labelMapPosPath = os.path.join(wrtDir, "pos_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapPos, labelMapPosPath)
        print("label Map POS written at {}".format(labelMapPosPath))

    f.close()
    nerW.close()
    posW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))
