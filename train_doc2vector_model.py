#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import imp
import multiprocessing

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from random import shuffle

imp.reload(sys)
# reload(sys)
# sys.setdefaultencoding('utf-8') #允许打印unicode字符

print(__name__)
if __name__ == '__main__':  # to check if this thread is main? not import
    print('argu===',sys.argv[0])
    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    #if len(sys.argv) < 4:
    #    print (globals()['__doc__'] % locals())
    #    sys.exit(1)
    inp = str("D:/NSFC/project/data/20NewLong/title_StackOverflow_nostopwords.txt",'utf8')
    outp2 = str("D:/NSFC/project/data/20NewLong/title_StackOverflow_nostopwords.d2v.emb",'utf8')
    outp1 = str("D:/NSFC/project/data/20NewLong/title_StackOverflow_nostopwords.d2v.model",'utf8')

    sentences=TaggedLineDocument(inp)
    logger.info("******")
    for index,doc in enumerate(sentences):
        logger.info(doc.words)
        line=[]
        for item in doc.words:
            line.append(item.split(str=":")[0])
        doc.words=line
        logger.info(doc.words)
        logger.info(doc.tags)
        if(int(doc.tags[0])>10):
            break
    logger.info("******")

    model = Doc2Vec( sentences,size=128, window=100, min_count=0, dm=0, sample=1e-4, negative=5, iter=100, workers=multiprocessing.cpu_count())
    # model.build_vocab(sentences)
    # for epoch in range(10):
    #     shuffle(sentences)
    #     model.train(sentences)

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(=replaceTrue)
    model.save(outp1)#save dov2vec
    model.save_word2vec_format(outp2, doctag_vec=True,word_vec=False, binary=False)#save word2vec
    vector = model.infer_vector(inp)
    #print(vector)
