# -*- coding: utf-8 -*-
#Paragraph vector thing
import gzip
import pickle
from itertools import islice
import numpy as np
import theano
import theano.tensor as T
from logistic import LogisticRegression


def main():

    data = load_data()
    vocabulary = get_vocabulary(data)
    #wf_data = wf_load_data
    #import pdb; pdb.set_trace()
    #Load the vectors
    inf = open('model_epoch_0','rb')
    
    vecs = pickle.load(inf)
    inf.close()

    #Evaluate them all
    #word_vecs = []
    #for wv in vecs[0]:
    #    word_vecs.append(wv.eval())
 
    import pdb; pdb.set_trace() 
    sentence_vecs = []
    for wv, sent in zip(vecs, data):
        sentence_vecs.append((normalize(wv), sent))

    #So just get a random word or paragraph and this will get you like its closest allies
    do_comparison(0, sentence_vecs)
    do_comparison(1, sentence_vecs)
    do_comparison(300, sentence_vecs)

    do_comparison(900, sentence_vecs)
    do_comparison(15, sentence_vecs)
    do_comparison(240, sentence_vecs)

    do_comparison(900, sentence_vecs)
    do_comparison(815, sentence_vecs)
    do_comparison(640, sentence_vecs)

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm


def do_comparison(comp_index, sentence_vecs):

    compare = sentence_vecs[comp_index]
    print
    print '-'*20
    print compare[1]
    print

    #import pdb; pdb.set_trace()
    dotted = []
    for d_tpl in sentence_vecs:
        dotted.append((np.dot(d_tpl[0], np.transpose(compare[0])), d_tpl[1]))

    dotted.sort()
    dotted.reverse()
    print
    for res in dotted[:10]:
        print res[0]
        print res[1]



def save_model(vt, fname):

    outf = open(fname, 'wb')
    pickle.dump([vt.word_vecs, vt.sentence_vecs], outf)
    outf.close()



def window(seq, n=2):

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def get_vocabulary(data):

    vocab = set()
    for sentence in data:
        for token in sentence:
            vocab.add(token)

    return list(vocab)

def wf_load_data():

    try:
        inf = open('wf_data', 'rt')
        sentences = pickle.load(inf)
        inf.close()
    except:
        #Load stuff from parsebank
        trees = 1000
        pbfile = gzip.open('/usr/share/ParseBank/parsebank_v3.conll09.gz', 'rt')
        sentences = load_sentences(pbfile, trees, lemmas=False)
        outf = open('wf_data', 'wt')
        pickle.dump(sentences, outf)
        outf.close()
    return sentences


def load_data():

    try:
        inf = open('data', 'rt')
        sentences = pickle.load(inf)
        inf.close()
    except:
        #Load stuff from parsebank
        trees = 1000
        pbfile = gzip.open('/usr/share/ParseBank/parsebank_v3.conll09.gz', 'rt')
        sentences = load_sentences(pbfile, trees, lemmas=True)
        outf = open('data', 'wt')
        pickle.dump(sentences, outf)
        outf.close()
    return sentences


def load_sentences(pbfile, trees, lemmas=False):

    sentences = []
    for t in range(0,trees):
        sentence = None
        while(sentence == None):
            sentence = get_a_sentence(pbfile, lemmas)
            if sentence != None:
                sentences.append(sentence)
    return sentences

            
def get_a_sentence(pbfile, lemmas):

    tokens = []
    line = ''
    tokens = []
    while(line != '\n'):

        line = pbfile.readline()
        if '##PB' in line:
            tokens = None
            break
        if '\t' not in line:
            break
        #print line
        if line.split('\t')[4] != 'Punct':
            if lemmas:
                tokens.append(line.split('\t')[2])
            else:
                tokens.append(line.split('\t')[1])

    if len(tokens) < 2:
        tokens = None
    return tokens


main()
