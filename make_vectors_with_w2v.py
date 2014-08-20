# -*- coding: utf-8 -*-
#Paragraph vector thing
import gzip
import pickle
from itertools import islice
import numpy as np
import theano
import theano.tensor as T
from logistic import LogisticRegression
from wvlib import wvlib



class VectorThing():

    def __init__(self, rng, vec_size, data):

        n_sentence_vecs = []
        for i, w in enumerate(data):

            vec = np.asarray(rng.uniform(low=-np.sqrt(6. / vec_size), high=np.sqrt(6. /vec_size), size=(1, vec_size)), dtype=theano.config.floatX)

            w_vec = theano.shared(value=vec, name='word_vec' + str(i), borrow=True)
            n_sentence_vecs.append(w_vec)

        self.sentence_vecs = n_sentence_vecs


class Classifier():

    def __init__(self, rng, w_size, vec_size, input, y):


        #This guy will eat input matrix and ref matrix, and will have changeable paragraph vector
        self.learning_rate = 0.04
        self.input = input
        self.y = y

        n_in = vec_size * w_size
        n_out = vec_size

        #W for paragraph_vector
        W_par = np.asarray(rng.uniform(low=-np.sqrt(6. / (vec_size * 2)), high=np.sqrt(6. /(vec_size * 2)), size=(vec_size, vec_size)), dtype=theano.config.floatX)
        self.W_par = theano.shared(value=W_par, name='W_par', borrow=True)
        #Shared variable for paragraph

        par = np.asarray(rng.uniform(low=-np.sqrt(6. / vec_size), high=np.sqrt(6. /vec_size), size=(1, vec_size)), dtype=theano.config.floatX)
        self.paragraph = theano.shared(value=par, name='paragraph_v', borrow=True)

        #W for concatenated w2v_vectors
        W_vec =  np.asarray(rng.uniform(low=-np.sqrt(6. / (vec_size + vec_size * (w_size - 1))), high=np.sqrt(6. /(vec_size + vec_size * (w_size - 1))), size=(vec_size * (w_size - 1), vec_size)), dtype=theano.config.floatX)
        self.W_vec = theano.shared(value=W_vec, name='W_vec', borrow=True)
        #b
        b =  np.asarray(rng.uniform(low=-np.sqrt(6. / vec_size), high=np.sqrt(6. /vec_size), size=(1, vec_size)), dtype=theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        #The parameters
        self.params = [self.W_par, self.paragraph, self.W_vec, self.b]

        #Okay, insanity ensues
        #Using scan is mandatory at this point
        #This will calculate the output of our thing, it loops the input vectors
        self.result, updates = theano.scan(fn = lambda v: T.nnet.sigmoid(T.dot(v, self.W_vec) + T.dot(self.paragraph, self.W_par) + self.b), sequences=self.input)
        #As abowe, but calculates error, also loops reference
        self.error_result, e_updates = theano.scan(fn = lambda v, y: T.sum(((T.nnet.sigmoid(T.dot(v, self.W_vec) + T.dot(self.paragraph, self.W_par) + self.b) - y) ** 2)), sequences=[self.input, self.y])

        #Some functions
        self.mean_err = T.mean(self.error_result)

        self.cost = theano.function(inputs=[self.input, self.y], outputs=[self.error_result])
        self.minibatch_cost = theano.function(inputs=[self.input, self.y], outputs=[self.mean_err])

        self.output = theano.function(inputs=[self.input], outputs=[self.result])



    def get_training_function(self):

        gparams = T.grad(self.mean_err, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        train = theano.function(inputs=[self.input, self.y], outputs=[self.mean_err], updates=updates)
        return train


    def get_training_function_only_paragraph(self):

        gparams = T.grad(self.mean_err, [self.paragraph])
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        train_p = theano.function(inputs=[self.input, self.y], outputs=[self.mean_err], updates=updates)
        return train_p



    def get_cost_and_updates(self, learning_rate=0.1):

        #Cost
        cost = self.mean_err
        #Updates
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))

        return (cost, updates)#y, L


def main():

    data = load_data()
    #import pdb; pdb.set_trace()
    train_vectors(data, vec_size=300, window_size=3)


def train_vectors(data, vec_size=50, window_size=2):


    wv = wvlib.load("/usr/share/ParseBank/vector-space-models/FIN/w2v_pbv3_lm.rev01.bin",max_rank=400000)

    #Let's not normalize, just for kicks
    #wv = wv.normalize()
    rng = np.random.RandomState(1234)
    vt = VectorThing(rng, vec_size, data)
    minibatch_size = 10

    parameters = []
    input_vectors = []

    input = T.dmatrix()
    ref = T.dmatrix()

    cls = Classifier(rng, window_size, vec_size, input, ref)
    #functions
    #cost, updates = cls.get_cost_and_updates(learning_rate=0.1)
    #train = theano.function([input, ref], cost, updates=updates)
    
    train = cls.get_training_function()


    #Make batches for sentence, cut them into parts of 10 or so,
    #train maximum of that amount at once

    for epoch in range(0,50):
        print epoch
        epoch_cost = []
        for i, sentence in enumerate(data):
            #Create training material for this sentence
            sentence_refs = []
            sentence_inputs = []
            #print i
            if i%100 == 0 and i > 0:
                print np.mean(epoch_cost), i
            for win in window(sentence, n=window_size):
                try:

                    ref_vector =  wv.word_to_vector(win[-1])
                    w_vectors = []
                    for w in win[:-1]:
                        w_vectors.append(wv.word_to_vector(w))

                    sentence_refs.append(ref_vector)
                    sentence_inputs.append(np.concatenate(w_vectors))

                except:
                    pass#print ':('

            batches_ref = []
            for b_ref in chunks(sentence_refs, minibatch_size):
                batches_ref.append(b_ref)
            batches_input = []
            for b_inp in chunks(sentence_inputs, minibatch_size):
                batches_input.append(b_inp)

            #insert paragraph vector
            cls.paragraph.set_value(vt.sentence_vecs[i].eval())
            #before = vt.sentence_vecs[i].eval()
            #Train them
            for rf, inpt in zip(batches_ref, batches_input):
                #import pdb;pdb.set_trace()
                batch_cost = train(inpt, rf)
                epoch_cost.append(batch_cost)
                #print batch_cost, len(sentence_refs)
            #recover the new vector
            vt.sentence_vecs[i].set_value(cls.paragraph.eval())
            #after = vt.sentence_vecs[i].eval()
        print 'mean_cost', np.mean(epoch_cost)
        save_model(vt, 'model_epoch_' + str(epoch))


    ####AGAIN! Now with only paragraph vectors#####

    cls.learning_rate = 0.1
    train_p = cls.get_training_function_only_paragraph()

    for epoch in range(0,100):
        print epoch
        epoch_cost = []
        for i, sentence in enumerate(data):
            #Create training material for this sentence
            sentence_refs = []
            sentence_inputs = []
            #print i
            if i%100 == 0 and i > 0:
                print np.mean(epoch_cost), i
            for win in window(sentence, n=window_size):
                try:

                    ref_vector =  wv.word_to_vector(win[-1])
                    w_vectors = []
                    for w in win[:-1]:
                        w_vectors.append(wv.word_to_vector(w))

                    sentence_refs.append(ref_vector)
                    sentence_inputs.append(np.concatenate(w_vectors))

                except:
                    pass#print ':('

            batches_ref = []
            for b_ref in chunks(sentence_refs, minibatch_size):
                batches_ref.append(b_ref)
            batches_input = []
            for b_inp in chunks(sentence_inputs, minibatch_size):
                batches_input.append(b_inp)

            #insert paragraph vector
            cls.paragraph.set_value(vt.sentence_vecs[i].eval())
            #before = vt.sentence_vecs[i].eval()
            #Train them
            for rf, inpt in zip(batches_ref, batches_input):
                #import pdb;pdb.set_trace()
                batch_cost = train(inpt, rf)
                epoch_cost.append(batch_cost)
                #print batch_cost, len(sentence_refs)
            #recover the new vector
            vt.sentence_vecs[i].set_value(cls.paragraph.eval())
            #after = vt.sentence_vecs[i].eval()
        print 'mean_cost', np.mean(epoch_cost)
        save_model(vt, 'model_epoch_' + str(epoch))




def chunks(l, n):

    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def save_model(vt, fname):

    #If I cannot load them like I should, I'll do this
    outf = open(fname, 'wb')
    vecs = []
    for v in vt.sentence_vecs:
        vecs.append(v.eval())
    pickle.dump(vecs, outf)
    outf.close()

def save_model_old(vt, fname):

    #Why cannot I open this after being pickled?
    outf = open(fname, 'wb')
    pickle.dump(vt.sentence_vecs, outf)
    outf.close()


def train_vectors_old(data, vec_size=50, window_size=2):

    #Vocabulary
    vocabulary = get_vocabulary(data)
    vocabulary.append('##FILLER##')

    input = T.vector()
    y = T.iscalar()

    rng = np.random.RandomState(1234)
    #vt = VectorThing(rng, vocabulary, vec_size, 'filler', 'filler')
    vt = VectorThing(rng, vocabulary, vec_size, data)
    lr = LogisticRegression(vt, window_size, vec_size, vocabulary, input, y)

                            #(vectors, w_size, vec_size, vocab, input, y

    cost = lr.negative_log_likelihood(y)

    #minibatch_size = 1
    learning_rate = 0.01

    x = T.ivector()
    parameters = []
    input_vectors = []





    for i, sentence in enumerate(data):
        for win in window(sentence, n=window_size):
            reference_word = vocabulary.index(win[-1])
            input = []
            input.append(i)
            parameters.append(vt.sentence_vecs[i])

            for w in win[:-1]:
                input.append(vocabulary.index(w))
                parameters.append(vt.word_vecs[input[-1]])

            #reference on int missä on "oikea vastaus"
            #input alkaa paragraph idllä
            #ja jatkuu word vecceinä

            #Toivottavasti sisällä networkissa, ebun!
            concatenated_input = T.concatenate(parameters)


            #Ebin!
            #do updates

            learning_rate = 0.01

            gparams_lr = []
            for param in lr.params:
                gparam = T.grad(cost, param)
                gparams_lr.append(gparam)

            gparams_v = []
            for param in parameters:
                gparam = T.grad(cost, param)
                gparams_v.append(gparam)

            updates = []
            for param, gparam in zip(lr.params, gparams):
                updates.append((param, param - learning_rate * gparam))

            for param, gparam in zip(parameters, gparams_v):
                updates.append((param, param - learning_rate * gparam))

            train_model = theano.function(inputs=[input, y], outputs=cost, updates=updates)

            print train_model(concatenated_input, reference_word)

            #Noni, nyt on YKSI esimerkki. Muuta niin että on monta


            import pdb; pdb.set_trace()


            #Ok, we've got refrence and input vectors
            #Make the above matrix with theano

            #soo, var vector will be the last thing to be concatenated!
            #Ok, eli nyt mulla on listassa oikeestaan kaikki offsetit ja info tuloksesta
            #Täytyy nyt sitten niinku pistää ne yhdistettyyn thano tensoriin
            #Elis koitetaan ensin tällaista:

            #1. scannerilla word vecit listaan
            #2. referenssi tensoriksi kans
            #


            #
            #Tarkista myöhemmin onko noi parametrit kunnossa nyt ollenskaan


            



            #concatenate it using theano
            concatenated = T.reshape(vector_matrix, (1, (window_size + 1)* vec_size))
            #get params from vectors and log_reg




    #How to create the windows?
    #Don't include any fillers for now



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

def load_data():

    try:
        inf = open('data', 'rt')
        sentences = pickle.load(inf)
        inf.close()
    except:
        #Load stuff from parsebank
        trees = 50000
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
