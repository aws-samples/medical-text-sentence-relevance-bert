
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import json
from scipy.stats import gaussian_kde
import nltk
import scipy
import numpy
import nltk
nltk.download('punkt')

#configure logging
import logging 
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

class MedicalBertModel():
    def __init__(self,run_null_model=None):
        self.run_null_model=run_null_model
        pass
    
    def run_null(self,null_location='null_hypothesis.txt',score=None):
        '''run null hypothesis to get pvalue'''
        f_in=open(null_location,'r')
        hist_dists_null=json.load(f_in)
        #next, create the kernel density estimation (kde) for the null hypothesis
        kde= gaussian_kde(hist_dists_null)
        pvalue=self.get_pvalue(kde,score)
        return(pvalue)

    def get_pvalue(self,kde,score):
        '''given kde of the null hypothesis, get the probability of achieving that score or better'''
        logger.debug(score)
        pvalue=kde.integrate_box_1d(0,score)
        return(pvalue)

    def break_up_by_sentence(self,the_string):
        '''break up supporting documentation into separate sentences'''
        to_return=nltk.tokenize.sent_tokenize(the_string)
        return(to_return)

    def get_best_n_sentences(self,corpus,distances,max_to_return=5):
        '''get the top n closest sentences; where lower distance is better'''
        max_to_return=5
        if len(corpus) <5: #
            max_to_return=len(corpus) 
        the_indices=numpy.argsort(distances)
        top_n_indices=the_indices[0:max_to_return]
        top_n_dist=[distances[i] for i in top_n_indices]
        logger.debug(corpus)
        top_n_corpus=[corpus[i] for i in top_n_indices]
        top_n_pvalues=[]
        if self.run_null_model==True:
            for i in range(0,len(top_n_dist)):
                temp_result=self.run_null(score=top_n_dist[i])
                top_n_pvalues.append(temp_result)	
        list_to_return=[]
        for i in range(0,len(top_n_indices)):
            if self.run_null_model==True:
                the_dict={"rank":i+1,"matching_sentence":top_n_corpus[i],"pvalue":top_n_pvalues[i],"distance":top_n_dist[i]}
            else:
                the_dict={"rank":i+1,"matching_sentence":top_n_corpus[i],"distance":top_n_dist[i]}
            list_to_return.append(the_dict)
        return(list_to_return)
    
    def run_model_and_null(self,request):
        '''run model and null model if specified'''
        request_dict=json.loads(request)
        input_sentence=request_dict['input_sentence']
        the_paragraph=request_dict['input_paragraph']
        model_result=self.run_model(input_sentence,the_paragraph)
        return(model_result)
    
    def run_model(self,input_sentence,the_paragraph):
        #get the embedder
        word_embedding_model = models.Transformer('emilyalsentzer/Bio_ClinicalBERT')
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        # create a corpus of every individual sentence within the paragraph
        corpus=self.break_up_by_sentence(the_paragraph)
        corpus_embeddings = embedder.encode(corpus)
        # define the input sentence as the query
        queries = [input_sentence]
        query_embedding = embedder.encode(queries)
        # calculate the distance between the query embedding and the corpus embeddings
        distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]
        to_return=self.get_best_n_sentences(corpus,distances)
        return(to_return)


if __name__=='__main__':
    mbm=MedicalBertModel(run_null_model=False)
    mock_input=json.dumps({'input_sentence':'The patient is healthy.',"input_paragraph":"The patient's health is good. The patient does not have a fever. The patient is a 28 year old female."})
    result=mbm.run_model_and_null(mock_input)
    logger.info(f'Model result: {result}')


