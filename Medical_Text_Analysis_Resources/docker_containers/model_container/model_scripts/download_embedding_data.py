#download data needed for the sentence embeddings
import nltk
from sentence_transformers import models
nltk.download('punkt')
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer
#SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
models.Transformer('emilyalsentzer/Bio_ClinicalBERT')
