from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from tqdm import tqdm
from arxiv_preprocess import corpus, titles, abstracts
from arxiv_categories_preprocess import corpus_with_categories

def extract_best_indices(m, topk):
    """
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score
    mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) # eliminate 0 cosine distance
    best_index = index[mask][:topk]
    for ind in best_index:
        print(cos_sim[ind])
    return best_index

BERT_BATCH_SIZE = 4
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

class BertModel:
    def __init__(self, model_name, batch_size=BERT_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.load_pretrained_model()

    def load_pretrained_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = 'blabla/cache')
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir = 'blabla/cache')
        self.pipeline = pipeline('feature-extraction',
                                 model=self.model, tokenizer=self.tokenizer)

    def embed(self, data):
        """ Create the embedded matrice from original sentences """
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        return mean_pooled

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        if 'str' in data.__class__.__name__:
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        # each of the 512 token has a 768 or 384-d vector depends on model)
        attention_mask = token_dict['attention_mask']
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(token_embed, attention_mask)
        return mean_pooled

    def predict(self, in_sentence, embed_mat, topk=6):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, embed_mat)
        # best cos sim for each token independantly
        best_index = extract_best_indices(mat, topk=topk)
        return best_index

bert_model = BertModel(model_name=MODEL_NAME, batch_size=BERT_BATCH_SIZE)

embed_mat = bert_model.embed(corpus)

label_embed_mats = {}
for label, articles in corpus_with_categories.items():
    label_embed_mats[label] = bert_model.embed(corpus_with_categories[label])
