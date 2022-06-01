from arxiv_preprocess import corpus, titles, abstracts
from bert_model import bert_model, embed_mat, label_embed_mats
from label_classification_roberta import predict_category

query_sentence = 'Climate change'

indices = bert_model.predict(query_sentence, embed_mat=embed_mat)
print('First way predictions:\n')
for i in range(3):
    print(titles[indices[i]], '\nAbstract: ', abstracts[indices[i]], '\n')

query_label = predict_category(query_sentence)
indices = bert_model.predict(query_sentence, embed_mat=label_embed_mats[query_label])
print('Second way predictions in category: ', query_label, '\n')
for i in range(3):
    print(corpus[query_label][indices[i]], '\n')
