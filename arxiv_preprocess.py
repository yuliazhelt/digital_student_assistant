import json
from tqdm import tqdm

data_file = 'data/arxiv_updated.json'

CORPUS_SIZE = 50

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line

titles = []
abstracts = []
metadata = get_metadata()
for paper in tqdm(metadata):
    paper_dict = json.loads(paper)
    titles.append(paper_dict.get('title'))
    abstracts.append(paper_dict.get('abstract'))

corpus = []
for i in tqdm(range(CORPUS_SIZE)):
    corpus.append(titles[i])
