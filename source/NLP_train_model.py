import os
import re
import nltk
from nltk import word_tokenize
import string
from tqdm import tqdm
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, KneserNeyInterpolated, WittenBellInterpolated
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')

path = os.getcwd()
path = path[:-7]

full = []
for dirname, _, filenames in os.walk(f'{path}/data_nlp'):
    for filename in filenames:
        with open(os.path.join("", os.path.join(dirname, filename)), 'r', encoding='UTF-16') as f:
            full.append(f.read())

#Tách từ 
def tokenize(doc):
    tokens = word_tokenize(doc.lower())
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

full_data = ". ".join(full)
full_data = full_data.replace("\n", ". ")
corpus = []
sents = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', full_data)
for sent in tqdm(sents):
    corpus.append(tokenize(sent))
print(len(full))

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurance  
for sentence in tqdm(corpus):
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
 
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

#Xây dựng mô hình ngram trên nltk
train_data, padded_sents = padded_everygram_pipeline(3, corpus)

#Phương pháp làm mịn dữ liệu (do có nhiều dữ liệu xác suất = 0 khi chọn n của ngrams quá lớn --> làm mịn (có thể thay thế = Laplace)
n = 3
vi_model = KneserNeyInterpolated(n)

vi_model.fit(train_data, padded_sents)
print(len(vi_model.vocab))

#Upload model
model_dir = f"{path}/models_nlp"
with open(os.path.join(model_dir, 'ngram_model.pkl'), 'wb') as fout:
    pickle.dump(vi_model, fout)