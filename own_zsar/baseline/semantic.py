from gensim.models import KeyedVectors as Word2Vec
import numpy as np
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import normalize
import sent2vec
import gensim.downloader
import clip

def semantic_embeddings(semantic_name, dataset_name, labels, vit_backbone = "ViT-B/16"):
    if semantic_name == 'word2vec':
        model = word2vec_model()
        return word2vec_embeddings(dataset_name, labels, model, semantic_name)
    elif semantic_name == 'fasttext':
        model = fasttext_model()
        return word2vec_embeddings(dataset_name, labels, model, semantic_name)
    elif semantic_name == 'sent2vec':
        model = sent2vec_model()
        return sent2vec_embeddings(dataset_name, labels, model, semantic_name)
    elif 'clip' in semantic_name:
        device = 'cuda'
        model, _ = clip.load(vit_backbone, device=device)
        return clip_text_encoder(dataset_name, labels, model, semantic_name)


def clip_text_encoder(dataset_name, labels, model, semantic_name):
    if dataset_name == 'ucf' or dataset_name == 'test':
        sent2vec_function = sent2vec_ucf
    elif dataset_name == 'hmdb':
        sent2vec_function = sent2vec_hmdb
    elif dataset_name == 'olympics':
        sent2vec_function = sent2vec_olympics
    # elif dataset_name == 'kinetics400':
    elif dataset_name == 'kinetics':
        sent2vec_function = sent2vec_kinetics400
    elif dataset_name == 'activitynet':
        sent2vec_function = sent2vec_anet

    sentences = [sent2vec_function(label, semantic_name) for label in labels]
    # if semantic_name == 'clip_manual':
    sentences = [("a video of a person " + sent) for sent in sentences]
    # print(sentences)

    device = 'cuda'
    text = clip.tokenize(sentences).to(device)
    text_features = model.encode_text(text)
    text_features = text_features.float()
    text_features = text_features.cpu().detach().numpy()
    return text_features


# Manual changes
ucf_change_word = {
    'CleanAndJerk': ['weight', 'lift'],
    'Skijet': ['Skyjet'],
    'PushUps': ['pushups'],
    'PullUps': ['pullups'],
    'WalkingWithDog': ['walk', 'dog'],
    'ThrowDiscus': ['throw', 'disc'],
    'TaiChi': ['taichi'],
    'CuttingInKitchen': ['cut', 'kitchen'],
    'YoYo': ['yoyo'],
}

ucf_change_sent = {
    'FrontCrawl': ['front', 'crawl', 'swimming'],
    'Mixing': ['mixing', 'batter'],
    'PommelHorse': ['pommel', 'horse', 'gymnastics'],
    'CleanAndJerk': ['weight', 'lift'],
    'Skijet': ['skyjet'],
    'PushUps': ['pushups'],
    'PullUps': ['pullups'],
    'WalkingWithDog': ['walk', 'dog'],
    'ThrowDiscus': ['throw', 'disc'],
    'TaiChi': ['taichi'],
    'CuttingInKitchen': ['cut', 'kitchen'],
    'YoYo': ['yoyo'],
}

hmdb_change_sent = {
    'chew': ['chew', 'food'],
    'pour': ['pour', 'liquid'],
    'turn': ['turn', 'around'],
}

olympics_change = {
    'clean_and_jerk': ['weight', 'lift'],
}

kinetics400_change = {
    'faceplanting': ['face', 'plant'],
    'situp': ['sit', 'up'],
    'barbequing': ['barbeque'],
}

# Word2Vec
def word2vec_model():
    wv_model = Word2Vec.load_word2vec_format(
            '../../datasets/semantic/GoogleNews-vectors-negative300.bin', binary=True)
    wv_model.init_sims(replace=True)
    return wv_model

def word2vec_embeddings(dataset_name, labels, wv_model, semantic_name):
    if dataset_name == 'ucf' or dataset_name == 'test':
        word2vec_function = word2vec_embeddings_ucf
    elif dataset_name == 'hmdb':
        word2vec_function = word2vec_embeddings_hmdb
    elif dataset_name == 'olympics':
        word2vec_function = word2vec_embeddings_olympics
    elif dataset_name == 'kinetics400':
        word2vec_function = word2vec_embeddings_kinetics

    embeddings = [word2vec_function(label, wv_model, semantic_name) for label in labels]
    embeddings = np.stack(embeddings)
    embeddings = normalize(embeddings.squeeze())
    return normalize(embeddings.squeeze())

def word2vec_embeddings_ucf(label, wv_model, semantic_name):
    vec = vectorize_ucf(label, semantic_name)
    return wv_model[vec].mean(0)

def word2vec_embeddings_hmdb(label, wv_model, semantic_name):
    vec = vectorize_hmdb(label, semantic_name)
    return wv_model[vec].mean(0)

def word2vec_embeddings_olympics(label, wv_model, semantic_name):
    vec = vectorize_olympics(label, semantic_name)
    return wv_model[vec].mean(0)

def word2vec_embeddings_kinetics(label, wv_model, semantic_name):
    vec = vectorize_kinetics400(label, semantic_name)
    mn = [1 for _ in range(300)]
    for v in vec:
        if v not in wv_model:
            return mn
    return wv_model[vec].mean(0)

# FastText
def fasttext_model():
    fasttext = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    return fasttext

# Sent2Vec
def sent2vec_model():
    model = sent2vec.Sent2vecModel()
    model.load_model('../../datasets/semantic/wiki_unigrams.bin')
    return model

def sent2vec_embeddings(dataset_name, labels, model, semantic_name):
    if dataset_name == 'ucf' or dataset_name == 'test':
        sent2vec_function = sent2vec_ucf
    elif dataset_name == 'hmdb':
        sent2vec_function = sent2vec_hmdb
    elif dataset_name == 'olympics':
        sent2vec_function = sent2vec_olympics
    elif dataset_name == 'kinetics400':
        sent2vec_function = sent2vec_kinetics400

    sentences = [sent2vec_function(label, semantic_name) for label in labels]
    vectors = model.embed_sentences(sentences)
    return vectors

def sent2vec_ucf(label, semantic_name):
    vec = vectorize_ucf(label, semantic_name)
    sent = " ".join(vec)
    return sent

def sent2vec_hmdb(label, semantic_name):
    vec = vectorize_hmdb(label, semantic_name)
    sent = " ".join(vec)
    return sent

def sent2vec_anet(label, semantic_name):
    vec = vectorize_anet(label, semantic_name)
    sent = " ".join(vec)
    return sent

def sent2vec_olympics(label, semantic_name):
    vec = vectorize_olympics(label, semantic_name)
    sent = " ".join(vec)
    return sent

def sent2vec_kinetics400(label, semantic_name):
    vec = vectorize_kinetics400(label, semantic_name)
    sent = " ".join(vec)
    return sent

# Labels Processing
def vectorize_ucf(label, semantic_name):
    vec = []
    if (semantic_name == 'word2vec' or semantic_name == 'fasttext') and label in ucf_change_word:
            vec = ucf_change_word[label]
    elif semantic_name == 'sent2vec' and label in ucf_change_sent:
            vec = ucf_change_sent[label]
    else:
        upper_idx = np.where([x.isupper() for x in label])[0].tolist()
        upper_idx += [len(label)]
        vec = []
        for i in range(len(upper_idx)-1):
            vec.append(label[upper_idx[i]: upper_idx[i+1]])
        vec = [n.lower() for n in vec]
    return vec

def vectorize_hmdb(label, semantic_name):
    if semantic_name == 'sent2vec' and label in hmdb_change_sent:
        vec = hmdb_change_sent[label]
    else:
        vec = label.split('_')
    return vec

def vectorize_anet(label, semantic_name):
    label = label.lower()
    return label.split(" ")

def vectorize_olympics(label, semantic_name):
    if label in olympics_change:
        vec = olympics_change[label]
    else:
        vec = label.split('_')
    return vec

def vectorize_kinetics400(label, semantic_name):
    if label in kinetics400_change:
        vec = kinetics400_change[label]
    else:
        label = label.replace('(', '')
        label = label.replace(')', '')
        vec = label.split(' ')
    return vec

