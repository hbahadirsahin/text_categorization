from gensim.models import FastText
from TextCategorization.load_dataset import load_data_and_labels
import numpy as np
import pickle
import argparse


def create_preloaded_fasttext_embeddings_and_vocabulary(fasttext_model_path = "D:/PycharmProjects/TextCategorization/wiki.tr",
                                                        vocab_file_output_path = "fasttext_tr_vocab_cache.dat",
                                                        embedding_cache_output_path = "fasttext_tr_embedding_cache.npy"):
    model = FastText.load_fasttext_format(fasttext_model_path)
    vocabulary = model.wv.vocab
    embeddings = np.array([model.wv.word_vec(word) for word in vocabulary.keys()])

    with open(vocab_file_output_path, 'wb') as fw:
        pickle.dump(vocabulary, fw, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(embedding_cache_output_path, embeddings)
    print("Preloading ends!")

def create_oov_vocabulary(dataset_path, fasttext_model_path):
    sentences, _ = load_data_and_labels(dataset_path)
    model = FastText.load_fasttext_format(fasttext_model_path)

def create_data_and_labels_using_fasttext_embeddings(dataset_path, fasttext_model_path, embedding_size=300, language='turkish'):
    print("Loading raw sentences and one hot encoded labels")
    sentences, labels = load_data_and_labels(dataset_path)

    print("Finding maximum sentence length in the dataset")
    max_sentence_length = max([len(sentence.split(" ")) for sentence in sentences])
    print("Loading Fasttext model")
    model = FastText.load_fasttext_format(fasttext_model_path)
    sentence_embedding_list = list()
    print("Transform raw sentences to Fasttext embeddings")
    count = 0
    for sentence_idx, sentence in enumerate(sentences):
        tokens = sentence.split(" ")
        sentence_embedding = np.zeros(shape=(max_sentence_length, embedding_size))
        for idx, word in enumerate(tokens):
            try:
                sentence_embedding[idx] = model[word]
            except KeyError:
                print("Sentence:", sentence, "- Error word:", word)
        sentence_embedding_list.append(sentence_embedding)
        if sentence_idx % 100000 == 0:
            outputPath = "D:/PycharmProjects/TextCategorization/FasttextEmbeddingsWithNGrams/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction_" + str(count)
            np.save(outputPath, np.asarray(sentence_embedding_list))
            sentence_embedding_list.clear()
            count += 1

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft-path', dest='ft_path', help='Dataset path!', required=True)
    parser.add_argument('--vocab-path', dest='vocab_path', help='Output path for training set!', required=True)
    parser.add_argument('--embedding-path', dest='embed_path', help='Output path for validation set!', required=True)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    ft_model_path = options.ft_path
    vocab_output_path = options.vocab_path
    embedding_cache_output = options.embed_path
    create_preloaded_fasttext_embeddings_and_vocabulary(fasttext_model_path=ft_model_path,
                                                        vocab_file_output_path=vocab_output_path,
                                                        embedding_cache_output_path=embedding_cache_output)

# turkish_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP"
# model_path = "D:/PycharmProjects/TextCategorization/wiki.tr"
# create_data_and_labels_using_fasttext_embeddings(turkish_dataset_path, model_path)

