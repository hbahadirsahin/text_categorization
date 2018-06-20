from gensim.models import FastText
from TextCategorization.load_dataset import load_data_and_labels
import numpy as np
import pickle


def create_preloaded_fasttext_embeddings_and_vocabulary():
    fasttext_model_path = "D:/PycharmProjects/TextCategorization/wiki.tr"
    print("Preloading starts...")
    model = FastText.load_fasttext_format(fasttext_model_path)
    vocabulary = model.wv.vocab
    embeddings = np.array([model.wv.word_vec(word) for word in vocabulary.keys()])

    with open('fasttext_tr_vocab_cache.dat', 'wb') as fw:
        pickle.dump(vocabulary, fw, protocol=pickle.HIGHEST_PROTOCOL)

    np.save('fasttext_tr_embedding_cache.npy', embeddings)
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

create_preloaded_fasttext_embeddings_and_vocabulary()
# turkish_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP"
# model_path = "D:/PycharmProjects/TextCategorization/wiki.tr"
# create_data_and_labels_using_fasttext_embeddings(turkish_dataset_path, model_path)

