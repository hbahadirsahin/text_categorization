from TextCategorization.preprocessor import Preprocessor
from random import shuffle
from collections import Counter
import numpy as np
import pickle

category_dict = dict()

def load_data_and_labels(dataset_path, language='turkish'):
    global category_dict

    preprocessor = Preprocessor(language)
    category_labels = list()
    entity_labels = list()
    sentences = list()
    category_id = 0
    with open(dataset_path, 'r', encoding="utf-8") as file:
        for line in file:
            line_tokens = line.split('\t')
            if line_tokens[0] in category_dict:
                category_labels.append(category_dict[line_tokens[0]])
            else:
                category_dict[line_tokens[0]] = category_id
                category_labels.append(category_dict[line_tokens[0]])
                category_id += 1
            entity_labels.append(line_tokens[1])
            sentences.append(preprocessor.preprocess(line_tokens[2]))
    return sentences, one_hot_encoded(category_labels, len(category_dict))


def load_train_vali_test_sets(train_path, vali_path, test_path="", language='turkish'):
    train_data, train_labels = load_data_and_labels(train_path, language)
    vali_data, vali_labels = load_data_and_labels(vali_path, language)
    if test_path != "":
        test_data, test_labels = load_data_and_labels(test_path, language)

    return train_data, train_labels, vali_data, vali_labels, test_data, test_labels

def load_test_set(train_path, test_path, language='turkish'):
    _, _ = load_data_and_labels(train_path, language)
    test_data, test_labels = load_data_and_labels(test_path, language)
    return test_data, test_labels

def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1
    return np.eye(num_classes)[class_numbers]


def create_batches(data, batch_size, num_epoch, shuffle=False):
    data = np.array(data)
    num_batches = int((data.shape[0] - 1) / batch_size) + 1
    for e in range(num_epoch):
        if shuffle:
            shuffled_idx = np.random.permutation(np.arange(data.shape[0]))
            data = data[shuffled_idx]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data.shape[0])
            yield data[start_index:end_index]

def create_deterministic_train_vali_test_sets(dataset_path, train, vali, test):
    with open(dataset_path, encoding="utf-8") as dataset:
        lines = dataset.readlines()

    shuffle(lines)

    begin_train = 0
    end_train = int(len(lines) * 0.7)
    begin_vali = end_train + 1
    end_vali = begin_vali + int(len(lines)*0.15)
    begin_test = end_vali + 1

    with open(train, "w", encoding="utf-8") as tf:
        # pickle.dump(lines[begin_train:end_train], tf, protocol=pickle.HIGHEST_PROTOCOL)
        for idx,line in enumerate(lines[begin_train:end_train]):
            tf.write(line)
        print("Train file has been written")

    with open(vali,  "w", encoding="utf-8") as vf:
        # pickle.dump(lines[begin_vali:end_vali], vf, protocol=pickle.HIGHEST_PROTOCOL)
        for idx,line in enumerate(lines[begin_vali:end_vali]):
            vf.write(line)
        print("Validation file has been written")

    with open(test,  "w", encoding="utf-8") as tef:
        # pickle.dump(lines[begin_test:], tef, protocol=pickle.HIGHEST_PROTOCOL)
        for idx,line in enumerate(lines[begin_test:]):
            tef.write(line)
        print("Test file has been written")



# turkish_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP"
# # load_data_and_labels(turkish_dataset_path)
# train_out_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.train"
# vali_out_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.vali"
# test_out_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.test"
# create_deterministic_train_vali_test_sets(turkish_dataset_path, train_out_path, vali_out_path, test_out_path)
