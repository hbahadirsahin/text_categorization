import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from TextCategorization.load_dataset import create_batches, load_test_set
from TextCategorization.preprocessor import Preprocessor

SET_OR_INTERACTIVE = input("Train from start or restore the latest checkpoint: (S/I)\n")

while not (SET_OR_INTERACTIVE.isalpha() and SET_OR_INTERACTIVE.upper() in ['S', 'I']):
    SET_OR_INTERACTIVE = input('Invalid input. (S) for test set, (I) for interactive evaluation:')

SET_OR_INTERACTIVE.upper()

test_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.test"
fasttext_model_path = "D:/PycharmProjects/TextCategorization/fasttext_tr_embedding_cache.npy"

tf.flags.DEFINE_string("test_file", test_path, "Test data source")
tf.flags.DEFINE_string("fasttext_model", fasttext_model_path, "FastText embeddings source")

# Model Parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate, default(0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Learning rate, default(0.5)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Filter size, default(3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters, default(128)")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size, default(300)")
tf.flags.DEFINE_integer("num_classes", 25, "Number of tasks, default(25)")
tf.flags.DEFINE_integer("top_k_category", 5, "Number of categories to return as result, default(5)")

tf.flags.DEFINE_integer("epoch", 30, "Number of epochs, default(25)")
tf.flags.DEFINE_integer("save_step", 500, "Step to save the model, default(500)")
tf.flags.DEFINE_integer("evaluate_step", 500, "Step to evaluate the model, default(500)")
tf.flags.DEFINE_integer("batch_size", 512, "Training batch size, default(512)")
tf.flags.DEFINE_integer("max_sentence_length", 40, "Maximum sentence length, default(512)")
tf.flags.DEFINE_integer("decay_step", 250, "How many steps before decay learning rate, default(250)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate, (default: 0.95)")
tf.flags.DEFINE_float("norm_ratio", 1.25,
                      "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)


def main():
    if SET_OR_INTERACTIVE == "S":
        test_sentences, y_test = load_test_set(FLAGS.test_file)
    # else:
    #     print("Broken!")
    #     preprocessor = Preprocessor(language="turkish")
    #     test_sentences = []
    #     test_sentence = input("Enter a test sentence:")
    #     y_test = None
    #     test_sentence = preprocessor.preprocess(test_sentence)
    #     test_sentences.append(test_sentence)

    embedding = np.load(FLAGS.fasttext_model)

    print("Restoring the model for test!")
    model_number = input("Enter the checkpoint number:")
    while not model_number.isdigit() and len(model_number) == 10:
        model_number = input("Invalid format. Re-enter:")
    checkpoint_dir = "runs/" + model_number + "/checkpoints/"
    outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_number))
    vocabulary_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(outdir, "vocab"))

    print(test_sentences)
    x_test = np.array(list(vocabulary_processor.transform(test_sentences)))
    print(x_test)

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            batch_norm = graph.get_operation_by_name("batch_norm").outputs[0]
            embedding_placeholder = graph.get_operation_by_name("embedding/pretrained_embedding").outputs[0]

            predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]
            predictions_top_k = graph.get_operation_by_name("top-k-accuracy/top2").outputs[0]

            test_batches = create_batches(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

            all_preds = []
            all_preds_k = []
            for batch in test_batches:
                x_batch, y_batch = zip(*batch)
                batch_pred, batch_pred_top_k = sess.run([predictions, predictions_top_k], {input_x: x_batch,
                                                                                           input_y: y_batch,
                                                                                           embedding_placeholder: embedding,
                                                                                           keep_prob: 1.0,
                                                                                           batch_norm: False})

                all_preds = np.concatenate([all_preds, batch_pred])
                # all_preds_k = np.concatenate([all_preds_k, batch_pred_top_k])

    if y_test is not None:
        correct_predictions = float(sum(all_preds == np.argmax(y_test, axis=1)))
        print("Total number of test examples:", len(y_test))
        print("Accuracy:", correct_predictions / float(len(y_test)))
    else:
        print(all_preds)
        print(all_preds_k)


if __name__ == '__main__':
    main()
