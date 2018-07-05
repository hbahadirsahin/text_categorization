import datetime
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from TextCategorization.cnn_model import CNN_Categorizer_v2
from TextCategorization.load_dataset import create_batches, load_train_vali_test_sets

RESTORE_TRAINING = input("Train from start or restore the latest checkpoint: (T/R)\n")

while RESTORE_TRAINING.upper() not in ['T', 'R']:
    RESTORE_TRAINING = input('Invalid input. (T) for train, (R) for restore:')

CHARACTER_EMBEDDING_SIZE = 128
USE_TF_EMBEDDING_LOOKUP = True

turkish_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP"
train_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.train"
vali_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.vali"
test_path = "D:/PycharmProjects/TextCategorization/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.test"

fasttext_vocab_path = "D:/PycharmProjects/TextCategorization/fasttext_tr_vocab_cache.dat"
fasttext_model_path = "D:/PycharmProjects/TextCategorization/fasttext_tr_embedding_cache.npy"

tf.flags.DEFINE_string("train_or_restore", RESTORE_TRAINING, "Train or Restore.")
# Data sources
tf.flags.DEFINE_string("training_file", train_path, "Training data source")
tf.flags.DEFINE_string("validation_file", vali_path, "Validation data source")
tf.flags.DEFINE_string("test_file", test_path, "Test data source")
tf.flags.DEFINE_string("fasttext_vocab", fasttext_vocab_path, "Vocabulary source of FastText embeddings")
tf.flags.DEFINE_string("fasttext_model", fasttext_model_path, "FastText embeddings source")

# Model Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate, default(0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Learning rate, default(0.5)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Filter size, default(3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters, default(128)")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size, default(300)")
tf.flags.DEFINE_integer("num_classes", 25, "Number of tasks, default(25)")
tf.flags.DEFINE_integer("top_k_category", 5, "Number of categories to return as result, default(5)")

tf.flags.DEFINE_integer("epoch", 100, "Number of epochs, default(25)")
tf.flags.DEFINE_integer("save_step", 500, "Step to save the model, default(500)")
tf.flags.DEFINE_integer("evaluate_step", 500, "Step to evaluate the model, default(500)")
tf.flags.DEFINE_integer("train_batch_size", 512, "Training batch size, default(512)")
tf.flags.DEFINE_integer("validation_batch_size", 512, "Validation batch size, default(512)")
tf.flags.DEFINE_integer("max_sentence_length", 40, "Maximum sentence length, default(512)")
tf.flags.DEFINE_integer("decay_step", 250, "How many steps before decay learning rate, default(250)")
tf.flags.DEFINE_float("decay_rate", 0.1, "Rate of decay for learning rate, (default: 0.95)")
tf.flags.DEFINE_float("norm_ratio", 1.25,
                      "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)


def main():
    print("Loading training/test data...")
    train_sentences, y_train, vali_sentences, y_vali, test_sentences, y_test = load_train_vali_test_sets(
        FLAGS.training_file, FLAGS.validation_file, FLAGS.test_file)

    vocabulary_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    print("Loading pre-trained fasttext vectors")
    with open(FLAGS.fasttext_vocab, 'rb') as f:
        vocabulary = pickle.load(f)
    embedding = np.load(FLAGS.fasttext_model)

    if RESTORE_TRAINING.upper() == "R":
        print("Restoring the model!")
        model_number = input("Enter the checkpoint number:")
        while not model_number.isdigit() and len(model_number) == 10:
            model_number = input("Invalid format. Re-enter:")
        checkpoint_dir = "runs/" + model_number + "/checkpoints/"
        outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_number))
        vocabulary_processor.restore(os.path.join(outdir, "vocab"))
    else:
        outdir = create_output_directory()
        vocabulary_processor.fit(vocabulary.keys())

    x_train = np.array(list(vocabulary_processor.transform(train_sentences)))
    x_vali = np.array(list(vocabulary_processor.transform(vali_sentences)))
    x_test = np.array(list(vocabulary_processor.transform(test_sentences)))
    vocabulary_size = len(vocabulary)

    print("Train size: ", len(x_train))
    print("Validation size: ", len(x_vali))
    print("Test size: ", len(x_test))
    print("Vocabulary size:", vocabulary_size)

    print("Training phase starts...")
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    cnn = CNN_Categorizer_v2(sentence_length=x_train.shape[1],
                             num_classes=FLAGS.num_classes,
                             vocabulary_size=vocabulary_size,
                             embedding_size=FLAGS.embedding_size,
                             filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                             num_filters=FLAGS.num_filters,
                             l2_reg_lambda=0,
                             embedding_type="nonstatic")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
        #                                            global_step=global_step, decay_steps=FLAGS.decay_step,
        #                                            decay_rate=FLAGS.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads, vars = zip(*optimizer.compute_gradients(cnn.loss))
        # grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
        train_op = optimizer.apply_gradients(zip(grads, vars), global_step=global_step, name="train_op")

    grad_summaries = track_gradients(zip(grads, vars))
    loss_summary, acc_summary, top2_summary, top3_summary, top4_summary, top5_summary = create_summaries(cnn)

    train_summary_op = tf.summary.merge([loss_summary,
                                         acc_summary,
                                         top2_summary,
                                         top3_summary,
                                         top4_summary,
                                         top5_summary,
                                         grad_summaries])
    train_summary_dir = os.path.join(outdir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary,
                                       acc_summary,
                                       top2_summary,
                                       top3_summary,
                                       top4_summary,
                                       top5_summary,
                                       grad_summaries])
    dev_summary_dir = os.path.join(outdir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=1)
    tf.add_to_collection('train_op', train_op)

    if RESTORE_TRAINING.upper() == "R":
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        train_op = tf.get_collection('train_op')[0]
    else:
        checkpoint_dir = os.path.abspath(os.path.join(outdir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if USE_TF_EMBEDDING_LOOKUP:
            vocabulary_processor.save(os.path.join(outdir, "vocab"))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {cnn.input_x: x_batch,
                     cnn.input_y: y_batch,
                     cnn.keep_prob: FLAGS.dropout_keep_prob,
                     cnn.embedding_placeholder: embedding,
                     cnn.batch_norm: True}
        _, step, summaries, loss, accuracy, top2, top3, top4, top5 = sess.run([train_op,
                                                                               global_step,
                                                                               train_summary_op,
                                                                               cnn.loss,
                                                                               cnn.accuracy,
                                                                               cnn.pred_top_2,
                                                                               cnn.pred_top_3,
                                                                               cnn.pred_top_4,
                                                                               cnn.pred_top_5], feed_dict=feed_dict)
        train_summary_writer.add_summary(summaries, step)
        return step, loss, accuracy, top2, top3, top4, top5

    def validation_step(x_validation, y_validation, writer=None):
        validation_batches = create_batches(list(zip(x_validation, y_validation)), FLAGS.validation_batch_size, 1)

        vali_counter, vali_accuracy = 0, 0.0

        for vali_batch in validation_batches:
            x_vali_batch, y_vali_batch = zip(*vali_batch)
            feed_dict_vali = {cnn.input_x: x_vali_batch,
                              cnn.input_y: y_vali_batch,
                              cnn.embedding_placeholder: embedding,
                              cnn.keep_prob: 1.0,
                              cnn.batch_norm: False}
            step, summaries, current_accuracy, top2, top3, top4, top5 = sess.run(
                [global_step, dev_summary_op, cnn.accuracy, cnn.pred_top_2, cnn.pred_top_3,
                 cnn.pred_top_4, cnn.pred_top_5], feed_dict_vali)

            vali_accuracy += current_accuracy

            vali_counter += 1

            print("Validation batch:", vali_counter, "- Batch accuracy", current_accuracy)
            if writer:
                writer.add_summary(summaries, step)
        vali_accuracy /= vali_counter
        return vali_accuracy, top2, top3, top4, top5

    def save_step(current_step):
        print(datetime.datetime.now().isoformat(),
              "Saving!")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print(datetime.datetime.now().isoformat(),
              "Saved model at", current_step, "to", path)
        print("-----------------------------------------------------")

    train_batches = create_batches(list(zip(x_train, y_train)), FLAGS.train_batch_size, FLAGS.epoch)

    num_batch = int((len(x_train) - 1) / FLAGS.train_batch_size) + 1
    print("Number of batches (train):", num_batch)

    best_validation_accuracy = -1
    best_validation_top2 = -1
    best_validation_top3 = -1
    best_validation_top4 = -1
    best_validation_top5 = -1

    for train_batch in train_batches:
        x_batch, y_batch = zip(*train_batch)
        _, train_loss, train_accuracy, train_top2, train_top3, train_top4, train_top5 = train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        print(datetime.datetime.now().isoformat(),
              "Train Step:", current_step,
              "Train Loss:", train_loss,
              "Train Accuracy:", train_accuracy,
              "Train Top-2 Accuracy:", train_top2,
              "Train Top-3 Accuracy:", train_top3,
              "Train Top-4 Accuracy:", train_top4,
              "Train Top-5 Accuracy:", train_top5)
        if current_step % FLAGS.evaluate_step == 0 or current_step // num_batch == FLAGS.epoch:
            print("-----------------------------------------------------")
            print(datetime.datetime.now().isoformat(),
                  "Evaluation!")

            vali_acc, vali_top2, vali_top3, vali_top4, vali_top5 = validation_step(
                x_vali, y_vali, writer=dev_summary_writer)
            print(datetime.datetime.now().isoformat(),
                  "Total validation accuracy:", vali_acc,
                  "Total validation Top-2 accuracy:", vali_top2,
                  "Total validation Top-3 accuracy:", vali_top3,
                  "Total validation Top-4 accuracy:", vali_top4,
                  "Total validation Top-5 accuracy:", vali_top5)
            print("-----------------------------------------------------")
            if best_validation_accuracy < vali_acc:
                best_validation_accuracy = vali_acc
                best_validation_top2 = vali_top2
                best_validation_top3 = vali_top3
                best_validation_top4 = vali_top4
                best_validation_top5 = vali_top5

            print(datetime.datetime.now().isoformat(),
                  "Best validation accuracy:", best_validation_accuracy,
                  "Best validation Top-2 accuracy:", best_validation_top2,
                  "Best validation Top-3 accuracy:", best_validation_top3,
                  "Best validation Top-4 accuracy:", best_validation_top4,
                  "Best validation Top-5 accuracy:", best_validation_top5)

        if current_step % FLAGS.save_step == 0 or current_step // num_batch == FLAGS.epoch:
            if best_validation_accuracy == vali_acc:
                save_step(current_step)
        if current_step % num_batch == 0:
            current_epoch = current_step // num_batch
            print(datetime.datetime.now().isoformat(),
                  "Epoch", current_epoch, "has finished!")
    print("Done!")


def track_gradients(grads_and_vars):
    grad_summaries = list()
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram(
                "{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar(
                "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    return tf.summary.merge(grad_summaries)


def create_summaries(cnn):
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
    top2_summary = tf.summary.scalar("top2", cnn.pred_top_2)
    top3_summary = tf.summary.scalar("top3", cnn.pred_top_3)
    top4_summary = tf.summary.scalar("top4", cnn.pred_top_4)
    top5_summary = tf.summary.scalar("top5", cnn.pred_top_5)
    return loss_summary, acc_summary, top2_summary, top3_summary, top4_summary, top5_summary


def create_output_directory():
    timestamp = str(int(time.time()))
    outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    return outdir


if __name__ == '__main__':
    main()
