from collections import defaultdict
import numpy as np
import six
from gensim.models import FastText
from TextCategorization.preprocessor import Preprocessor


class Vocabulary(object):
    """
    Taken and extended from Tensorflow's categorical_vocabulary.py.
    Note that original version is deprecated and does not provide the necessary support I need for OOV.
    """
    def __init__(self, unknown_token="<UNK>", support_reverse=True, support_unknown_vocab=True):
        self._unknown_token = unknown_token
        self._mapping = {unknown_token: 0}
        self._support_reverse = support_reverse
        self._support_unknown_vocab = support_unknown_vocab
        if support_unknown_vocab:
            self._oov_mapping = {}
        if support_reverse:
            self._reverse_mapping = [unknown_token]
            self._reverse_oov_mapping = []
        self._freq = defaultdict(int)
        self._freeze = False

    def __len__(self):
        """Returns total count of mappings. Including unknown token."""
        if self._support_unknown_vocab:
            return len(self._mapping), len(self._oov_mapping)
        else:
            return len(self._mapping)

    def freeze(self, freeze=True):
        """Freezes the vocabulary, after which new words return unknown token id.

        Args:
          freeze: True to freeze, False to unfreeze.
        """
        self._freeze = freeze

    def get_from_in_vocab_mapping(self, category):
        """Returns word's id in the vocabulary.

        If category is new, creates a new id for it.

        Args:
          category: string or integer to lookup in vocabulary.

        Returns:
          interger, id in the vocabulary.
        """
        if category not in self._mapping:
            if self._freeze:
                return 0
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def get_from_oov_mapping(self, category):
        """Returns word's id out of the vocabulary.

        If category is new, creates a new id for it.

        Args:
          category: string or integer to lookup in vocabulary.

        Returns:
          integer, id for out-of-vocabulary category.
        """
        if category not in self._oov_mapping:
            self._oov_mapping[category] = len(self._mapping) + len(self._oov_mapping)
            if self._support_reverse:
                self._reverse_oov_mapping.append(category)
        return self._reverse_oov_mapping[category]

    def add(self, category, count=1):
        """Adds count of the category to the frequency table.

        Args:
          category: string or integer, category to add frequency to.
          count: optional integer, how many to add.
        """
        category_id = self.get_from_oov_mapping(category)
        if category_id <= 0:
            return
        self._freq[category] += count

    def trim(self, min_frequency, max_frequency=-1):
        """Trims vocabulary for minimum frequency.

        Remaps ids from 1..n in sort frequency order.
        where n - number of elements left.

        Args:
          min_frequency: minimum frequency to keep.
          max_frequency: optional, maximum frequency to keep.
            Useful to remove very frequent categories (like stop words).
        """
        # Sort by alphabet then reversed frequency.
        self._freq = sorted(
            sorted(
                six.iteritems(self._freq),
                key=lambda x: (isinstance(x[0], str), x[0])),
            key=lambda x: x[1],
            reverse=True)
        self._mapping = {self._unknown_token: 0}
        if self._support_reverse:
            self._reverse_mapping = [self._unknown_token]
        idx = 1
        for category, count in self._freq:
            if max_frequency > 0 and count >= max_frequency:
                continue
            if count <= min_frequency:
                break
            self._mapping[category] = idx
            idx += 1
            if self._support_reverse:
                self._reverse_mapping.append(category)
        self._freq = dict(self._freq[:idx - 1])

    def reverse(self, class_id):
        """Given class id reverse to original class name.

        Args:
          class_id: Id of the class.

        Returns:
          Class name.

        Raises:
          ValueError: if this vocabulary wasn't initialized with support_reverse.
        """
        if not self._support_reverse:
            raise ValueError("This vocabulary wasn't initialized with "
                             "support_reverse to support reverse() function.")
        return self._reverse_mapping[class_id]

    def reverse_oov_mapping(self, class_id):
        if not self._support_reverse and self._support_unknown_vocab:
            raise ValueError("This vocabulary wasn't initialized with "
                             "support_reverse to support reverse() function.")
        return self._reverse_oov_mapping[class_id]


def load_fasttext(fasttext_model_path = "D:/PycharmProjects/TextCategorization/wiki.tr",
                  full_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP",
                  language='turkish'):

    preprocessor = Preprocessor(language)
    model = FastText.load_fasttext_format(fasttext_model_path)
    vocabulary = model.wv.vocab

    with open(full_dataset_path, 'r', encoding="utf-8") as file:
        for line in file:
            line_tokens = line.split('\t')
            preprocessed_sentence = preprocessor.preprocess(line_tokens[2])
            print(preprocessed_sentence)
            for token in preprocessed_sentence.split(" "):
                print(token)
                if token not in vocabulary.keys():
                    print(token, "is out of vocabulary")
                    embeddings = np.array(model.wv.word_vec(token))
                    print(embeddings)
                else:
                    print(token, "is in vocabulary")

load_fasttext()