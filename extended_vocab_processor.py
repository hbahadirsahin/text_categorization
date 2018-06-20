from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import six
from tensorflow.python.platform import gfile

try:
    # pylint: disable=g-import-not-at-top
    import cPickle as pickle
except ImportError:
    # pylint: disable=g-import-not-at-top
    import pickle


TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

def tokenizer(iterator):
    """Tokenizer generator.

    Args:
      iterator: Input iterator with strings.

    Yields:
      array of tokens per each value in the input.
    """
    for value in iterator:
        yield value.split(" ")