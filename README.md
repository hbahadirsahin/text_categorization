# Text Categorization experiments for Turkish and English

Dataset: https://data.mendeley.com/datasets/cdcztymf4k/1

Experiment Details:

- For now, my ongoing experiments are consists of a basic TextCNN with FastText embeddings (cite: https://github.com/satojkovic/cnn-text-classification-tf/tree/use_fasttext). 
- Pre-trained FastText embeddings taken from: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- I focused on Turkish versions of the dataset nowadays. Hence, preprocessing part falls short for English. 
- Out-of-Vocabulary (OOV) words are ignored in this first commit. FastText can create embeddings for OOV by using char-ngrams; however, 
it is not trivial to use this feature in a usual flow of the code. Hence, an extended version of tensorflow's VocabularyProcessor will be developed.
- Parameter searches, different architectural changes are also included in my experiments.
- I will share the results (with parameters and network architecture info); however, I won't provide any model files. Obviously, one cannot replicate
my results due to different train/validation/test splits, but I don't think the end results will differ significantly. 


To-do:

- More readable code =) 
- Eventually, this codebase will evolve. I won't stuck to TextCNN. According to my initial plan, basic TextCNN will provide baseline results for the dataset.
- Attention layers, variational dropout, GRU-based models will be added. As a note, I won't use LSTM since GRU provides better results (cite: https://arxiv.org/pdf/1702.01923.pdf).   
- Obviously different kinds of embedding models/language models will be tested too. Sentence embeddings (basically averaging FastText vectors), Elmo and Universal Sentence Encoder are in my list.
