# Text Categorization experiments for Turkish and English

Experiment Details:

- I use the dataset that me and my old colleagues constructed 3 years ago. Dataset is publicly available: https://data.mendeley.com/datasets/cdcztymf4k/1
- For now, my ongoing experiments are consists of a basic TextCNN with FastText embeddings (cite: https://github.com/satojkovic/cnn-text-classification-tf/tree/use_fasttext). 
- Pre-trained FastText embeddings taken from: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- I focus on Turkish versions of the dataset nowadays. Hence, preprocessing part falls short for English. 
- Out-of-Vocabulary (OOV) words are ignored in this first commit. FastText can create embeddings for OOV by using char-ngrams; however, 
it is not trivial to use this feature in a usual flow of the code. Hence, an extended version of tensorflow's VocabularyProcessor will be developed.
- Parameter searches, different architectural changes are also included in my experiments.
- I will share the results (with parameters and network architecture info); however, I won't provide any model files. Obviously, one cannot replicate my results due to different train/validation/test splits, but I don't think the end results will differ significantly. 
- As far as I tested, the training part is working without any problem. There is an evaluation code piece too, but I could not find time to test it, properly. So, you may encounter weird, illogical, stupid stuff. I'd be happy to fix such issues if anyone out there reports =)

To-do:

- More readable code =) 
- Eventually, this codebase will evolve. I won't stuck to TextCNN. According to my initial plan, basic TextCNN will provide baseline results for the dataset.
- Attention layers, variational dropout, GRU-based models will be added. As a note, I won't use LSTM since GRU provides better results (cite: https://arxiv.org/pdf/1702.01923.pdf).   
- Obviously different kinds of embedding models/language models will be tested too. Sentence embeddings (basically averaging FastText vectors), Elmo and Universal Sentence Encoder are in my list.
- PyTorch version of my model and rest of the code will be prepared.

Not-gonna-do:

- Probably, I will leave my pretty local paths alone in the code. Lucky you, you can always change those paths from code or give them as arguments.
- I do not have any plans to use any kind of morphological stuffs for Turkish (such as https://arxiv.org/pdf/1702.03654.pdf). But, you can always give it a shot. 
- For FastText, I won't train my own model but will use the pre-trained model (both for Turkish and English). However, the case may be different for Elmo and Universal Sentence Encoder for Turkish (the last time I checked, both approaches do not have any pre-trained Turkish model).
- Not gonna provide any pre-trained models. 
- Not gonna provide my every single parameter or architectural related experiments. I guess putting only the best parameters and architecture to github is better (hence, there may be too much commented-out code).  But, to prevent code stealing/plagiarism kind of stuff, I may hide tiny, little surprises =)
