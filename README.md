# README

## WARNING!!

This repository is deprecated, since I don't like Tensorflow as I used to be. Check this repository for equivalent but PyTorch version:

https://github.com/hbahadirsahin/nlp-experiments-in-pytorch

## Experiment Details:

- I use the dataset that me and my old colleagues constructed 3 years ago. "English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset" is publicly available: https://data.mendeley.com/datasets/cdcztymf4k/1
- For now, my ongoing experiments are consists of a basic TextCNN with FastText embeddings (cite: https://github.com/satojkovic/cnn-text-classification-tf/tree/use_fasttext). 
- Pre-trained FastText embeddings taken from: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- I focus on Turkish versions of the dataset nowadays. Hence, preprocessing part falls short for English. 
- Out-of-Vocabulary (OOV) words are ignored in this first commit. FastText can create embeddings for OOV by using char-ngrams; however, 
it is not trivial to use this feature in the usual flow of the code. Hence, an extended version of tensorflow's VocabularyProcessor will be developed.
- Parameter searches, different architectural changes are also included in my experiments.
- I will share the results (with parameters and network architecture info); however, I won't provide any model files. Obviously, one cannot replicate my results due to different train/validation/test splits, but I don't think the end results will differ significantly. 
- As far as I tested, the training part is working without any problem. There is an evaluation code piece too, but I could not find time to test it, properly. So, you may encounter weird, illogical, stupid stuff. I'd be happy to fix such issues if anyone out there reports =)

## To-do:

- More readable code =) 
- Leverage DatasetAPI/tf.data in near future (Added 24/06/2018). Since finding a good resource is hard for this API's usage in text, I leave a link in here which I will use as a referance (https://github.com/LightTag/BibSample/blob/master/preppy.py).
- Eventually, this codebase will evolve. I won't stuck to TextCNN. According to my initial plan, basic TextCNN will provide baseline results for the dataset.
- Attention layers, variational dropout, GRU-based models will be added. As a note, I won't use LSTM since GRU provides better results (cite: https://arxiv.org/pdf/1702.01923.pdf).   
- Obviously different kinds of embedding models/language models will be tested too. Sentence embeddings (basically averaging FastText vectors), Elmo and Universal Sentence Encoder are in my list.
- PyTorch version of my model and rest of the code will be prepared. 

## Not-gonna-do:

- Probably, I will leave my pretty local paths alone in the code. Lucky you, you can always change those paths from code or give them as arguments.
- I do not have any plans to use any kind of morphological stuffs for Turkish (such as https://arxiv.org/pdf/1702.03654.pdf). But, you can always give it a shot. 
- For FastText, I won't train my own model but will use the pre-trained model (both for Turkish and English). However, the case may be different for Elmo and Universal Sentence Encoder for Turkish (the last time I checked, both approaches do not have any pre-trained Turkish model).
- Not gonna provide any pre-trained models. 
- Not gonna provide my every single parameter or architectural related experiments. I guess putting only the best parameters and architecture to github is better (hence, there may be too much commented-out code).  But, to prevent code stealing/plagiarism kind of stuff, I may hide tiny, little surprises =)

## How-to-run (Last Update: 16.07.2018)

Well, I wrote lots of stuff but forgot to add how you can use my code (if you want =)).  Current flow may change, so I will keep last updated date at the header. 

#### Creating initial FastText embedding and vocabulary

Working with FastText is a little bit different than word2vec or doc2vec. Also, loading same, pre-trained embedding everytime using Gensim is not a good way to waste your time =) The solution is `fasttext_preload.py` file (the citation is at the beginning). 

- One can use `python fasttext_preload.py -ft-path -v -vocab-path -embedding-path` to load whole pre-trained embedding files and save embedding cache (in ".npy" format) and vocabulary file (in ".dat" format). 
  - `-ft-path` is the original FastText embedding files path (both .vec and .bin files. One can check the initial value of this from code for better understanding).
  - `-vocab-path` is the output path for the vocabulary cache file.
  - `-embedding-path` is the output path for the embedding cache file
- Loading ".npy" file is much faster than using Gensim for only to get embeddings. 
- Note that, vocabulary file only contains the words in the "pre-trained" model, not the words in your dataset. 
- In addition, there are several methods in the file for a future update (the one about OOV words and their embeddings). You can ignore them.

#### Creating deterministic training/validation/test sets 

I wanted to use this file as my helper class for reading the whole dataset file (note that the dataset I am using does not have any splits). However, getting random splits in my experiments (every time I execute the code), it was hard for me to compare the results, easily.  For comparable results, I decided to create a split and save it for future experimental runs. If you also want to use it, you can use `python load_dataset.py -dataset-path -train-path -vali-path -test-path` line. 

  - `-dataset-path` is the input dataset's path. 
  - `-train-path` is the output path for the training set file.
  - `-vali-path` is the output path for the validation set file
  - `-test-path` is the output path for the test set file

Note that initial splits are 70%, 15%, 15% for training, validation and test sets, respectively. One can change them manually from the code (but I guess I will make it an argument in near future).

#### Training

`main.py` contains all the stuffs for the training phase. Since I am a lazy dude, I do not want to explain each argument/flag for now. One can just check the file for their explanations and initial values. Obviously, do not forget to change the input and output paths. 

- Code asks for the paths of training, validation and test files as input (I'll remove test set from here in near future). 
- When you run the code, you will be asked to type "(T)"rain or "(R)"restore. If you type "T", then the code will start training from scratch. If you want to continue your training from the last saved model, type "R" first and then type your saved model's number (the folder name).
- Almost, all input, output and CNN-based parameters are defined as FLAG. However, embedding_type and l2_reg_lambda are not arguments for now (there is no reason for it actually =)).
- If one wants to save more than one checkpoints, find `saver = tf.train.Saver(max_to_keep=1)` line and change the parameter. 

#### Evaluation

`eval_main.py` contains the stuffs for the evaluation. You can either evaluate a test set to get metric results (accuracy/top-k accuracy, etc.) or write sentences to find out their predicted categories. 

- You will be asked to type "S" or "I". "S" is for evaluation of a test set and "I" is for an interactive session to get only predictions to the input sentences. 
- You will be asked the model's number to evaluate. 
- Until I find a better way (actually I did but have not implemented), this code will read both training and test sets at the beginning. The reason is getting same integers for the labels. 
- Evaluation is bugged for "top-k accuracy". Hence, you can only get "accuracy" metric for the test set evaluation for a while.  

#### Encountered a bug/Have a question/Code is rubbish?

Just let me know =)
