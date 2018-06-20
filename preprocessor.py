from nltk.corpus import stopwords
import string
import re

class Preprocessor:

    def __init__(self, language):
        self.language = language
        self.__create_stop_words()

    def __create_stop_words(self):
        self.stop_words = stopwords.words(self.language)

    def remove_new_line(self, sentence):
        return sentence.replace('\n', ' ').replace('\r', '')

    def remove_punctuations(self, sentence):
        return "".join([ch for ch in sentence if ch not in string.punctuation])

    def remove_multiple_white_spaces(self, sentence):
        return ' '.join(sentence.split())

    def lowercase_sentence(self, sentence):
        return sentence.lower()

    def remove_digits(self, sentence):
        # ''.join(filter(lambda c: not c.isdigit(), sentence))
        return ''.join([c for c in sentence if not c.isdigit()])

    def remove_stop_words(self, sentence):
        return ' '.join([i for i in sentence.split() if i not in self.stop_words])

    def change_currency_chars(self, sentence):
        return sentence.replace('$', 'dolar').replace('£', 'sterlin').replace('€', 'euro')

    def remove_non_latin(self, sentence):
        regex = re.compile(r'[A-Za-z0-9]')
        return regex.sub("", sentence)

    def preprocess(self, raw_sentence):
        preprocessed_sentence = self.lowercase_sentence(raw_sentence)
        preprocessed_sentence = self.remove_new_line(preprocessed_sentence)
        preprocessed_sentence = self.change_currency_chars(preprocessed_sentence)
        preprocessed_sentence = self.remove_punctuations(preprocessed_sentence)
        preprocessed_sentence = self.remove_digits(preprocessed_sentence)
        preprocessed_sentence = self.remove_stop_words(preprocessed_sentence)
        preprocessed_sentence = self.remove_multiple_white_spaces(preprocessed_sentence)
        return preprocessed_sentence.strip()


if __name__ == '__main__':
    preprocessor = Preprocessor('turkish')
    turkish_dataset_path = "D:/TWNERTC_All_Versions/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP"
    category_labels = list()
    entity_labels = list()
    sentences = list()

    with open(turkish_dataset_path, encoding="utf-8") as dataset:
        for line in dataset:
            line_tokens = line.split('\t')
            category_labels.append(line_tokens[0])
            entity_labels.append(line_tokens[1])
            sentence = line_tokens[2]
            print(preprocessor.preprocess(sentence))


