from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re, string, unicodedata
import nltk
import inflect

class TextNomalizer(BaseEstimator, TransformerMixin):
    def __init__(self, language='english'):
        self.stopwords  = stopwords
        self.tokenizer = RegexpTokenizer(r'\w+')

    # Define normalization function

    #https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
    #def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        #new_words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        #return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word.lower())
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = [word for word in words if word not in self.stopwords]
        return new_words

    def normalize(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        #words = self.remove_non_ascii(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, sentences):
        for sent in sentences:
            yield self.normalize(str(sent))