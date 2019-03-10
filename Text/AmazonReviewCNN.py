import numpy as np
# import re
import pickle

from keras.models import load_model
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
import spacy

class AmazonReviewCNN:
    # Initializing
    def __init__(self, path):
        self.model_path = path + "/models/amazon_reviews_char_cnn.h5"
        self.tokenizer_path = path + "/models/amazon_reviews_char_cnn_dictionary.pkl"
        self.output_path = path + "/static/output/"
        self.out_words = "word_level_explanation.png"
        self.out_full_text = "full_text_explanation.html"

        self.text = "default value of text"
        self.page_title = "Amazon Reviews CNN"
        self.class_1_text = "Amongst the fastest for less than half the price, yes this is a nice product. I tested this Samsung 128GB 100MB/s (U3) MicroSD EVO Select extensively . all I can say this is a solid performer on a bargain price."
        self.class_1_label = "Positive"
        self.class_2_text = "I had this card in my phone for less than a month. Suddenly, my phone started restarting itself. One day it went into a restart loop . Would not boot up. After some troubleshooting I removed the SD card and my phone finally restarted. Since I format the card to the phone, I can get nothing off of it. It will not work in my phone. It's a total loss! Don't buy this for a phone."
        self.class_2_label = "Negative"
        self.class_names = ['negative', 'positive']
        
        # load the model here
        self.model = self.load_model_text()
        self.dictionary = self.load_tokenizer_model()
        self.stopwords_spacy = self.load_spacy_stopwords()
        self.explainer = self.create_explainer_object()

    # Deleting (Calling destructor) 
    def __del__(self): 
        print('Destructor called, AmazonReviewLSTM object deleted.') 


    def load_model_text(self):
        return(load_model(self.model_path))


    def load_tokenizer_model(self):
        # loading tokenizer model
        with open(self.tokenizer_path, 'rb') as handle:
            return(pickle.load(handle))


    def load_spacy_stopwords(self):
        nlp = spacy.load('en')
        stopwords_spacy = list(set(nlp.Defaults.stop_words))
        stopwords_spacy.extend(["it's","i've"])
        return stopwords_spacy


    def tokenize_string(self, string):
        string = string.replace(',','').replace('!','').replace('.','').replace("?","")
        return ([word for word in string.split(' ') if word.lower() not in self.stopwords_spacy])


    def prediction_pipeline(self, text_test):
        final_output = []
        for t in text_test:
            review_idx = np.array(self.dictionary.doc2idx(list(t)))
    
            filtered_review_idx = review_idx[review_idx!=-1]
    
            pad_review_idx = pad_sequences([filtered_review_idx], maxlen=800, padding='post', value=-1)[0]
         
            outputs = self.model.predict(np.array([pad_review_idx]))

            final_output.append(outputs[0].tolist())
        return np.array(final_output)

    
    def create_explainer_object(self):
        explainer = LimeTextExplainer(
            split_expression=self.tokenize_string,
            # bow=True,
            class_names=self.class_names
        )
        return explainer


    def explain(self):
        explainer = self.explainer.explain_instance(self.text, classifier_fn=self.prediction_pipeline, num_features=10)
        plot = explainer.as_pyplot_figure()
        plot.tight_layout()
        plot.savefig(self.output_path + self.out_words)
        explainer.save_to_file(self.output_path + self.out_full_text)

        with open(self.output_path + self.out_full_text) as oldfile, open(self.output_path + '/full_explanation.html', 'w') as newfile:
            for line in oldfile:
                if 'exp_div' not in line:
                    newfile.write(line)

        print("Success", self.text)
