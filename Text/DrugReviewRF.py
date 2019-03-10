import gc
from joblib import load
import spacy
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer


class DrugReviewRF:
    # Initializing
    def __init__(self, path):
        self.model_path = path + "/models/drug_reviews_rf.pkl"
        self.output_path = path + "/static/output/"
        self.out_words = "word_level_explanation.png"
        self.out_full_text = "full_text_explanation.html"

        self.text = "default value of text"
        self.page_title = "Drug Reviews Random Forest Classifier"
        self.class_1_text = "My pain management doctor put me on Butrans patches about weeks ago 5 mg dose. The first box of four was a lifesaver. No more agony at work. Able to sleep. more in two weekends than I had in two years. I am hoping to bump up to the ten mg dose soon to cut down on my Norco. I have had chronic pain for many years and have been through many medicines including Oxycontin. This patch is the best so far."
        self.class_1_label = "Positive"
        self.class_2_text = "I have had  nothing but problems with the Keppera : constant shaking in my arms legs pins needles feeling in my arms legs severe light headedness no appetite etc"
        self.class_2_label = "Negative"
        self.class_names = ['negative', 'positive']

        # load the model here
        self.model = self.load_model_text()
        self.stopwords_spacy = self.load_spacy_stopwords()
        self.explainer = self.create_explainer_object()


    # Deleting (Calling destructor) 
    def __del__(self): 
        print('Destructor called, DrugReviewRF object deleted.') 


    def load_model_text(self):
        with open(self.model_path, 'rb') as handle:
            model = pickle.load(handle)
        return(model)


    def load_tokenizer_model(self):
        # loading tokenizer model
        pass


    def load_spacy_stopwords(self):
        nlp = spacy.load('en')
        stopwords_spacy = list(set(nlp.Defaults.stop_words))
        stopwords_spacy.extend(["it's","i've"])
        return stopwords_spacy


    def tokenize_string(self, string):
        string = string.replace(',','').replace('!','').replace('.','').replace("?","")
        return ([word for word in string.split(' ') if word.lower() not in self.stopwords_spacy])


    def create_explainer_object(self):
        explainer = LimeTextExplainer(
            split_expression= self.tokenize_string,
            # bow=True,
            class_names= self.class_names
        )
        return explainer


    def explain(self):
        explainer = self.explainer.explain_instance(self.text, classifier_fn=self.model.predict_proba, num_features=10)
        plot = explainer.as_pyplot_figure()
        plot.tight_layout()
        plot.savefig(self.output_path + self.out_words)
        explainer.save_to_file(self.output_path + self.out_full_text)

        with open(self.output_path + self.out_full_text) as oldfile, open(self.output_path + '/full_explanation.html', 'w') as newfile:
            for line in oldfile:
                if 'exp_div' not in line:
                    newfile.write(line)

        print("Success", self.text)
