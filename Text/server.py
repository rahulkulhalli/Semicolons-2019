import os

from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

from werkzeug import secure_filename
from werkzeug import SharedDataMiddleware

from AmazonReviewLSTM import AmazonReviewLSTM
from DrugReviewLR import DrugReviewLR
from AmazonReviewCNN import AmazonReviewCNN
from DrugReviewRF import DrugReviewRF

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

OUTPUT_DIR = '/output'

app = Flask(__name__)
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
api = Api(app)


CORS(app)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/text")
def text():
    return render_template("text.html", test_var=0)

@app.route("/image")
def image():
    return render_template("image.html", test_var=0)

@app.route("/text/model/<model_id>", methods=['GET', 'POST'])
def text_model(model_id):
    model = get_model("text", model_id)
    show_explanation = None
    if request.method == 'POST':
        input_text = request.form['sample_text']
        key = g_model_keys_dict["text"][model_id]["name"]
        model.text = input_text
        model.explain()
        show_explanation = "initial"
    return render_template("text_model.html", model_id=model_id, page_title=model.page_title, class_1_text=model.class_1_text, class_1_label=model.class_1_label, class_2_text=model.class_2_text, class_2_label=model.class_2_label, show_explanation=show_explanation)

def get_model(component, model_id):
    print("Received GET_MODEL request for component : " + component + " and model : " + model_id)
    if "model" not in g_model_keys_dict[component][model_id]:
        g_model_keys_dict[component][model_id]["model"] = create_model(g_model_keys_dict[component][model_id]["name"])
    return g_model_keys_dict[component][model_id]["model"]

def create_model(model_name):
    print("Received CREATE_MODEL request for model : " + model_name)
    if model_name == "amazon_review_lstm":
        return AmazonReviewLSTM(".")
    elif model_name == "amazon_review_char_level_cnn":
        return AmazonReviewCNN(".")
    elif model_name == "drug_review_lr":
        return DrugReviewLR(".")
    elif model_name == "drug_review_rf":
        return DrugReviewRF(".")



g_model_keys_dict = {
  "text" : {
    # "0" : {
    #   "name" : "amazon_review_lstm",
    #   "model" : obj
    #   },
    "1" : {
      "name" : "amazon_review_lstm",
      },
    "2" : {
      "name" : "amazon_review_char_level_cnn",
      },
    "3" : {
      "name" : "drug_review_lr",
      },
    "4" : {
      "name" : "drug_review_rf",
      }
  },
  "image" : {
    "1" : {
      "name" : "malaria",
      },
    "2" : {
      "name" : "pneumonia",
      }
  },
}

app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/output':  app.config['OUTPUT_DIR']
})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8889, debug=True)
