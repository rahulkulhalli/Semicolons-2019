import matplotlib
matplotlib.use("Agg")
import argparse
import pandas as pd
import pickle
import explain
import shap
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug import secure_filename
from werkzeug import SharedDataMiddleware

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="uploads"
app.config['O']="output"

@app.route("/")
def numeric():
    print("started")
    return render_template("index.html")

@app.route("/program",methods=['GET','POST'])
def program():
    if request.method == 'POST':
        data_csv = request.files["data_csv"]
        filename = secure_filename(data_csv.filename)
        data_csv = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #data_csv=data_csv.filename()
        target = request.form["target"]
        model = request.files["model"]
        filename = secure_filename(model.filename)
        model = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #model=model.filename()
        kernel = request.form["text"]
        row = request.files["data_csv_row"]
        filename = secure_filename(row.filename)
        row = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        return("Invalid_input")
    print("success")
    #output path
    out = "./output"

    #target = request.args.get('target')
    # loading model
    model = pickle.load(open( model, 'rb'))

    # loading data-set
    df = pd.read_csv(data_csv)

    # creating list of columns
    features = list(df.columns)
    features.remove(target)

    # creating x and y
    x = df[features]
    y = df[[target]]

    # creating explainer of appropriate type
    e = kernel
    if e == 'k':
        explainer = shap.KernelExplainer(model.predict_proba, x)
    elif e=='t':
        explainer = shap.TreeExplainer(model)
    elif e=='d':
        explainer = shap.DeepExplainer(model)

    # creating object of class Explain
    obj = explain.Explain(model, x, y, explainer, out)

    # calling function to create permutation importance table and data frame
    PI = obj.get_permutation_importance()

    # creating top features list
    top_features = list(PI["Features"])

    # calling function to crate pdp plots of top 2 features
    obj.plot_pdp(PI.iloc[0]["Features"], 1)
    obj.plot_pdp(PI.iloc[1]["Features"], 2)

    # plotting top 2 feature interaction contour pdp plot
    features_2d_plot = top_features[:2]
    obj.plot_2d_pdp(features_2d_plot)

    # reading row if present and calculating individual shap explanation
    try:
        row = pd.read_csv(row)
        obj.print_shap(row.iloc[0], 1)
    except:
        print("single data shap error")

    # creating summary plot for outcome 1
    obj.summary_plot(1)

    # creating shap dependence plot for top important feature with interaction to second most important feature
    obj.shap_dep_plot(top_features,1)
    print(os.getcwd())
    return render_template("result.html",x="../output/dep_plot1.jpg")

app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/output':  app.config['O']
})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8890, debug=True)
