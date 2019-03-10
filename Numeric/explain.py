import matplotlib.pyplot as plt
from pdpbox import pdp
from eli5.sklearn import PermutationImportance
import eli5
import pandas as pd
import shap

class Explain:
    def __init__(self, model, x, y, explainer, out):
        self.model = model
        self.x = x
        self.y = y
        self.explainer = explainer
        self.out = out

    # function to create permutation importance table and data frame
    def get_permutation_importance(self):
        # calculating permutation importance
        perm = PermutationImportance(self.model, random_state=1).fit(self.x, self.y)

        # saving Permutation Importance table
        html_str=eli5.show_weights(perm, feature_names=list(self.x.columns)).data
        html_file = open(self.out + "/permutation_importance.html", "w")
        html_file.write(html_str)
        html_file.close()

        # creating data frame for weights and features
        PI = pd.DataFrame(perm.feature_importances_, columns=["Weights"])
        PI["Features"] = list(self.x.columns)
        PI = PI.sort_values("Weights", ascending=False)
        return PI

    # function to create pdp plots
    def plot_pdp(self, feature_to_plot, i):
        # creating data to plot
        pdp_feature = pdp.pdp_isolate(model=self.model, dataset=self.x, model_features=list(self.x.columns), feature=feature_to_plot)

        # plot it
        pdp.pdp_plot(pdp_feature, feature_to_plot)

        # saving the plot
        plt.tight_layout()
        plt.savefig(self.out + '/dep_plot' + str(i) + '.jpg', dpi=400)
        plt.close()

    # function to create 2D pdp
    def plot_2d_pdp(self, features_2d_plot):
        # creating data to plot
        inter = pdp.pdp_interact(model=self.model, dataset=self.x, model_features=list(self.x.columns), features=features_2d_plot)

        # plot it
        plot_params = {
            # plot title and subtitle
            'title_fontsize': 15,
            'subtitle_fontsize': 12,
            # color for contour line
            'contour_color': 'white',
            'font_family': 'Arial',
            # matplotlib color map for interact plot
            'cmap': 'viridis',
            # fill alpha for interact plot
            'inter_fill_alpha': 0.8,
            # fontsize for interact plot text
            'inter_fontsize': 9,
        }
        pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_2d_plot, plot_type='contour',plot_params=plot_params)

        # saving the plot
        plt.tight_layout()
        plt.savefig(self.out + '/dep_plot_2d.jpg', dpi=300)
        plt.close()

    # creating function to create shap value for a particular row and outcome 1
    def print_shap(self, data_for_pred, outcome):
        #shap.initjs()
        shap_values = self.explainer.shap_values(data_for_pred)
        shap.save_html(self.out + "/individual_shap.html", shap.force_plot(self.explainer.expected_value[outcome], shap_values[outcome], data_for_pred))

    # creating function to create summary plot
    def summary_plot(self, outcome):
        shap_values = self.explainer.shap_values(self.x)
        shap.summary_plot(shap_values[outcome], self.x, show=False)
        plt.tight_layout()
        plt.savefig(self.out + "/summary_plot.jpg", dpi=400)
        plt.close()

    # creating function to create dependence plot
    def shap_dep_plot(self, top_features, outcome):
        shap_values = self.explainer.shap_values(self.x)
        shap.dependence_plot(top_features[0], shap_values[outcome], self.x, interaction_index=top_features[1], show=False)
        plt.tight_layout()
        plt.savefig(self.out + "/shap_dependence.jpg", dpi=400)
        plt.close()
