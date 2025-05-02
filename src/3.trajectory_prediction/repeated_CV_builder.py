import os

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.utils._testing  import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
class DCV():
    """
    Do Cross validation

    """
    outer_repeats = 5
    inner_repeats = 10
    grid_search_method = GridSearchCV
    cv_outer = StratifiedShuffleSplit(n_splits=outer_repeats, test_size=1 / outer_repeats)
    cv_inner = StratifiedShuffleSplit(n_splits=inner_repeats, test_size=1 / inner_repeats)

    def __init__(self, Model: sklearn.base.BaseEstimator or sklearn.base.ClassifierMixin):
        self.Model = Model
        self.hyperParams = dict()
        self.selected_features = []
        self.reset_model_results()

    def reset_model_results(self):
        """
        creates all variables used and resets them if necessary.
        :return:
        """
        
        self.features = []
        self.indexes = None
        self._train_accuracies = []
        self._train_precision = []
        self._train_recall = []
        self._train_f1 = []
        self.train_accuracies = []
        self.train_precision = []
        self.train_recall = []
        self.train_f1 = []
        self.test_accuracy = []
        self.test_precision = []
        self.test_recall = []
        self.test_f1 = []
        self.all_features = []

    def train_fit(self, data: pd.DataFrame, classes: pd.DataFrame or pd.Series or pd.Index,
                  loop: int = 1, verbose: bool = False):
        """
        the main functions that does the repeated cross validation and does the feature selection.
        first it creates the parameters search space.
        after which it will start doing the repeated cross validation with a stratified shuffle split.

        :param data: the data that the model need to train on. prefferd to just be a pandas Dataframe
        :param classes: the classification of the data. this can be a row or the index of the data DataFrame
        :param loop: nr of repeated CV that will be done
        :param verbose:
        :return:
        """
        data = self.__data_cleaner__(data)
        classes = self.__class_cleaner__(classes)
        self.reset_model_results()

        self.hyperParams_est = {"{}".format(x): self.hyperParams[x] for x in self.hyperParams.keys()}
        self.feature_names = list(data)
        self.indexes = pd.MultiIndex.from_product(self.hyperParams_est.values(), names=self.hyperParams_est.keys())
       
        self.update_progress(0)
        for i in range(loop):
            for j, (outer_train_index, outer_test_index) in enumerate(self.cv_outer.split(data, classes)):
                # splits all the data in to test and training
                outer_train_x, outer_test_x = data.iloc[outer_train_index], data.iloc[outer_test_index]
                outer_train_y, outer_test_y = classes.iloc[outer_train_index], classes.iloc[outer_test_index]
                # creates the pipeline for feature selection and classification
                search = self.grid_search_method(self.Model, self.hyperParams_est, scoring=self.__scoring__, cv=self.cv_inner,
                                      n_jobs=-1, refit="f1_score")
                # fits the data in to the pipeline
                search.fit(outer_train_x, outer_train_y.values.ravel())
                #preprocess = search.best_estimator_.named_steps['preprocess']
                # saves the training data
                self._train_accuracies += [search.cv_results_["mean_test_accuracy"]]
                self._train_precision += [search.cv_results_["mean_test_precision"]]
                self._train_recall += [search.cv_results_["mean_test_recall"]]
                self._train_f1 += [search.cv_results_["mean_test_f1_score"]]
                # tests the best model out of the grid search on the test data
                self.__test__(search, outer_test_x, outer_test_y)
                # saves all the useful data
                #self.all_features += [preprocess.transform(np.arange(len(data.columns)).reshape(1, -1))]
                if verbose:
                    # prints useful information if True
                    print(
                        f"accuracy: {self.test_accuracy[-1][0]};\n"
                        f"precision: {self.test_precision[-1][0]};\n"
                        f"recall: {self.test_recall[-1][0]};\n"
                        f"f1-score: {self.test_f1[-1][0]};\n"
                        f"model: {search.best_estimator_[-1]}")
                    # updates the progress bar
                self.update_progress(progress=(i * self.outer_repeats + j + 1) / (self.outer_repeats * loop))
        # stores all the data in to a pandas DataFrame
        self.train_accuracies = pd.DataFrame(self._train_accuracies, columns=self.indexes)
        self.train_precision = pd.DataFrame(self._train_precision, columns=self.indexes)
        self.train_recall = pd.DataFrame(self._train_recall, columns=self.indexes)
        self.train_f1 = pd.DataFrame(self._train_f1, columns=self.indexes)
        self.test_accuracy = pd.DataFrame(self.test_accuracy, columns=["values"] + list(self.hyperParams.keys()))
        self.test_precision = pd.DataFrame(self.test_precision, columns=["values"] + list(self.hyperParams.keys()))
        self.test_recall = pd.DataFrame(self.test_recall, columns=["values"] + list(self.hyperParams.keys()))
        self.test_f1 = pd.DataFrame(self.test_f1, columns=["values"] + list(self.hyperParams.keys()))


    def __test__(self, model: sklearn.model_selection.GridSearchCV, data, classes):
        """
        calculates all the accuracy, precision, and recall of a given model with data and classes
        :param model: an Sklearn model that is already fitted.
        :param data:
        :param classes:
        :return:
        """
        self.test_accuracy += [
            [accuracy_score(classes, model.predict(data))] + list(model.best_params_.values())]
        self.test_precision += [[precision_score(classes, model.predict(data), average="macro", zero_division=0)] +
                                list(model.best_params_.values())]
        self.test_recall += [[recall_score(classes, model.predict(data), average="macro", zero_division=0)] +
                             list(model.best_params_.values())]
        self.test_f1 += [[f1_score(classes, model.predict(data), average="macro", zero_division=0)] +
                             list(model.best_params_.values())]

    def __data_cleaner__(self, data: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        """
        cleans the data so that it is always a pandas dataframe
        :param data: the data in an numpy array of pandas dataframe
        :return:
        returns a pandas dataframe
        """
        if type(data) == pd.DataFrame:
            # print(data.shape)
            return data
        elif type(data) == pd.Series:
            return data
        elif type(data) == np.ndarray:
            return pd.DataFrame(data)
        raise TypeError("TypeError: data is not correct")

    def __class_cleaner__(self, classes: pd.core or np.ndarray) -> pd.DataFrame:
        """
        cleans the data that consist of the classes so that the script can use the data for most applications
        :param classes: an array (pandas, numpy or pandas series) of all the classes.
        :return:
        returns the classes in a pandas array.
        """
        if type(classes) == pd.core.indexes.base.Index or type(classes) == pd.core.indexes.range.RangeIndex:
            return classes.to_frame()

        elif type(classes) == np.ndarray:
            return pd.DataFrame(classes)
        elif type(classes) == pd.Series:
            return classes.to_frame()
        elif type(classes) == pd.DataFrame:
            return classes
        raise TypeError("TypeError: class data is not correct")

    def save_all_data(self, modelname: str = "model", directory:str=""):
        """
        saves all the data for analyses and figures.
        :param modelname: the name that the files will receive before a static name.
        """
        self.test_accuracy.to_csv(directory+"{}_test_accuracy.csv".format(modelname))
        self.test_precision.to_csv(directory+"{}_test_precision.csv".format(modelname))
        self.test_recall.to_csv(directory+"{}_test_recall.csv".format(modelname))
        self.test_f1.to_csv(directory+"{}_test_f1.csv".format(modelname))
        self.train_accuracies.to_csv(directory+"{}_train_accuracy.csv".format(modelname))
        self.train_precision.to_csv(directory+"{}_train_precision.csv".format(modelname))
        self.train_recall.to_csv(directory+"{}_train_recall.csv".format(modelname))
        self.train_f1.to_csv(directory+"{}_train_f1.csv".format(modelname))
        pd.DataFrame([x[0] for x in self.all_features]).to_csv(directory+"{}_features.csv".format(modelname))

    def __scoring__(self, model: sklearn.base.BaseEstimator or sklearn.base.ClassifierMixin, x: pd.DataFrame, y: np.array) -> dict:
        """
        scoring for the grid search. this function is used during the inner loop to create all the data.
        :param model: the model
        :param x: the data.
        :param y: the classes
        :return:
        :returns the accuracy, precision, and recall for the grid search in a dict format
        """
        y_pred = model.predict(x)
        acc = accuracy_score(y_true=y, y_pred=y_pred)
        pres = precision_score(y_true=y, y_pred=y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true=y, y_pred=y_pred, average="macro", zero_division=0)
        f1_score_value = f1_score(y_true=y, y_pred=y_pred, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": pres, "recall": recall, "f1_score": f1_score_value}

    def update_progress(self, progress: int or float):
        """
        update_progress() : Displays or updates a console progress bar
        Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        :param progress:
        :return:
        """
        barLength = 50  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rTraining progress: [{0}] {1:.1f}% {2}".format("#" * block + "-" * (barLength - block), progress * 100,
                                                                status)
        print(text)
        
        @classmethod
        def formLatex(
            cls,
            acc,
            prec,
            recall,
            index=None,
            header=[],
            add=False
        ):
            if index == None:
                header_text = "\\hline " + " & ".join(["count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"
                table = []
            elif len(index) == 1:
                table = [str(x) for x in index] + ["{}"]
                index = tuple(index)[0]
                header_text = "\\hline " + " & ".join(
                    header + ["{}", "count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"
            else:
                table = [str(x) for x in index]
                index = tuple(index)
                header_text = "\\hline " + " & ".join(header + ["count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"

            for nr, i in enumerate([acc, prec, recall]):
                if index == None:
                    stats = i["values"].agg(['mean', 'count', 'std'])
                else:
                    stats = i.groupby(header).get_group(index)["values"].agg(['mean', 'count', 'std'])
                m, c, s = stats
                ci95_hi = m + 1.95 * s / math.sqrt(c)
                ci95_lo = m - 1.95 * s / math.sqrt(c)
                if nr == 0:
                    table += [str(c)]
                table += ["{:.2f} ({:.2f}-{:.2f})".format(m, ci95_lo, ci95_hi)]
            text = ""
            # print(header_text)
            # print(table)
            table_text = " & ".join(table) + "\\\\"
            # print(table_text)
            if add:
                return table_text
            else:
                text += header_text.replace("_", " ") + "\n"
                text += table_text

            return text


