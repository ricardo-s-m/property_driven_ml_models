from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load

import file_io
import os


def get_training_and_test_data(x, y, training_size_rate, random_state):
    if training_size_rate == 1.0:
        x_train = x
        x_test = None
        y_train = y
        y_test = None
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=training_size_rate,
                                                            random_state=random_state,
                                                            shuffle=True,
                                                            stratify=y)

    return x_train, x_test, y_train, y_test


class DecisionTreeModel:

    def __init__(self, random_state):
        self.random_state = random_state
        self.classificator = tree.DecisionTreeClassifier(random_state=self.random_state)
        self.decision_tree = None

    def execute_validator(self, x, y):
        validador = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=self.random_state)
        for train_id, test_id in validador.split(x, y):
            x_train, x_test = x[train_id], x[test_id]
            y_train, y_test = y[train_id], y[test_id]
        return x_train, x_test, y_train, y_test

    def executar_classificador(self, classificador, x_train, x_test, y_train):
        arvore = classificador.fit(x_train, y_train)
        y_pred = arvore.predict(x_test)
        return y_pred, arvore

    def get_model(self, x_train, y_train):
        self.decision_tree = self.classificator.fit(x_train, y_train)

        return self.decision_tree

    def predict(self, x_test):
        y_pred = self.decision_tree.predict(x_test)

        return y_pred

    def create_image(self, image_name, export_directory):
        plt.figure(figsize=(90, 45))
        tree.plot_tree(self.classificator, filled=True, fontsize=14, node_ids=True, precision=12)
        path_fig = file_io.get_fig_directory(image_name, export_directory)
        plt.savefig(path_fig)
        plt.close()

    def validate(self, y_test, y_pred):
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))


    def get_target_model(self, target_ml_model, x_train, y_train):
        # print(f"get_target_model: {target_ml_model}")
        if target_ml_model.name() == 'Decision_Tree':
            learn_algorithm = tree.DecisionTreeClassifier(random_state=self.random_state)
            return learn_algorithm.fit(x_train, y_train)

        elif target_ml_model.name() == 'KNN':
            learn_algorithm = KNeighborsClassifier(n_neighbors=5)
            model = learn_algorithm.fit(x_train, y_train)
            return model

        elif target_ml_model.name() == 'Gaussian_Naive_Bayes':
            learn_algorithm = GaussianNB()
            model = learn_algorithm.fit(x_train, y_train)
            return model

        elif target_ml_model.name() == 'Logistic_Regression':
            learn_algorithm = LogisticRegression(random_state=self.random_state)
            return learn_algorithm.fit(x_train, y_train)

        elif target_ml_model.name() == 'Support Vector Machine':
            learn_algorithm = SVC(kernel='linear', C=1.0, random_state=self.random_state)
            return learn_algorithm.fit(x_train, y_train)

        elif target_ml_model.name() == 'Kernel Support Vector Machine':
            learn_algorithm = SVC(kernel='rbf', C=1.0, random_state=self.random_state)
            return learn_algorithm.fit(x_train, y_train)

        else:
            return None


    def dump_target_model(self, model_name, target_model, export_directory):
        model_name = model_name.lower()
        file_name = model_name + '_model.joblib'
        file_path = os.path.join(export_directory, file_name)
        dump(target_model, file_path)
