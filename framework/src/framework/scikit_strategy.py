import file_io


class KFoldStrategy:

    def __init__(self):
        pass

    def imports(self, is_k_fold_used):
        if is_k_fold_used:
            return 'from sklearn.model_selection import cross_val_score\n'
        else:
            return ''

    def k_fold(self, classifier_name, x_name, y_name):
        return 'request.cls.k_fold_scores = cross_val_score(' + classifier_name + ', ' \
                                    + x_name + ', ' \
                                    + y_name + ', ' \
                                    + 'cv=10, scoring="accuracy")\n'


class GenericClassifierStrategy:

    def name(self) -> str:
        return

    def imports(self) -> str:
        return

    def classifier(self) -> str:
        return

    def image(self) -> str:
        return


class DecisionTreeClassifierStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Decision_Tree'

    def imports(self) -> str:
        str_imports = 'from sklearn import tree\n'
        str_imports += 'import matplotlib.pyplot as plt\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = tree.DecisionTreeClassifier(random_state=0)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        experiment_dt_image_path = file_io.get_experiment_dt_image_path(export_directory)

        # Temporary Lines
        str_image = '        plt.figure(figsize=(29, 14))\n'
        str_image += '        tree.plot_tree(request.cls.classifier, filled=True, fontsize=14)\n'
        str_image += '        plt.savefig(\'' + experiment_dt_image_path + '\')\n'
        str_image += '        plt.close()\n'

        return str_image


class KnnClassifierStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'KNN'

    def imports(self) -> str:
        str_imports = 'from sklearn.neighbors import KNeighborsClassifier\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = KNeighborsClassifier(n_neighbors=3)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class LogisticRegressionStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Logistic_Regression'

    def imports(self) -> str:
        str_imports = 'from sklearn.linear_model import LogisticRegression\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = LogisticRegression(C=100.0, random_state=0)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class SupportVectorMachineStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Support_Vector_Machine'

    def imports(self) -> str:
        str_imports = 'from sklearn.svm import SVC\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = SVC(kernel=\'linear\', C=1.0, random_state=0)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class KernelSupportVectorMachineStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Kernel_Support_Vector_Machine'

    def imports(self) -> str:
        str_imports = 'from sklearn.svm import SVC\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = SVC(kernel=\'rbf\', C=1.0, random_state=0)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class GaussianProcessStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Gaussian_Process'

    def imports(self) -> str:
        str_imports = 'from sklearn.gaussian_process import GaussianProcessClassifier\n'
        str_imports += 'from sklearn.gaussian_process.kernels import RBF\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = GaussianProcessClassifier(1.0 * RBF(1.0))\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class GaussianNaiveBayesStrategy(GenericClassifierStrategy):

    def name(self) -> str:
        return 'Gaussian_Naive_Bayes'

    def imports(self) -> str:
        str_imports = 'from sklearn.naive_bayes import GaussianNB\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = GaussianNB()\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class QuadraticDiscriminantAnalysisStrategy(GenericClassifierStrategy):

    def imports(self) -> str:
        str_imports = 'from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = QuadraticDiscriminantAnalysis()\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


# Neural Network
class MLPStrategy(GenericClassifierStrategy):

    def imports(self) -> str:
        str_imports = 'from sklearn.neural_network import MLPClassifier\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = MLPClassifier(alpha=1, max_iter=1000)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


# ensemble models
class RandomForestStrategy(GenericClassifierStrategy):

    def imports(self) -> str:
        str_imports = 'from sklearn.ensemble import RandomForestClassifier\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''


class AdaBoostStrategy(GenericClassifierStrategy):

    def imports(self) -> str:
        str_imports = 'from sklearn.ensemble import AdaBoostClassifier\n'
        return str(str_imports)

    def classifier(self) -> str:
        str_classifier = '        request.cls.classifier = AdaBoostClassifier()\n'
        return str(str_classifier)

    def image(self, export_directory) -> str:
        return ''
