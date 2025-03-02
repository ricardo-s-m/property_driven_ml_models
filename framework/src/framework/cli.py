import argparse
import os
import copy
from data import Data, InputData, load_built_in_data
from scikit_strategy import DecisionTreeClassifierStrategy, GaussianNaiveBayesStrategy, KnnClassifierStrategy, \
    LogisticRegressionStrategy, SupportVectorMachineStrategy, KernelSupportVectorMachineStrategy
from hypothesis_strategy import SimpleTypeGenerationStrategy, TupleTypeGenerationStrategy
from data import load_built_in_data
from file_io import read_file_as_np, read_file_as_df, create_export_directory, create_complete_export_directory


class CLIError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message):
        self.message = message


class InputArgumentError(CLIError):
    """Exception raised for errors in the input argument.

    Attributes:
        argument -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class CLI:

    def __init__(self):
        self.args = self.__parse_args()

    def __parse_args(self):
        parser = argparse.ArgumentParser(description='Generate test data for machine learning models',
                                         fromfile_prefix_chars='@')

        parser.add_argument('class_name',
                            help='name of the test class that will be generated')
        parser.add_argument('module_name',
                            help='name of the file (python module) that will be generated')
        parser.add_argument('image_name',
                            help='name of the decision tree image that will be generated')
        parser.add_argument('-c',
                            '--criteria',
                            choices=['DTC', 'BVA', 'DTC,BVA', 'DTC, BVA'],
                            default='DTC,BVA',
                            help='decision tree coverage criteria used to derive test cases (default: DTC,BVA)')
        parser.add_argument('-i',
                            '--internal_ml_model',
                            choices=['Decision Tree', 'Random Forest'],
                            default='Decision Tree',
                            help='machine learning model used to derive test cases (default: Decision Tree)')
        parser.add_argument('-m',
                            '--target_ml_model',
                            choices=['Decision Tree',
                                     'GaussianNB',
                                     'KNN',
                                     'Logistic Regression',
                                     'Support Vector Machine',
                                     'Kernel Support Vector Machine'],
                            default='Decision Tree',
                            help='machine learning model that will be tested (default: Decision Tree)')
        parser.add_argument('-b',
                            '--built_in_data',
                            choices=['iris',
                                     'breast_cancer',
                                     'wine',
                                     'digits'],
                            default='iris',
                            help='internal dataset used to train the model (default: iris)')
        parser.add_argument('-g',
                            '--generation_strategy',
                            choices=['simple_values',
                                     'tuple_of_values'],
                            default='simple_values',
                            help='used to indicate whether the hypothesis will generate a value for each feature or a '
                                 'tuple of values for each feature (default: simple_values)')
        parser.add_argument('-r',
                            '--random_state',
                            default=42,
                            type=int,
                            help='random state number (int) used in scikit-learn algorithms (default: 42)')
        parser.add_argument('-n',
                            '--n_samples_per_test',
                            default=100,
                            type=int,
                            help='value (int) that determines how many samples will be generated by the hypothesis in '
                                 'each test (property). (default: 100)')
        parser.add_argument('-t',
                            '--training_size_rate',
                            default='0.9',
                            type=float,
                            help='value between (0.5, 1.0) that will be used to determine the size of the training '
                                 'dataset. Where, 0.5 means that 50% of the samples from the input dataset will be '
                                 'used in the training dataset and 1.0 means that 100% of the samples will be used. '
                                 'By default 90% of the samples are used in the training dataset. (default: 0.9)')
        parser.add_argument('-l',
                            '--boundary_value_rate',
                            default='20.0',
                            type=float,
                            help='value between (1.0, 100.0) that will be used to determine the restriction of the size '
                                 'of the intervals generated when using the BVA criterion. 0.1 means that the ranges '
                                 'generated by the BVA criterion will be 99% smaller (stricter) than the DTC criterion. '
                                 '1.0 means that the intervals generated by the BVA criterion will be identical to '
                                 'those generated by the DTC. (default: 80.0)')
        parser.add_argument('-f',
                            '--float_as_decimal',
                            action='store_true',
                            help='used to indicate whether the hypothesis should generate values of type decimal '
                                 'instead of values of type float')
        parser.add_argument('-e',
                            '--experimentation',
                            action='store_true',
                            help='used to indicate whether generated test cases will be used in a research project')
        parser.add_argument('-d',
                            '--destination_directory',
                            help='used to indicate where to save the files that will be generated by the framework')
        parser.add_argument('-s',
                            '--source_file',
                            default='',
                            type=str,
                            help='used to indicate the file containing the input data')
        parser.add_argument('-u',
                            '--update_precision',
                            action='store_true',
                            help='update the number of decimal places of Decimal values when using the '
                                 '"--treat_flaot_as_decimal" option')
        parser.add_argument('-x',
                            '--export_target_model',
                            choices=['True', 'False'],
                            default='True',
                            help='used to indicate whether the target model should be exported')

        return copy.deepcopy(parser.parse_args().__dict__)

    def process_args(self):
        if not self.args['destination_directory'] or len(self.args['destination_directory'].strip()) == 0:
            self.args['destination_directory'] = os.getcwd()
        if not os.path.isdir(self.args['destination_directory']):
            raise InputArgumentError('The provided destination directory was not found. Check the value given in the '
                                     'argument: -d or --destination_directory.')

        if self.args['source_file'] != '' and not os.path.isfile(self.args['source_file']):
            raise FileNotFoundError(self.args['source_file'])

        return Settings(self.args)


class Settings:

    def __init__(self, args):
        self.args = args
        self.class_name = args['class_name']
        self.module_name = args['module_name']
        self.image_name = args['image_name']
        self.criteria = args['criteria']
        self.internal_ml_model = args['internal_ml_model']
        self.target_ml_model = self.__init_target_ml_model()
        self.built_in_data = args['built_in_data']
        self.generation_strategy = None
        self.random_state = args['random_state']

        self.n_samples_per_test = args['n_samples_per_test']
        self.training_size_rate = args['training_size_rate']
        self.boundary_value_rate = args['boundary_value_rate']
        self.float_as_decimal = args['float_as_decimal']
        self.experimentation = args['experimentation']
        self.destination_directory = args['destination_directory']
        self.source_file = args['source_file']
        self.update_precision = args['update_precision']

        if args['export_target_model'] == 'True':
            self.export_target_model = True
        else:
            self.export_target_model = False

        self.data = None
        self.export_directory = self.__init_export_directory()
        self.__init_data()

        print(f"criteria: {self.criteria}")

    def __init_internal_ml_model(self):
        pass

    def __init_target_ml_model(self):
        if self.args['target_ml_model'] == 'Decision Tree':
            return DecisionTreeClassifierStrategy()
        elif self.args['target_ml_model'] == 'GaussianNB':
            return GaussianNaiveBayesStrategy()
        elif self.args['target_ml_model'] == 'KNN':
            return KnnClassifierStrategy()
        elif self.args['target_ml_model'] == 'Logistic Regression':
            return LogisticRegressionStrategy()
        elif self.args['target_ml_model'] == 'Support Vector Machine':
            return SupportVectorMachineStrategy()
        elif self.args['target_ml_model'] == 'Kernel Support Vector Machine':
            return KernelSupportVectorMachineStrategy()

    def __init_data(self):
        if not self.source_file or len(self.source_file.strip()) == 0:
            data, image_name, class_name = load_built_in_data(self.args['built_in_data'])
            self.data = data
        else:
            input_df = read_file_as_df(self.source_file)
            self.data = Data(input_df)

    def __init_export_directory(self):
        if os.path.isdir(self.destination_directory):
            return create_complete_export_directory(self.destination_directory,
                                                    self.module_name,
                                                    self.float_as_decimal,
                                                    self.target_ml_model.name().lower())
        else:
            raise InputArgumentError('The provided destination directory was not found. Check the value given in the '
                                     'argument: -d or --destination_directory.')


    def conf_generation_strategy(self, input_data, criteria):
        if self.args['generation_strategy'] == 'simple_values':
            self.generation_strategy = SimpleTypeGenerationStrategy(input_data, criteria)
        elif self.args['generation_strategy'] == 'tuple_of_values':
            self.generation_strategy = TupleTypeGenerationStrategy(input_data, criteria)

        return self.generation_strategy
