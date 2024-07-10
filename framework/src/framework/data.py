import numpy as np
import pandas as pd
import sys
from sklearn import datasets
from decimal import *
import util
import copy
import random
import collections
import random
import math

class FeatureValuesSpecifier:

    def __init__(self, id, min_value, max_value, treat_flaot_as_decimal):
        self.id = id
        self.min_value = min_value
        self.max_value = max_value
        self.min_is_exclusive = False
        self.max_is_exclusive = False
        self.min_is_derived_from_tree = False
        self.max_is_derived_from_tree = False
        self.type = None
        self.generate_float_as_decimal = treat_flaot_as_decimal
        self.__n_decimal_places = 1

        self.min_bva_value = None
        self.max_bva_value = None
        self.min_bva_is_derived_from_min_value = False
        self.max_bva_is_derived_from_max_value = False
        self.min_bva_is_derived_from_training_data = False
        self.max_bva_is_derived_from_training_data = False

    def get_tuple_values(self):
        tuple_values = (self.min_value, self.max_value)
        return tuple_values

    def update_values(self, min_value, max_value):
        if self.min_value == sys.float_info.min:
            self.min_value = min_value
        if self.max_value == sys.float_info.max:
            self.max_value = max_value

    def set_min_bva_value(self, min_bva):
        if min_bva is not None:
            if min_bva == self.min_value:
                self.min_bva_is_derived_from_min_value = True

            self.min_bva_value = max([min_bva, self.min_value])

    def set_max_bva_value(self, max_bva):
        if max_bva is not None:
            if max_bva == self.max_value:
                self.max_bva_is_derived_from_max_value = True

            self.max_bva_value = min([max_bva, self.max_value])

    def update_feature_type(self, feature_type):
        np_int_types = (np.int0, np.int8, np.int16, np.int32, np.int64)
        np_float_types = (np.float16, np.float32, np.float64)

        if feature_type == int:
            self.type = 'int'
        if feature_type in np_int_types:
            self.type = 'np_int'
        if feature_type == float:
            self.type = 'float'
        if feature_type in np_float_types:
            self.type = 'np_float'


    def get_hypothesis_decorator(self, criteria):
        min_ = None
        max_ = None

        if criteria == 'dtc':
            min_ = self.min_value
            max_ = self.max_value
        else:
            min_ = self.min_bva_value
            max_ = self.max_bva_value

        decorator = ''

        if self.type == 'float' or self.type == 'np_float' or self.type == 'int' or self.type == 'np_int':
            if not self.generate_float_as_decimal:
                decorator = 'floats(min_value=' + str(min_) + ', max_value=' + str(max_)

                if self.min_is_exclusive:
                    decorator += ', exclude_min=True'
                if self.max_is_exclusive:
                    decorator += ', exclude_max=True'

                decorator += ', allow_nan=False)'
            else:
                decorator = 'decimals(min_value=' + str(min_) + ', max_value=' + str(max_)

                decorator += ', places=' + str(self.__n_decimal_places) + ', allow_nan=False, allow_infinity=False)'
        else:
            decorator = ''
        
        return decorator

    @property
    def n_decimal_places(self):
        return self.__n_decimal_places

    @n_decimal_places.setter
    def n_decimal_places(self, n_decimal_places):
        self.__n_decimal_places = n_decimal_places

    def update_precision(self, n_decimal_places: int, float_as_decimal):
        self.__n_decimal_places = n_decimal_places + 1

        if n_decimal_places < 6:
            n_decimal_places = n_decimal_places + 1
        else:
            n_decimal_places = 6

        decimal_places = util.create_decimal_with(n_decimal_places)
        
        min_value_decimal = util.create_decimal(self.min_value)
        max_value_decimal = util.create_decimal(self.max_value)

        if self.min_is_exclusive and self.min_is_derived_from_tree:
            if min_value_decimal >= 0:
                min_value_decimal_quantize = min_value_decimal.quantize(decimal_places, rounding=ROUND_UP)
            else:
                min_value_decimal_quantize = min_value_decimal.quantize(decimal_places, rounding=ROUND_DOWN)

            min_value_decimal_quantize = min_value_decimal_quantize + decimal_places

            if min_value_decimal >= min_value_decimal_quantize:
                min_value_decimal_quantize = min_value_decimal_quantize + decimal_places
                
            self.min_value = float(min_value_decimal_quantize)

        if self.max_is_derived_from_tree:
            if max_value_decimal >= 0:
                max_value_decimal_quantize = max_value_decimal.quantize(decimal_places, rounding=ROUND_DOWN)
            else:
                max_value_decimal_quantize = max_value_decimal.quantize(decimal_places, rounding=ROUND_UP)

            max_value_decimal_quantize = max_value_decimal_quantize - decimal_places
            self.max_value = float(max_value_decimal_quantize)

        # Evita  hypothesis.errors.InvalidArgument: There are no decimals with 1 places
        # between min_value=Decimal('x') and max_value=Decimal('x')
        if (self.min_is_exclusive and self.min_is_derived_from_tree) or self.max_is_derived_from_tree:
            self_decimal_places = util.create_decimal_with(self.__n_decimal_places)
            if (self.max_value - self.min_value) < float(self_decimal_places):
                self.__n_decimal_places += 1

    def update_precision_bva(self, n_decimal_places: int, float_as_decimal):
        if n_decimal_places < 6:
            n_decimal_places = n_decimal_places + 1
        else:
            n_decimal_places = 6

        decimal_places = util.create_decimal_with(n_decimal_places)

        # min_bva case
        if self.min_bva_is_derived_from_min_value:
            max_bva_n_decimal_places = util.count_decimal_places(self.max_bva_value)
            if max_bva_n_decimal_places > n_decimal_places:
                max_bva_value_decimal = util.create_decimal(self.max_bva_value)
                if max_bva_value_decimal >= 0:
                    max_bva_value_decimal_quantize = max_bva_value_decimal.quantize(decimal_places, rounding=ROUND_DOWN)
                else:
                    max_bva_value_decimal_quantize = max_bva_value_decimal.quantize(decimal_places, rounding=ROUND_UP)

                self.max_bva_value = float(max_bva_value_decimal_quantize)

        # max_bva case
        if self.max_bva_is_derived_from_max_value:
            min_bva_n_decimal_places = util.count_decimal_places(self.min_bva_value)
            if min_bva_n_decimal_places > n_decimal_places:
                # special case --> trata apenas o min_bva
                min_bva_value_decimal = util.create_decimal(self.min_bva_value)
                if min_bva_value_decimal >= 0:
                    min_bva_value_decimal_quantize = min_bva_value_decimal.quantize(decimal_places, rounding=ROUND_UP)
                else:
                    min_bva_value_decimal_quantize = min_bva_value_decimal.quantize(decimal_places, rounding=ROUND_DOWN)

                self.min_bva_value = float(min_bva_value_decimal_quantize)

        # Evita  hypothesis.errors.InvalidArgument: There are no decimals with 1 places
        # between min_value=Decimal('x') and max_value=Decimal('x')
        if self.min_bva_is_derived_from_min_value or self.max_bva_is_derived_from_max_value:
            self_decimal_places = util.create_decimal_with(self.__n_decimal_places)
            if (self.max_bva_value - self.min_bva_value) < float(self_decimal_places):
                self.__n_decimal_places += 1


class SampleValuesSpecifier:

    def __init__(self, n_features, treat_flaot_as_decimal):
        self.n_features = n_features

        self.__feature_values_list = []

        self.__create_feature_values(n_features, treat_flaot_as_decimal)

    def __getitem__(self, item):
        return self.__feature_values_list[item]

    def __len__(self):
        return len(self.__feature_values_list)

    @property
    def n_tuples(self):
        return len(self.__feature_values_list)

    def __create_feature_values(self, n_features, treat_flaot_as_decimal):

        for feature_id in range(n_features):
            min_value = sys.float_info.min
            max_value = sys.float_info.max
            feature_values = FeatureValuesSpecifier(feature_id, min_value, max_value, treat_flaot_as_decimal)
            self.__feature_values_list.append(feature_values)

    def add_feature_values(self, min_value, max_value):
        feature_values_tuple = FeatureValuesSpecifier(min_value, max_value)
        self.__feature_values_list.append(feature_values_tuple)

    def get_feature_values(self, feature_id):
        return self.__feature_values_list[feature_id]

    def merge_sample_values(self, x_train_sample_values):

        for feature_id in range(len(self.__feature_values_list)):
            feature_values = self.__feature_values_list[feature_id]
            input_data_feature_values = x_train_sample_values.get_feature_values(feature_id)

            feature_values.update_values(input_data_feature_values.min_value, input_data_feature_values.max_value)

    def update_decimal_places(self, decimal_places_in_each_feature, float_as_decimal):
        for feature_id in range(len(self.__feature_values_list)):
            n_decimal_places = decimal_places_in_each_feature[feature_id]
            feature_values = self.__feature_values_list[feature_id]
            feature_values.update_precision(n_decimal_places, float_as_decimal)

    def update_decimal_places_bva(self, decimal_places_in_each_feature, float_as_decimal):
        for feature_id in range(len(self.__feature_values_list)):
            n_decimal_places = decimal_places_in_each_feature[feature_id]
            feature_values = self.__feature_values_list[feature_id]
            feature_values.update_precision_bva(n_decimal_places, float_as_decimal)

            # Desfaz a aplicação de bva caso o intervalo seja muito restrito apos aplicar precisao em bva
            if feature_values.min_bva_value is not None and feature_values.max_bva_value is not None:
                if feature_values.min_bva_value >= feature_values.max_bva_value:
                    feature_values.min_bva_value = feature_values.min_value
                    feature_values.max_bva_value = feature_values.max_value
                    
    def set_boundary_values(self, boundary_value, random_state):
        for feature_specifier in self.__feature_values_list:
            min_bva = None
            max_bva = None

            if feature_specifier.min_is_derived_from_tree or feature_specifier.max_is_derived_from_tree:
                interval_size = math.dist([feature_specifier.min_value], [feature_specifier.max_value])
                interval_size_op = (interval_size / 100.0) * boundary_value

            if feature_specifier.min_is_derived_from_tree and feature_specifier.max_is_derived_from_tree:
                options = ['min', 'max']
                random.seed(random_state)
                choice = random.choice(options)
                if choice == 'min':
                    min_bva = feature_specifier.min_value
                    max_bva = feature_specifier.min_value + interval_size_op
                    feature_specifier.min_bva_is_derived_from_min_value = True
                else:
                    min_bva = feature_specifier.max_value - interval_size_op
                    max_bva = feature_specifier.max_value
                    feature_specifier.max_bva_is_derived_from_max_value = True
            if feature_specifier.min_is_derived_from_tree and not feature_specifier.max_is_derived_from_tree:
                min_bva = feature_specifier.min_value
                max_bva = feature_specifier.min_value + interval_size_op
                feature_specifier.min_bva_is_derived_from_min_value = True

            if not feature_specifier.min_is_derived_from_tree and feature_specifier.max_is_derived_from_tree:
                min_bva = feature_specifier.max_value - interval_size_op
                max_bva = feature_specifier.max_value
                feature_specifier.max_bva_is_derived_from_max_value = True

            feature_specifier.set_min_bva_value(min_bva)
            feature_specifier.set_max_bva_value(max_bva)

    def print(self):
        sample_values = []
        sample_values_bva = []
        for features_values in self.__feature_values_list:
            sample_values.append((features_values.min_value, features_values.max_value))
            sample_values_bva.append((features_values.min_bva_value, features_values.max_bva_value))

        print(sample_values)
        print(sample_values_bva)


class InputData:

    def __init__(self, x_train: np.ndarray, x_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray,
                       random_state):

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.random_state = random_state

        self.x_train_T = x_train.T
        self.y_train_T = y_train.T

        self.sorted_x_train_T = None
        self.__sort_x_train_T()

        self.max_decimal_places_in_each_feature = self.count_max_decimal_places_in_each_feature()

        self.feature_value_by_class = {}
        self.feature_value_by_class_by_property_id = dict()
        self.unique_classes = []
        self.__compile_each_feature_value_in_each_class()

    def get_samples(self, sample_values: SampleValuesSpecifier):
        samples = []
        samples_classification = []

        ignored_samples_ids = set()

        for feature_id in range(len(sample_values)):
            feature_values = sample_values.get_feature_values(feature_id)

            min_value = feature_values.min_value
            max_value = feature_values.max_value

            for sample_id in range(len(self.x_train)):
                if not (min_value <= self.x_train[sample_id][feature_id] <= max_value):
                    ignored_samples_ids.add(sample_id)

        for sample_id in range(len(self.x_train)):
            if not (sample_id in ignored_samples_ids):
                sample = self.x_train[sample_id]
                sample_classification = self.y_train[sample_id]
                samples.append(sample)
                samples_classification.append(sample_classification)

        return np.array(samples), np.array(samples_classification)

    def create_x_train_sample_values(self, samples, n_features):
        samples_T = samples.T

        x_train_sample_values = SampleValuesSpecifier(n_features)

        for feature_id in range(n_features):
            min_value = np.min(samples_T[feature_id])
            max_value = np.max(samples_T[feature_id])

            feature_values = x_train_sample_values.get_feature_values(feature_id)
            feature_values.update_values(min_value, max_value)

        return x_train_sample_values

    def to_json_string(self):
        json_dict = {}

        json_dict['x_train'] = self.x_train.tolist()
        json_dict['y_train'] = self.y_train.tolist()
        if self.x_test is not None:
            json_dict['x_test'] = self.x_test.tolist()
        if self.y_test is not None:
            json_dict['y_test'] = self.y_test.tolist()
        json_dict['random_state'] = self.random_state

        return json_dict

    def count_max_decimal_places_in_each_feature(self):
        features_max_decimal_places = []

        for feature_T in self.x_train_T:
            places = util.max_decimal_places_in_list(feature_T)
            features_max_decimal_places.append(places)

        return features_max_decimal_places

    def __compile_each_feature_value_in_each_class(self):

        for class_id in self.y_train:
            if not class_id in self.feature_value_by_class:
                self.feature_value_by_class[class_id] = []
                self.unique_classes.append(class_id)

        for sample_id in range(len(self.x_train)):
            sample = self.x_train[sample_id]
            class_id = self.y_train[sample_id]

            self.feature_value_by_class[class_id].append(sample.tolist())

        for class_id in self.unique_classes:
            self.feature_value_by_class[class_id] = np.array(self.feature_value_by_class[class_id]).T

    def __sort_x_train_T(self):
        self.sorted_x_train_T = copy.deepcopy(self.x_train_T)
        for feature_id in range(len(self.x_train_T)):
            self.sorted_x_train_T[feature_id] = np.sort(self.sorted_x_train_T[feature_id])

    def get_feature_values_by_class(self, feature_id, class_id, property_id):
        key = str(feature_id) + '_' + str(class_id) + '_' + str(property_id)

        if key not in self.feature_value_by_class_by_property_id:
            feature_values = self.feature_value_by_class[class_id][feature_id].tolist()

            # unique feature values
            feature_values = list(set(feature_values))

            n_feature_values = len(feature_values)

            if n_feature_values > 10:
                n_feature_values = 10

            self.feature_value_by_class_by_property_id[key] = sorted(random.sample(feature_values, n_feature_values))

        return self.feature_value_by_class_by_property_id[key]

    def get_min_feature_value(self, feature_id):
        return self.sorted_x_train_T[feature_id][0]

    def get_max_feature_value(self, feature_id):
        return self.sorted_x_train_T[feature_id][-1]

    def get_feature_type(self, feature_id):
        return type(self.sorted_x_train_T[feature_id][0])


class Data:
    def __init__(self, input_df):
        self.input_df = input_df
        self.data_df = self.__init_data_df()
        self.data = self.__init_data()
        self.feature_names = []
        self.target_df = self.__init_target_df()
        self.target = self.__init_target()
        self.target_names = None

        self.n_samples = len(self.data)
        self.n_features = self.__init_n_features()
        self.n_classes = self.__init_n_classes()
        self.classes = self.__init_classes()

        self.samples_per_class = self.__init_samples_per_class()
        self.samples_per_class_as_list = self.__init_samples_per_class_as_list()

    def __init_data_df(self):
        n_columns = len(self.input_df.columns)
        data_df = self.input_df.iloc[:, range(n_columns - 1)]
        return data_df

    def __init_data(self):
        return self.data_df.to_numpy()

    def __init_target_df(self):
        n_columns = len(self.input_df.columns)
        target_df = self.input_df.iloc[:, n_columns - 1]
        return target_df

    def __init_target(self):
        return self.target_df.to_numpy()

    def __init_n_features(self):
        sample_0 = self.data[0]

        if len(sample_0) > 0:
            return len(sample_0)
        else:
            raise SamplesNotFound("No valid samples were found in the input dataset.")

    def __init_n_classes(self):
        if self.target.any():
            unique_target = np.unique(self.target)
            return len(unique_target)
        else:
            raise SamplesNotFound("No valid samples were found in the input dataset.")

    def __init_classes(self):
        if self.target .any():
            unique_target = np.unique(self.target)
            return unique_target
        else:
            raise SamplesNotFound("No valid samples were found in the input dataset.")

    def __init_samples_per_class(self):
        samples_per_class = dict()
        counter = collections.Counter(self.target)

        for class_label, n_samples in counter.items():
            samples_per_class[class_label] = n_samples

        return samples_per_class

    def __init_samples_per_class_as_list(self):
        samples_per_class_as_list = list()

        for class_label, n_samples in self.samples_per_class.items():
            samples_per_class_as_list.append(n_samples)

        return samples_per_class_as_list

    def get_report_info(self):
        report_info = dict()

        report_info['Samples'] = self.n_samples
        report_info['Attributes'] = self.n_features + 1
        report_info['Features'] = self.n_features
        report_info['Classes'] = self.n_classes
        report_info['Samples per Class'] = str(self.samples_per_class_as_list)

        return report_info


class SamplesNotFound(Exception):
    """Exception raised when no valid samples are found in the input dataset."""

    def __init__(self, message):
        self.message = message

def load_built_in_data(data_set_name):
    data = None
    image_name = None
    class_name = None

    if data_set_name == 'iris':
        data = datasets.load_iris()
        image_name = 'decision_tree_iris.png'
        class_name = 'iris'
    elif data_set_name == 'breast_cancer':
        data = datasets.load_breast_cancer()
        image_name = 'decision_tree_breast_cancer.png'
        class_name = 'breast_cancer'
    elif data_set_name == 'wine':
        data = datasets.load_wine()
        image_name = 'decision_tree_wine.png'
        class_name = 'wine'
    elif data_set_name == 'digits':
        data = datasets.load_digits()
        image_name = 'decision_tree_digits.png'
        class_name = 'digits'

    return data, image_name, class_name
