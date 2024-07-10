import autopep8
import file_io
from scikit_strategy import KFoldStrategy, GenericClassifierStrategy


class GenericTestClassGenerator:

    def __init__(self, class_name, modulo_name, n_features, classification_paths, input_data,
                 classifier_strategy: GenericClassifierStrategy, hypothesis_strategy,
                 is_for_experimentation, export_directory, treat_flaot_as_decimal,
                 export_target_model, max_examples, criteria):
        self.class_name = class_name
        self.modulo_name = modulo_name
        self.classification_paths = classification_paths
        self.n_properties = len(classification_paths)
        self.n_features = n_features
        self.input_data = input_data
        self.classifier_strategy = classifier_strategy
        self.hypothesis_strategy = hypothesis_strategy
        self.property_id = 0
        self.feature_id = 0
        self.is_for_experimentation = is_for_experimentation
        self.export_directory = export_directory
        self.treat_flaot_as_decimal = treat_flaot_as_decimal
        self.k_fold_strategy = KFoldStrategy()
        self.export_target_model = export_target_model
        self.max_examples = max_examples
        self.criteria = criteria.lower()

    def _generate_imports(self):
        return ''

    def _generate_class_setup(self):
        return ''

    def _generate_class_name(self):
        return ''

    def _generate_object_setup(self):
        return ''

    def _generate_hypothesis_decorators(self):
        return ''

    def _generate_test_implementation(self):
        return ''

    def generate_test_class(self):
        text_class = self._generate_imports()
        text_class += self.__autopep8_format(self._generate_class_setup(), 2)
        text_class += self._generate_class_name()
        text_class += self._generate_object_setup()

        for n_property in range(self.n_properties):
            text_class += self._generate_hypothesis_decorators()
            text_class += self._generate_test_implementation()
            self.property_id += 1

        self.__create_test_class_file(text_class)
        self.__create_json_data_file()

        return text_class

    def __create_test_class_file(self, text_class):

        if self.criteria == 'dtc':
            modulo_name = self.modulo_name + '_dtc'
        else:
            modulo_name = self.modulo_name + '_bva'

        file_io.create_python_file(text_class, modulo_name, self.export_directory)

    def __input_data_to_json(self, input_data):
        return self.input_data.__dict__

    def __create_json_data_file(self):

        file_io.create_json_file(self.input_data.to_json_string(), self.modulo_name, self.export_directory)

    def __autopep8_format(self, text, aggressive_level):
        text_pep8 = autopep8.fix_code(text,
                                      options={'diff': True,
                                               'recursive': True,
                                               'ignore_local_config': True,
                                               'aggressive': aggressive_level,
                                               'max_line_length': 120,
                                               'exit_code': True},
                                      apply_config=False)

        return text_pep8


class PytestClassGenerator(GenericTestClassGenerator):
    def _generate_imports(self):
        text_imports = 'import json\n'
        text_imports += 'import pytest\n'
        text_imports += 'from hypothesis import given, strategies as st, settings, Phase\n'
        text_imports += self.classifier_strategy.imports()
        text_imports += self.k_fold_strategy.imports(True)
        text_imports += 'import pathlib\n'
        text_imports += 'import os.path\n'

        if self.export_target_model:
            text_imports += 'from joblib import load\n'

        text_imports += '\n'

        return text_imports

    def _generate_class_setup(self):
        text_class_setup = '\n'
        text_class_setup += '@pytest.fixture(autouse=True, scope=\'class\')\n'
        text_class_setup += 'def _setup(request):\n'

        if self.export_target_model:

            target_model_name = file_io.get_target_model_name(self.classifier_strategy.name())
            text_class_setup += '    model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), \'' + target_model_name + '\')\n'
            text_class_setup += '    request.cls.model = load(model_path)\n'
        else:
            train_data_path = file_io.get_train_data_file_path(self.modulo_name, self.export_directory)
            experiment_data_path = file_io.get_experiment_data_file_path(self.modulo_name, self.export_directory)

            text_class_setup += '    with open(\'' + train_data_path + '\') as json_file:\n'
            text_class_setup += '        request.cls.input_data = json.load(json_file)\n'
            text_class_setup += '        request.cls.x_train = request.cls.input_data[\'x_train\']\n'
            text_class_setup += '        request.cls.y_train = request.cls.input_data[\'y_train\']\n'
            text_class_setup += self.classifier_strategy.classifier()
            text_class_setup += '        request.cls.model = request.cls.classifier.fit(request.cls.x_train, ' \
                                'request.cls.y_train)\n'

            text_class_setup += self.classifier_strategy.image(self.export_directory)

        if self.is_for_experimentation:
            
            text_class_setup += '\n'
            text_class_setup += '    request.cls.data = dict()\n'
            text_class_setup += '    request.cls.data[\'n_test\'] = ' + str(self.n_properties) + '\n'
            text_class_setup += '    request.cls.data[\'n_samples_per_test\'] = ' + str(self.max_examples) + '\n'
            text_class_setup += '    request.cls.data[\'tests\'] = dict()\n\n'

            text_class_setup += '    for i in range(request.cls.data[\'n_test\']):\n'
            text_class_setup += '        teste_id = \'test_\' + str(i + 1)\n'
            text_class_setup += '        request.cls.data[\'tests\'][teste_id] = {\'n_samples\': 0, \'samples\': [], \'y_expected\': [], \'y_predicted\': []}\n'

            text_class_setup += '\n'

            file_name = self.modulo_name + '_' + self.criteria + '_'

            experiment_data_file_name = file_io.get_experiment_data_file_name(file_name)
            text_class_setup += '    experiment_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), \'' + experiment_data_file_name + '\')\n'
            text_class_setup += '    yield experiment_data_path\n'
            text_class_setup += '    with open(experiment_data_path, mode=\'w\') as json_file:\n'
            text_class_setup += '        json.dump(request.cls.data, json_file)\n'

        text_class_setup += '        \n'

        return text_class_setup

    def _generate_class_name(self):
        text_class_name = '\n\n'
        text_class_name += 'class ' + self.class_name + 'Property:\n'

        return text_class_name

    def _generate_hypothesis_decorators(self):
        
        classification_path = self.classification_paths[self.property_id]
        text_decorators = self.hypothesis_strategy.get_method_decorators(classification_path, self.property_id)

        max_examples = 'max_examples=' + str(self.max_examples)

        text_decorators += '\n'
        text_decorators += '    @settings(phases=[Phase.generate], ' + max_examples + ')'

        return text_decorators

    def _generate_test_implementation(self):
        text_test_impl = self.hypothesis_strategy.get_method_signature(self.property_id, self.n_features)
        text_test_impl += self.hypothesis_strategy.get_test_data(self.n_features, self.treat_flaot_as_decimal)

        text_test_impl += self.hypothesis_strategy.get_test_assertions(
            self.classification_paths[self.property_id].class_type, 10, self.property_id, self.is_for_experimentation)

        return text_test_impl


class UnittestClassGenerator(GenericTestClassGenerator):
    pass
