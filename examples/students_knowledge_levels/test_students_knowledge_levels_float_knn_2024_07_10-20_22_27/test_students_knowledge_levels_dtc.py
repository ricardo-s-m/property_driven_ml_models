import json
import pytest
from hypothesis import given, strategies as st, settings, Phase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pathlib
import os.path
from joblib import load


@pytest.fixture(autouse=True, scope='class')
def _setup(request):
    model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'knn_model.joblib')
    request.cls.model = load(model_path)

    request.cls.data = dict()
    request.cls.data['n_test'] = 31
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_students_knowledge_levels_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestStudentsKnowledgeLevelsProperty:

    @given(st.sampled_from([0.0, 0.2, 0.25, 0.29, 0.32, 0.36, 0.41, 0.52, 0.55, 0.59]),
           st.sampled_from([0.0, 0.02, 0.04, 0.18, 0.28, 0.29, 0.3, 0.33, 0.52, 0.56]),
           st.sampled_from([0.0, 0.03, 0.05, 0.09, 0.2, 0.32, 0.33, 0.5, 0.52, 0.53]),
           st.floats(min_value=0.0, max_value=0.619, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.133, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.05, 0.15, 0.19, 0.24, 0.25, 0.32, 0.33, 0.41, 0.42, 0.51]),
           st.sampled_from([0.1, 0.24, 0.29, 0.31, 0.32, 0.36, 0.55, 0.6, 0.64, 0.66]),
           st.sampled_from([0.13, 0.16, 0.36, 0.38, 0.52, 0.53, 0.59, 0.72, 0.75, 0.78]),
           st.floats(min_value=0.622, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.059, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.05, 0.1, 0.12, 0.25, 0.32, 0.33, 0.36, 0.41, 0.54, 0.6]),
           st.sampled_from([0.02, 0.1, 0.17, 0.18, 0.2, 0.27, 0.33, 0.34, 0.58, 0.6]),
           st.sampled_from([0.01, 0.03, 0.3, 0.33, 0.35, 0.37, 0.5, 0.52, 0.53, 0.65]),
           st.floats(min_value=0.622, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.062, max_value=0.088, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.11, 0.12, 0.18, 0.24, 0.35, 0.39, 0.49, 0.56, 0.57, 0.58]),
           st.sampled_from([0.0, 0.02, 0.05, 0.09, 0.2, 0.24, 0.28, 0.33, 0.35, 0.4]),
           st.sampled_from([0.05, 0.06, 0.07, 0.09, 0.16, 0.18, 0.54, 0.57, 0.73, 0.8]),
           st.floats(min_value=0.622, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.091, max_value=0.133, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.05, 0.09, 0.28, 0.29, 0.36, 0.41, 0.45, 0.52, 0.55, 0.68]),
           st.sampled_from([0.02, 0.07, 0.1, 0.2, 0.28, 0.29, 0.33, 0.35, 0.51, 0.58]),
           st.sampled_from([0.0, 0.09, 0.28, 0.32, 0.34, 0.35, 0.37, 0.5, 0.52, 0.53]),
           st.floats(min_value=0.0, max_value=0.018, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.04, 0.08, 0.16, 0.18, 0.26, 0.32, 0.33, 0.52, 0.59, 0.6]),
           st.sampled_from([0.0, 0.07, 0.08, 0.15, 0.19, 0.2, 0.27, 0.52, 0.54, 0.56]),
           st.floats(min_value=0.0, max_value=0.073, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.243, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.06, 0.08, 0.11, 0.26, 0.32, 0.33, 0.37, 0.42, 0.55]),
           st.sampled_from([0.01, 0.02, 0.09, 0.14, 0.25, 0.26, 0.31, 0.51, 0.61, 0.85]),
           st.floats(min_value=0.0, max_value=0.073, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.246, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.08, 0.1, 0.2, 0.25, 0.26, 0.28, 0.29, 0.33, 0.54, 0.55]),
           st.floats(min_value=0.0, max_value=0.734, allow_nan=False),
           st.floats(min_value=0.076, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.288, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.203, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.08, 0.15, 0.26, 0.29, 0.37, 0.51, 0.52, 0.55, 0.68]),
           st.floats(min_value=0.0, max_value=0.734, allow_nan=False),
           st.floats(min_value=0.076, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.291, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.203, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.09, 0.19, 0.2, 0.28, 0.31, 0.33, 0.35, 0.55, 0.6, 0.64]),
           st.floats(min_value=0.0, max_value=0.734, allow_nan=False),
           st.floats(min_value=0.076, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.206, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.12, 0.13, 0.14, 0.15, 0.19, 0.26, 0.4, 0.41, 0.64, 0.79]),
           st.floats(min_value=0.737, max_value=0.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.076, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.268, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.05, 0.12, 0.13, 0.19, 0.28, 0.33, 0.38, 0.51, 0.6]),
           st.floats(min_value=0.737, max_value=0.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.076, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.021, max_value=0.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.271, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.1, 0.14, 0.16, 0.24, 0.25, 0.26, 0.3, 0.41, 0.58, 0.6]),
           st.sampled_from([0.1, 0.12, 0.25, 0.29, 0.33, 0.37, 0.4, 0.41, 0.65, 0.66]),
           st.sampled_from([0.08, 0.12, 0.18, 0.19, 0.31, 0.32, 0.4, 0.54, 0.59, 0.7]),
           st.floats(min_value=0.636, max_value=0.693, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.243, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.05, 0.09, 0.15, 0.28, 0.3, 0.33, 0.39, 0.5, 0.54, 0.6]),
           st.floats(min_value=0.0, max_value=0.268, allow_nan=False),
           st.sampled_from([0.08, 0.18, 0.32, 0.35, 0.5, 0.54, 0.62, 0.7, 0.75, 0.8]),
           st.floats(min_value=0.636, max_value=0.693, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.246, max_value=0.294, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.1, 0.12, 0.19, 0.28, 0.29, 0.34, 0.42, 0.48, 0.51, 0.78]),
           st.floats(min_value=0.271, max_value=0.9, exclude_min=True, allow_nan=False),
           st.sampled_from([0.06, 0.15, 0.19, 0.33, 0.39, 0.63, 0.64, 0.66, 0.79, 0.81]),
           st.floats(min_value=0.636, max_value=0.693, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.246, max_value=0.294, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.18, 0.26, 0.32, 0.38, 0.49, 0.51, 0.64, 0.65, 0.69]),
           st.sampled_from([0.03, 0.05, 0.15, 0.24, 0.25, 0.33, 0.38, 0.39, 0.41, 0.5]),
           st.sampled_from([0.02, 0.07, 0.12, 0.15, 0.18, 0.36, 0.58, 0.6, 0.63, 0.75]),
           st.floats(min_value=0.696, max_value=0.773, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.294, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.06, 0.16, 0.2, 0.24, 0.46, 0.7, 0.71, 0.76, 0.79, 0.8]),
           st.sampled_from([0.02, 0.11, 0.14, 0.15, 0.18, 0.21, 0.26, 0.32, 0.42, 0.65]),
           st.sampled_from([0.02, 0.25, 0.36, 0.38, 0.4, 0.44, 0.62, 0.64, 0.73, 0.79]),
           st.floats(min_value=0.636, max_value=0.773, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.297, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.13, 0.14, 0.25, 0.28, 0.3, 0.51, 0.54, 0.68, 0.73]),
           st.sampled_from([0.08, 0.28, 0.5, 0.51, 0.55, 0.61, 0.62, 0.66, 0.68, 0.85]),
           st.sampled_from([0.11, 0.16, 0.17, 0.33, 0.35, 0.4, 0.55, 0.57, 0.7, 0.8]),
           st.floats(min_value=0.776, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.136, max_value=0.228, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0, max_value=0.088, allow_nan=False),
           st.sampled_from([0.0, 0.26, 0.27, 0.29, 0.35, 0.42, 0.6, 0.64, 0.68, 0.72]),
           st.sampled_from([0.02, 0.05, 0.07, 0.15, 0.32, 0.34, 0.41, 0.72, 0.75, 0.79]),
           st.floats(min_value=0.776, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.091, max_value=0.99, exclude_min=True, allow_nan=False),
           st.sampled_from([0.18, 0.2, 0.28, 0.29, 0.36, 0.43, 0.52, 0.59, 0.72, 0.74]),
           st.sampled_from([0.14, 0.15, 0.25, 0.39, 0.4, 0.41, 0.54, 0.56, 0.63, 0.75]),
           st.floats(min_value=0.776, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231, max_value=0.389, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.02, 0.12, 0.15, 0.3, 0.32, 0.33, 0.37, 0.39, 0.41, 0.64]),
           st.sampled_from([0.0, 0.24, 0.29, 0.32, 0.51, 0.57, 0.61, 0.65, 0.66, 0.85]),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.124, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.15, 0.16, 0.24, 0.27, 0.4, 0.42, 0.7, 0.73, 0.75, 0.76]),
           st.sampled_from([0.02, 0.26, 0.3, 0.31, 0.32, 0.4, 0.43, 0.52, 0.67, 0.7]),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False),
           st.floats(min_value=0.127, max_value=0.829, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.2, 0.24, 0.26, 0.28, 0.29, 0.31, 0.39, 0.42, 0.5, 0.62]),
           st.sampled_from([0.02, 0.3, 0.32, 0.4, 0.51, 0.61, 0.65, 0.68, 0.75, 0.77]),
           st.floats(min_value=0.116, max_value=0.918, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.829, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.08, 0.11, 0.13, 0.22, 0.23, 0.33, 0.45, 0.61, 0.83, 0.91]),
           st.sampled_from([0.12, 0.18, 0.21, 0.31, 0.32, 0.45, 0.49, 0.51, 0.69, 0.78]),
           st.floats(min_value=0.921, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.829, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.1, 0.22, 0.41, 0.46, 0.6, 0.66, 0.75, 0.78, 0.8, 0.88]),
           st.sampled_from([0.0, 0.12, 0.2, 0.21, 0.24, 0.26, 0.33, 0.55, 0.7, 0.8]),
           st.sampled_from([0.02, 0.07, 0.12, 0.24, 0.28, 0.32, 0.43, 0.46, 0.63, 0.85]),
           st.floats(min_value=0.832, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.11, 0.15, 0.32, 0.39, 0.5, 0.6, 0.66, 0.7, 0.78, 0.8]),
           st.sampled_from([0.09, 0.13, 0.21, 0.29, 0.31, 0.39, 0.44, 0.61, 0.67, 0.68]),
           st.floats(min_value=0.0, max_value=0.404, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.083, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.08, 0.09, 0.1, 0.15, 0.26, 0.44, 0.47, 0.66, 0.77, 0.8]),
           st.sampled_from([0.22, 0.28, 0.34, 0.46, 0.49, 0.61, 0.68, 0.7, 0.82, 0.88]),
           st.floats(min_value=0.407, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.083, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.12, 0.18, 0.24, 0.3, 0.43, 0.44, 0.75, 0.83, 0.85, 0.9]),
           st.floats(min_value=0.0, max_value=0.834, allow_nan=False),
           st.sampled_from([0.04, 0.07, 0.15, 0.25, 0.31, 0.38, 0.46, 0.68, 0.91, 0.95]),
           st.floats(min_value=0.086, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 0.08, 0.3, 0.45, 0.48, 0.49, 0.54, 0.78, 0.85, 0.99]),
           st.floats(min_value=0.837, max_value=0.864, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.389, allow_nan=False),
           st.floats(min_value=0.086, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.15, 0.17, 0.19, 0.26, 0.39, 0.4, 0.45, 0.62, 0.75, 0.76]),
           st.floats(min_value=0.837, max_value=0.864, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.392, max_value=0.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.086, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.09, 0.15, 0.21, 0.32, 0.38, 0.4, 0.46, 0.66, 0.8, 0.83]),
           st.floats(min_value=0.867, max_value=0.9, exclude_min=True, allow_nan=False),
           st.sampled_from([0.02, 0.1, 0.12, 0.32, 0.44, 0.46, 0.65, 0.66, 0.67, 0.89]),
           st.floats(min_value=0.086, max_value=0.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.677, max_value=0.99, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
