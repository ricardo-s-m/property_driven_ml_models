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
    request.cls.data['n_test'] = 9
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'test_iris_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestIrisProperty:

    @given(st.sampled_from([4.5, 4.7, 4.8, 4.9, 5.0, 5.1, 5.3, 5.5, 5.7, 5.8]),
           st.sampled_from([2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.7, 4.0, 4.1, 4.2]),
           st.floats(min_value=2.15, max_value=2.43, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.9, 5.0, 5.2, 5.5, 5.6, 6.3, 6.4, 6.5, 6.7, 6.9]),
           st.sampled_from([2.0, 2.2, 2.3, 2.4, 2.8, 2.9, 3.0, 3.1, 3.3, 3.4]),
           st.floats(min_value=2.46, max_value=2.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.34, max_value=1.64, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([5.8, 6.0, 6.1, 6.2, 6.4, 6.5, 6.9, 7.4, 7.6, 7.7]),
           st.sampled_from([2.2, 2.5, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]),
           st.floats(min_value=2.46, max_value=2.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.67, max_value=1.68, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([5.7, 5.9, 6.0, 6.1, 6.2, 6.3, 6.8, 7.1, 7.3, 7.7]),
           st.sampled_from([2.2, 2.6, 2.7, 2.8, 3.0, 3.1, 3.3, 3.4, 3.6, 3.8]),
           st.floats(min_value=4.97, max_value=5.35, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=1.54, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.9, 5.0, 5.2, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 6.4]),
           st.sampled_from([2.0, 2.2, 2.4, 2.7, 2.8, 2.9, 3.0, 3.1, 3.3, 3.4]),
           st.floats(min_value=4.97, max_value=5.06, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.57, max_value=1.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([5.6, 5.9, 6.0, 6.1, 6.2, 6.3, 6.7, 7.2, 7.7, 7.9]),
           st.sampled_from([2.2, 2.5, 2.6, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8]),
           st.floats(min_value=5.47, max_value=5.75, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.57, max_value=1.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.62, max_value=5.94, allow_nan=False),
           st.sampled_from([2.3, 2.5, 2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]),
           st.floats(min_value=2.46, max_value=2.93, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.76, max_value=1.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.97, max_value=6.35, exclude_min=True, allow_nan=False),
           st.sampled_from([2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.6, 3.8]),
           st.floats(min_value=2.46, max_value=2.93, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.76, max_value=1.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.9, 5.6, 6.2, 6.4, 6.7, 6.8, 7.1, 7.4, 7.6, 7.7]),
           st.sampled_from([2.5, 2.6, 2.7, 2.8, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8]),
           st.floats(min_value=4.87, max_value=5.27, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.76, max_value=1.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
