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
    request.cls.data['n_test'] = 12
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'test_wine_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestWineProperty:

    @given(st.sampled_from([12.51, 12.7, 12.87, 13.23, 13.32, 13.48, 13.52, 13.69, 13.73, 13.88]),
           st.sampled_from([1.67, 2.46, 2.59, 2.67, 2.99, 3.45, 4.12, 4.28, 4.61, 5.04]),
           st.sampled_from([2.19, 2.25, 2.35, 2.37, 2.48, 2.54, 2.61, 2.69, 2.74, 2.75]),
           st.sampled_from([17.5, 18.0, 18.5, 19.5, 22.0, 22.5, 23.0, 23.5, 24.5, 25.5]),
           st.sampled_from([85.0, 86.0, 88.0, 94.0, 97.0, 101.0, 104.0, 113.0, 120.0, 123.0]),
           st.sampled_from([1.38, 1.48, 1.51, 1.55, 1.7, 1.83, 2.0, 2.05, 2.32, 2.8]),
           st.floats(min_value=0.34, max_value=1.579, allow_nan=False),
           st.sampled_from([0.21, 0.22, 0.26, 0.37, 0.4, 0.43, 0.44, 0.47, 0.58, 0.63]),
           st.sampled_from([0.75, 0.86, 1.02, 1.04, 1.06, 1.11, 1.25, 1.26, 1.35, 1.54]),
           st.sampled_from([4.9, 5.0, 5.4, 5.7, 6.62, 7.5, 7.7, 9.4, 10.26, 13.0]),
           st.floats(min_value=0.48, max_value=0.9349, allow_nan=False),
           st.floats(min_value=1.27, max_value=2.113, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.62, 11.79, 12.25, 12.34, 12.51, 12.52, 12.6, 12.99, 13.49, 13.86]),
           st.sampled_from([0.74, 1.17, 1.19, 1.29, 1.41, 1.63, 1.83, 2.45, 2.83, 4.3]),
           st.sampled_from([1.95, 2.02, 2.1, 2.13, 2.19, 2.2, 2.3, 2.36, 2.51, 2.53]),
           st.sampled_from([15.0, 17.0, 19.0, 19.6, 20.7, 21.5, 24.0, 25.0, 26.0, 26.5]),
           st.sampled_from([81.0, 92.0, 94.0, 99.0, 101.0, 103.0, 107.0, 110.0, 134.0, 151.0]),
           st.sampled_from([1.38, 1.6, 1.61, 1.68, 1.75, 2.48, 2.95, 2.98, 3.38, 3.52]),
           st.floats(min_value=1.582, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.13, 0.24, 0.28, 0.29, 0.3, 0.32, 0.4, 0.45, 0.55, 0.6]),
           st.sampled_from([0.42, 1.38, 1.44, 1.46, 1.63, 1.65, 1.71, 1.99, 2.5, 3.28]),
           st.sampled_from([2.4, 2.7, 2.76, 2.8, 2.94, 2.95, 3.27, 3.74, 4.5, 5.3]),
           st.floats(min_value=0.48, max_value=0.9349, allow_nan=False),
           st.floats(min_value=1.27, max_value=2.113, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=11.03, max_value=13.513, allow_nan=False),
           st.sampled_from([0.98, 1.07, 1.25, 1.29, 1.41, 1.45, 1.81, 2.89, 3.43, 5.8]),
           st.sampled_from([1.88, 1.95, 1.99, 2.12, 2.21, 2.23, 2.29, 2.36, 2.42, 2.62]),
           st.sampled_from([20.4, 20.5, 21.0, 22.5, 23.6, 24.0, 25.0, 26.0, 26.5, 28.5]),
           st.sampled_from([80.0, 84.0, 94.0, 96.0, 103.0, 110.0, 112.0, 119.0, 134.0, 139.0]),
           st.sampled_from([1.6, 1.65, 1.92, 1.98, 2.02, 2.1, 2.22, 2.48, 2.62, 3.5]),
           st.sampled_from([1.25, 1.58, 1.6, 1.79, 2.0, 2.03, 2.04, 2.13, 2.27, 2.5]),
           st.sampled_from([0.21, 0.24, 0.3, 0.32, 0.34, 0.35, 0.37, 0.39, 0.4, 0.58]),
           st.sampled_from([0.83, 0.95, 1.31, 1.35, 1.53, 1.56, 1.83, 1.87, 1.95, 2.08]),
           st.sampled_from([1.28, 2.2, 2.5, 2.9, 3.0, 3.21, 3.74, 3.9, 4.6, 4.68]),
           st.floats(min_value=0.9352, max_value=1.71, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.27, max_value=2.113, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.516, max_value=14.83, exclude_min=True, allow_nan=False),
           st.sampled_from([1.24, 2.31, 2.51, 3.17, 3.83, 3.88, 4.12, 4.61, 4.72, 4.95]),
           st.sampled_from([2.15, 2.19, 2.25, 2.28, 2.32, 2.37, 2.54, 2.62, 2.64, 2.74]),
           st.sampled_from([17.5, 18.0, 18.5, 19.0, 20.0, 22.0, 24.0, 24.5, 25.0, 25.5]),
           st.sampled_from([89.0, 91.0, 94.0, 102.0, 104.0, 111.0, 112.0, 113.0, 120.0, 123.0]),
           st.sampled_from([1.15, 1.25, 1.41, 1.48, 1.54, 1.59, 1.68, 1.7, 2.05, 2.6]),
           st.sampled_from([0.49, 0.63, 0.65, 0.66, 0.69, 0.8, 1.22, 1.25, 1.31, 1.39]),
           st.sampled_from([0.17, 0.27, 0.39, 0.43, 0.47, 0.5, 0.52, 0.56, 0.6, 0.63]),
           st.sampled_from([0.68, 0.75, 0.81, 1.03, 1.04, 1.06, 1.24, 1.25, 1.26, 1.46]),
           st.sampled_from([3.85, 4.0, 4.9, 5.4, 5.88, 6.62, 7.1, 7.3, 9.01, 10.2]),
           st.floats(min_value=0.9352, max_value=1.71, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.27, max_value=2.113, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([12.36, 12.6, 12.7, 12.81, 12.93, 13.16, 13.36, 13.69, 13.73, 14.16]),
           st.sampled_from([1.35, 1.9, 2.39, 2.51, 2.76, 2.81, 3.7, 4.12, 4.61, 5.19]),
           st.sampled_from([2.15, 2.2, 2.23, 2.37, 2.4, 2.6, 2.61, 2.62, 2.7, 2.72]),
           st.sampled_from([17.5, 18.5, 19.0, 19.5, 20.0, 21.5, 22.0, 23.0, 25.0, 25.5]),
           st.sampled_from([80.0, 85.0, 86.0, 89.0, 90.0, 93.0, 95.0, 98.0, 107.0, 116.0]),
           st.sampled_from([0.98, 1.15, 1.35, 1.4, 1.48, 1.51, 1.7, 1.74, 2.0, 2.3]),
           st.floats(min_value=0.34, max_value=0.794, allow_nan=False),
           st.sampled_from([0.17, 0.21, 0.26, 0.39, 0.41, 0.52, 0.53, 0.56, 0.6, 0.61]),
           st.sampled_from([0.55, 0.68, 0.86, 0.88, 1.11, 1.24, 1.3, 1.41, 1.54, 1.55]),
           st.sampled_from([4.1, 4.35, 5.58, 5.6, 5.88, 7.3, 8.21, 8.6, 10.8, 11.75]),
           st.sampled_from([0.48, 0.58, 0.6, 0.65, 0.66, 0.72, 0.73, 0.81, 0.89, 0.96]),
           st.floats(min_value=2.116, max_value=4.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=11.03, max_value=13.173, allow_nan=False),
           st.sampled_from([0.99, 1.13, 1.33, 1.51, 2.4, 2.68, 2.89, 3.74, 3.86, 3.87]),
           st.sampled_from([1.82, 1.92, 1.94, 1.99, 2.16, 2.19, 2.23, 2.24, 2.26, 2.51]),
           st.sampled_from([10.6, 16.0, 17.0, 17.5, 18.1, 20.5, 20.7, 22.8, 24.5, 26.5]),
           st.sampled_from([81.0, 84.0, 85.0, 87.0, 96.0, 97.0, 101.0, 119.0, 136.0, 151.0]),
           st.sampled_from([1.45, 2.2, 2.22, 2.55, 2.74, 2.83, 2.86, 2.98, 3.5, 3.52]),
           st.floats(min_value=0.797, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.14, 0.21, 0.25, 0.28, 0.34, 0.42, 0.47, 0.5, 0.53, 0.61]),
           st.sampled_from([0.95, 1.31, 1.34, 1.38, 1.43, 1.76, 1.9, 1.95, 1.99, 2.28]),
           st.sampled_from([1.9, 2.4, 2.5, 2.7, 2.76, 3.05, 3.3, 3.35, 3.94, 4.68]),
           st.sampled_from([0.69, 0.7, 0.75, 0.79, 0.89, 0.94, 0.98, 1.0, 1.04, 1.38]),
           st.floats(min_value=2.116, max_value=4.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.176, max_value=13.363, exclude_min=True, allow_nan=False),
           st.sampled_from([1.59, 1.6, 1.63, 1.66, 1.72, 1.76, 1.83, 2.16, 3.8, 3.98]),
           st.sampled_from([2.1, 2.14, 2.25, 2.29, 2.3, 2.31, 2.42, 2.43, 2.45, 2.51]),
           st.sampled_from([15.6, 16.0, 16.1, 16.2, 16.5, 17.5, 17.8, 19.0, 20.0, 20.5]),
           st.sampled_from([92.0, 93.0, 94.0, 102.0, 104.0, 111.0, 115.0, 117.0, 128.0, 132.0]),
           st.sampled_from([2.4, 2.61, 2.7, 2.85, 2.86, 2.95, 2.96, 3.0, 3.25, 3.27]),
           st.floats(min_value=0.797, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.17, 0.19, 0.21, 0.22, 0.25, 0.27, 0.32, 0.34, 0.37, 0.42]),
           st.sampled_from([1.37, 1.46, 1.48, 1.57, 1.68, 1.69, 1.76, 1.81, 1.87, 2.45]),
           st.sampled_from([3.52, 3.84, 4.25, 4.6, 5.05, 5.4, 5.6, 5.64, 6.2, 7.8]),
           st.sampled_from([0.82, 0.94, 0.96, 1.02, 1.05, 1.08, 1.09, 1.12, 1.17, 1.18]),
           st.floats(min_value=2.116, max_value=4.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.366, max_value=14.83, exclude_min=True, allow_nan=False),
           st.sampled_from([0.92, 1.29, 1.34, 1.36, 1.52, 1.66, 1.81, 2.06, 2.68, 4.43]),
           st.sampled_from([1.36, 1.88, 2.13, 2.16, 2.17, 2.27, 2.32, 2.6, 2.7, 2.73]),
           st.sampled_from([10.6, 18.1, 19.0, 19.5, 20.5, 21.0, 21.5, 23.0, 24.0, 25.0]),
           st.sampled_from([70.0, 86.0, 88.0, 97.0, 99.0, 101.0, 103.0, 112.0, 119.0, 162.0]),
           st.sampled_from([1.38, 1.89, 2.05, 2.13, 2.2, 2.45, 2.6, 2.9, 3.38, 3.52]),
           st.floats(min_value=0.797, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.25, 0.27, 0.37, 0.42, 0.43, 0.45, 0.47, 0.48, 0.58, 0.63]),
           st.sampled_from([0.62, 1.03, 1.71, 1.77, 1.87, 1.95, 1.96, 2.35, 2.91, 3.58]),
           st.sampled_from([2.4, 2.5, 2.57, 2.6, 2.95, 3.74, 3.8, 3.9, 4.45, 6.0]),
           st.sampled_from([0.73, 0.92, 1.0, 1.02, 1.04, 1.12, 1.15, 1.28, 1.42, 1.71]),
           st.floats(min_value=2.116, max_value=4.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=278.0, max_value=754.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([12.25, 12.85, 12.87, 12.88, 12.93, 12.96, 13.08, 13.27, 13.32, 13.71]),
           st.sampled_from([1.24, 2.39, 2.76, 3.03, 3.17, 3.37, 4.36, 4.72, 5.04, 5.65]),
           st.sampled_from([2.23, 2.36, 2.4, 2.58, 2.6, 2.62, 2.69, 2.72, 2.74, 2.86]),
           st.sampled_from([18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 23.5, 24.0, 24.5, 25.0]),
           st.sampled_from([89.0, 90.0, 93.0, 98.0, 101.0, 103.0, 104.0, 107.0, 112.0, 122.0]),
           st.sampled_from([1.38, 1.39, 1.4, 1.51, 1.54, 1.65, 1.79, 1.9, 2.05, 2.6]),
           st.floats(min_value=0.34, max_value=2.164, allow_nan=False),
           st.sampled_from([0.17, 0.22, 0.24, 0.34, 0.4, 0.43, 0.44, 0.45, 0.5, 0.56]),
           st.sampled_from([0.68, 0.73, 0.75, 0.97, 1.02, 1.04, 1.4, 1.54, 1.87, 2.7]),
           st.sampled_from([3.85, 5.45, 7.65, 8.6, 9.2, 9.3, 9.899999, 10.26, 10.8, 13.0]),
           st.floats(min_value=0.48, max_value=0.8029, allow_nan=False),
           st.sampled_from([1.27, 1.29, 1.33, 1.48, 1.55, 1.64, 1.8, 1.82, 1.96, 2.31]),
           st.floats(min_value=755.01, max_value=1680.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.45, 11.65, 11.66, 11.79, 12.0, 12.08, 12.42, 12.67, 12.69, 12.77]),
           st.sampled_from([1.01, 1.33, 1.47, 1.51, 2.68, 2.83, 3.17, 3.74, 4.43, 5.8]),
           st.sampled_from([1.7, 1.88, 1.99, 2.0, 2.24, 2.32, 2.46, 2.67, 2.92, 3.23]),
           st.sampled_from([18.0, 18.8, 19.0, 19.6, 20.4, 20.5, 20.8, 22.0, 22.8, 30.0]),
           st.sampled_from([81.0, 82.0, 94.0, 100.0, 104.0, 110.0, 112.0, 134.0, 136.0, 162.0]),
           st.sampled_from([1.75, 1.78, 1.95, 1.98, 2.11, 2.13, 2.2, 2.42, 2.48, 2.62]),
           st.floats(min_value=0.34, max_value=2.164, allow_nan=False),
           st.sampled_from([0.14, 0.21, 0.25, 0.27, 0.32, 0.35, 0.39, 0.45, 0.53, 0.63]),
           st.sampled_from([0.73, 1.22, 1.34, 1.4, 1.56, 1.62, 1.87, 1.9, 2.5, 2.81]),
           st.sampled_from([1.9, 2.15, 3.3, 3.38, 3.74, 3.94, 4.8, 5.3, 5.75, 6.0]),
           st.floats(min_value=0.8032, max_value=1.71, exclude_min=True, allow_nan=False),
           st.sampled_from([1.67, 2.12, 2.44, 2.63, 2.77, 2.78, 2.81, 3.12, 3.17, 3.64]),
           st.floats(min_value=755.01, max_value=1680.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.07, 13.16, 13.41, 13.48, 13.58, 13.72, 13.73, 13.88, 14.12, 14.75]),
           st.sampled_from([1.53, 1.59, 1.63, 1.78, 1.92, 1.95, 1.97, 2.16, 2.36, 3.8]),
           st.sampled_from([2.12, 2.17, 2.21, 2.28, 2.31, 2.4, 2.42, 2.51, 2.52, 2.87]),
           st.sampled_from([14.0, 14.6, 15.6, 16.7, 17.2, 17.5, 17.6, 18.8, 19.4, 19.5]),
           st.floats(min_value=70.0, max_value=135.49, allow_nan=False),
           st.sampled_from([2.2, 2.4, 2.42, 2.48, 2.53, 2.63, 2.85, 2.95, 3.0, 3.1]),
           st.floats(min_value=2.167, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.17, 0.19, 0.2, 0.22, 0.25, 0.26, 0.27, 0.28, 0.39, 0.43]),
           st.sampled_from([1.37, 1.46, 1.48, 1.57, 1.69, 1.81, 1.92, 2.29, 2.81, 2.91]),
           st.sampled_from([3.7, 4.2, 5.0, 5.4, 5.43, 5.75, 6.0, 6.25, 6.9, 7.8]),
           st.sampled_from([0.87, 0.96, 0.98, 1.06, 1.08, 1.11, 1.15, 1.17, 1.18, 1.19]),
           st.sampled_from([2.57, 2.78, 2.85, 2.88, 2.91, 3.31, 3.35, 3.38, 3.55, 3.92]),
           st.floats(min_value=755.01, max_value=1680.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.65, 12.04, 12.16, 12.21, 12.29, 12.67, 12.7, 12.77, 12.99, 13.05]),
           st.sampled_from([0.92, 0.98, 1.36, 1.41, 1.63, 2.05, 2.13, 2.43, 3.17, 3.43]),
           st.sampled_from([1.7, 1.98, 2.16, 2.29, 2.3, 2.32, 2.36, 2.5, 2.58, 2.74]),
           st.sampled_from([10.6, 15.0, 16.0, 17.0, 19.0, 20.5, 21.0, 22.8, 25.0, 26.0]),
           st.floats(min_value=135.51, max_value=162.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.45, 1.65, 1.68, 1.75, 1.78, 1.95, 2.53, 2.98, 3.02, 3.38]),
           st.floats(min_value=2.167, max_value=5.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.17, 0.19, 0.25, 0.35, 0.37, 0.4, 0.43, 0.52, 0.6, 0.61]),
           st.sampled_from([0.42, 0.73, 1.04, 1.34, 1.38, 1.43, 1.77, 1.9, 1.95, 2.91]),
           st.sampled_from([1.74, 2.12, 2.2, 2.4, 2.45, 2.6, 2.8, 3.0, 3.17, 3.21]),
           st.sampled_from([0.69, 0.73, 0.89, 0.906, 0.95, 0.99, 1.04, 1.08, 1.33, 1.42]),
           st.sampled_from([2.06, 2.27, 2.42, 2.48, 2.72, 2.78, 2.87, 3.14, 3.19, 3.64]),
           st.floats(min_value=755.01, max_value=1680.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
