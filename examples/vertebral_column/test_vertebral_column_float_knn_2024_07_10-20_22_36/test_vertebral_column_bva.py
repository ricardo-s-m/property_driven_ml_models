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
    request.cls.data['n_test'] = 45
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_vertebral_column_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestVertebralColumnProperty:

    @given(st.sampled_from([26.15, 31.48, 35.7, 38.66, 40.56, 41.17, 41.73, 44.32, 46.39, 48.11]),
           st.floats(min_value=11.805, max_value=16.393, allow_nan=False),
           st.floats(min_value=17.131, max_value=17.913, allow_nan=False),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=125.995, max_value=139.973, allow_nan=False),
           st.floats(min_value=-1.94, max_value=0.339, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.65, 35.88, 37.14, 41.65, 44.43, 46.37, 48.8, 49.83, 51.62, 53.91]),
           st.floats(min_value=11.805, max_value=16.393, allow_nan=False),
           st.floats(min_value=17.916, max_value=39.48, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=125.995, max_value=139.973, allow_nan=False),
           st.floats(min_value=-1.94, max_value=0.339, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([26.15, 39.06, 41.35, 41.73, 53.43, 53.85, 54.92, 56.03, 68.83, 74.43]),
           st.floats(min_value=16.396, max_value=23.002, exclude_min=True, allow_nan=False),
           st.sampled_from([30.12, 30.71, 31.33, 32.14, 33.47, 35.0, 36.18, 40.26, 47.5, 50.09]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=125.995, max_value=139.973, allow_nan=False),
           st.floats(min_value=-2.94, max_value=-0.911, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([41.65, 43.44, 44.36, 51.31, 51.62, 67.29, 67.54, 72.96, 74.98, 89.83]),
           st.floats(min_value=16.396, max_value=23.002, exclude_min=True, allow_nan=False),
           st.sampled_from([26.93, 31.47, 32.24, 36.67, 43.2, 43.46, 47.0, 50.45, 55.5, 66.0]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=114.543, max_value=125.658, allow_nan=False),
           st.floats(min_value=-0.908, max_value=-0.659, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([31.28, 39.06, 41.35, 43.58, 46.44, 47.66, 50.21, 63.83, 66.29, 66.88]),
           st.floats(min_value=16.396, max_value=23.002, exclude_min=True, allow_nan=False),
           st.sampled_from([15.5, 15.59, 26.79, 30.12, 31.0, 34.0, 35.0, 37.17, 42.69, 54.0]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=125.661, max_value=128.523, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.908, max_value=-0.659, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([36.13, 40.25, 43.58, 44.32, 46.44, 48.33, 49.71, 66.29, 69.3, 74.43]),
           st.sampled_from([8.4, 11.7, 13.07, 13.53, 14.18, 14.93, 15.35, 16.58, 19.23, 19.96]),
           st.sampled_from([26.79, 28.07, 30.12, 33.1, 35.87, 36.0, 36.18, 37.17, 42.69, 46.56]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=106.268, max_value=115.314, allow_nan=False),
           st.floats(min_value=0.342, max_value=1.658, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.65, 34.76, 38.13, 47.32, 51.08, 51.62, 53.91, 59.73, 62.14, 63.93]),
           st.sampled_from([2.06, 5.54, 6.46, 6.82, 9.15, 9.98, 12.94, 17.45, 18.97, 26.08]),
           st.sampled_from([32.39, 33.63, 39.71, 41.58, 43.0, 43.46, 46.9, 55.5, 62.58, 90.56]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=115.317, max_value=115.671, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.342, max_value=1.658, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([30.15, 31.23, 32.09, 36.13, 36.69, 40.56, 43.35, 44.94, 47.66, 66.29]),
           st.sampled_from([7.47, 8.4, 11.08, 11.92, 13.07, 20.46, 22.23, 22.76, 24.89, 28.85]),
           st.sampled_from([29.0, 30.12, 31.33, 32.78, 33.47, 35.0, 36.0, 47.0, 54.55, 62.28]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=117.092, max_value=121.668, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.342, max_value=1.658, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([30.74, 39.09, 40.75, 44.43, 51.33, 53.68, 56.45, 63.03, 63.96, 67.29]),
           st.sampled_from([-2.97, 13.35, 13.43, 13.62, 16.48, 16.72, 18.76, 18.9, 19.97, 21.49]),
           st.sampled_from([19.07, 24.0, 30.9, 35.0, 37.97, 42.84, 43.9, 46.17, 55.57, 62.58]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.floats(min_value=139.976, max_value=144.594, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.327, max_value=6.923, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([38.13, 38.51, 40.68, 42.52, 46.64, 47.32, 54.92, 63.79, 65.61, 67.29]),
           st.sampled_from([-1.33, -0.32, 5.59, 8.88, 9.41, 10.19, 16.48, 16.96, 21.79, 26.08]),
           st.sampled_from([33.77, 42.58, 42.7, 43.58, 43.9, 52.89, 53.0, 58.25, 64.0, 90.56]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.sampled_from([100.5, 103.58, 114.51, 115.39, 118.15, 121.04, 121.78, 129.22, 130.35, 142.41]),
           st.floats(min_value=6.926, max_value=7.238, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([30.15, 31.28, 41.35, 41.73, 47.66, 48.11, 50.21, 55.29, 63.03, 66.29]),
           st.sampled_from([7.47, 8.4, 11.7, 15.4, 17.72, 19.96, 20.36, 24.19, 29.76, 41.56]),
           st.sampled_from([15.59, 20.7, 25.02, 28.32, 34.0, 35.33, 39.61, 42.2, 47.5, 62.28]),
           st.floats(min_value=25.181, max_value=28.133, allow_nan=False),
           st.sampled_from([98.67, 106.94, 109.27, 112.31, 115.58, 116.25, 119.33, 120.06, 125.0, 132.26]),
           st.floats(min_value=8.491, max_value=10.008, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([30.15, 31.48, 35.49, 38.66, 41.77, 48.33, 48.92, 50.82, 50.91, 68.83]),
           st.floats(min_value=19.081, max_value=25.488, allow_nan=False),
           st.sampled_from([20.7, 24.28, 25.02, 28.32, 30.12, 35.56, 40.26, 41.95, 42.69, 47.0]),
           st.floats(min_value=28.136, max_value=31.837, exclude_min=True, allow_nan=False),
           st.floats(min_value=104.355, max_value=112.923, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([33.04, 37.73, 38.05, 39.36, 41.65, 43.44, 46.24, 46.37, 51.08, 66.51]),
           st.floats(min_value=25.491, max_value=30.278, exclude_min=True, allow_nan=False),
           st.sampled_from([19.07, 29.36, 29.5, 36.03, 37.0, 37.97, 43.58, 51.87, 66.0, 90.56]),
           st.floats(min_value=28.136, max_value=31.837, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.667, max_value=107.063, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([41.77, 43.79, 44.55, 44.94, 45.37, 48.33, 53.57, 54.12, 56.03, 74.43]),
           st.floats(min_value=25.491, max_value=30.278, exclude_min=True, allow_nan=False),
           st.sampled_from([20.03, 25.02, 25.12, 28.07, 31.0, 36.68, 44.31, 47.0, 50.09, 54.0]),
           st.floats(min_value=28.136, max_value=31.837, exclude_min=True, allow_nan=False),
           st.floats(min_value=107.066, max_value=108.237, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([37.73, 38.51, 42.92, 44.43, 45.25, 45.58, 48.17, 51.33, 63.93, 65.76]),
           st.sampled_from([6.82, 8.95, 10.66, 12.31, 13.52, 15.97, 16.49, 19.58, 19.68, 21.35]),
           st.sampled_from([26.93, 29.22, 32.24, 42.7, 44.0, 46.17, 46.9, 49.35, 54.0, 55.34]),
           st.floats(min_value=28.136, max_value=29.562, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.926, max_value=113.218, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([33.84, 43.12, 50.68, 50.91, 54.92, 54.95, 56.45, 62.14, 63.93, 64.31]),
           st.floats(min_value=6.25, max_value=9.449, allow_nan=False),
           st.sampled_from([20.24, 28.0, 34.46, 36.67, 40.0, 43.46, 43.58, 44.0, 49.35, 55.34]),
           st.floats(min_value=28.136, max_value=29.562, exclude_min=True, allow_nan=False),
           st.floats(min_value=114.392, max_value=114.985, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([31.28, 35.7, 36.13, 38.66, 43.58, 43.79, 45.37, 49.71, 53.85, 54.92]),
           st.floats(min_value=9.452, max_value=17.447, exclude_min=True, allow_nan=False),
           st.sampled_from([32.56, 33.47, 35.0, 36.68, 38.0, 40.0, 46.56, 47.69, 54.0, 54.55]),
           st.floats(min_value=28.136, max_value=29.562, exclude_min=True, allow_nan=False),
           st.floats(min_value=114.392, max_value=114.985, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.651, max_value=16.078, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([31.48, 32.09, 35.7, 36.69, 41.17, 43.79, 44.32, 48.33, 50.91, 56.03]),
           st.sampled_from([13.07, 14.18, 15.35, 16.3, 17.72, 20.44, 20.46, 22.55, 24.19, 41.56]),
           st.sampled_from([15.5, 27.7, 28.07, 29.04, 30.12, 31.33, 33.47, 35.87, 40.26, 44.31]),
           st.floats(min_value=35.271, max_value=36.811, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.926, max_value=113.812, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.297, max_value=2.393, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([37.14, 39.36, 43.12, 46.24, 50.75, 51.33, 56.45, 61.45, 64.26, 67.8]),
           st.sampled_from([1.84, 2.63, 7.72, 9.59, 9.98, 13.11, 13.63, 14.66, 16.72, 18.9]),
           st.sampled_from([26.93, 31.02, 33.26, 35.11, 42.0, 43.2, 52.0, 53.0, 62.58, 63.12]),
           st.floats(min_value=42.977, max_value=43.71, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.926, max_value=113.812, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.297, max_value=2.393, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=56.446, max_value=64.019, allow_nan=False),
           st.sampled_from([6.46, 9.98, 10.06, 13.35, 16.06, 16.93, 18.02, 18.9, 20.9, 21.35]),
           st.sampled_from([25.32, 29.36, 35.0, 35.95, 39.0, 48.0, 52.0, 58.0, 58.25, 66.0]),
           st.floats(min_value=35.271, max_value=37.545, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.926, max_value=113.812, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.396, max_value=5.132, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.022, max_value=77.183, exclude_min=True, allow_nan=False),
           st.sampled_from([6.47, 9.43, 10.54, 12.51, 13.46, 14.04, 14.32, 20.27, 21.26, 30.46]),
           st.sampled_from([41.47, 42.87, 50.82, 51.0, 56.0, 60.9, 63.23, 74.44, 78.75, 79.65]),
           st.floats(min_value=35.271, max_value=37.545, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.926, max_value=113.812, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.396, max_value=5.132, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.65, 39.09, 44.36, 48.17, 48.9, 53.68, 56.45, 63.79, 63.93, 67.54]),
           st.sampled_from([5.87, 7.51, 13.11, 13.45, 14.92, 15.72, 16.48, 17.11, 23.14, 29.89]),
           st.sampled_from([19.07, 30.98, 33.26, 42.58, 42.7, 43.46, 49.35, 58.0, 58.25, 62.58]),
           st.floats(min_value=46.647, max_value=61.603, exclude_min=True, allow_nan=False),
           st.floats(min_value=104.416, max_value=112.999, allow_nan=False),
           st.floats(min_value=5.047, max_value=9.073, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([57.04, 57.29, 60.04, 60.42, 63.77, 69.76, 71.24, 72.22, 81.66, 86.04]),
           st.sampled_from([-0.26, 5.27, 14.55, 15.38, 15.4, 24.82, 28.75, 30.35, 36.84, 48.9]),
           st.sampled_from([49.2, 52.0, 63.0, 65.48, 67.5, 67.9, 74.44, 77.48, 83.35, 86.96]),
           st.floats(min_value=46.647, max_value=61.603, exclude_min=True, allow_nan=False),
           st.floats(min_value=113.001, max_value=113.36, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.047, max_value=9.073, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([41.65, 45.58, 50.09, 50.75, 53.94, 54.75, 54.92, 61.82, 66.51, 69.0]),
           st.sampled_from([-1.33, 3.68, 5.87, 6.82, 10.19, 13.82, 16.55, 18.97, 19.97, 27.34]),
           st.sampled_from([28.0, 29.22, 36.67, 39.71, 40.18, 44.58, 50.96, 52.0, 69.02, 90.56]),
           st.floats(min_value=46.647, max_value=61.603, exclude_min=True, allow_nan=False),
           st.floats(min_value=114.801, max_value=115.312, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.047, max_value=9.073, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([41.19, 52.2, 65.54, 73.64, 76.33, 77.66, 80.43, 81.08, 85.58, 89.68]),
           st.sampled_from([-3.76, 12.1, 17.21, 19.24, 20.69, 20.8, 26.34, 30.47, 32.7, 33.28]),
           st.sampled_from([41.0, 63.01, 65.36, 66.54, 67.5, 69.22, 76.03, 79.69, 91.78, 100.74]),
           st.floats(min_value=46.647, max_value=61.603, exclude_min=True, allow_nan=False),
           st.floats(min_value=107.903, max_value=117.358, allow_nan=False),
           st.floats(min_value=9.076, max_value=10.476, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([31.28, 38.66, 41.35, 44.32, 45.37, 47.66, 48.33, 54.92, 55.84, 66.29]),
           st.floats(min_value=18.477, max_value=24.733, allow_nan=False),
           st.sampled_from([15.5, 20.03, 20.7, 26.79, 31.33, 32.14, 33.1, 36.18, 37.17, 47.0]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=117.501, exclude_min=True, allow_nan=False),
           st.floats(min_value=-5.908, max_value=-4.621, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.38, 36.42, 38.13, 38.51, 40.35, 46.37, 50.91, 51.62, 61.45, 69.0]),
           st.floats(min_value=18.477, max_value=24.733, allow_nan=False),
           st.sampled_from([19.07, 24.0, 29.22, 37.0, 44.58, 46.9, 51.6, 63.12, 66.0, 69.02]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=117.501, exclude_min=True, allow_nan=False),
           st.floats(min_value=-4.618, max_value=-2.655, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([36.42, 42.52, 46.37, 46.64, 49.83, 51.53, 64.26, 65.76, 66.51, 89.01]),
           st.floats(min_value=15.473, max_value=20.978, allow_nan=False),
           st.sampled_from([25.32, 31.73, 33.26, 37.97, 43.0, 48.0, 58.25, 61.01, 69.02, 90.56]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=118.067, max_value=127.067, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.948, max_value=5.199, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([33.84, 35.88, 40.35, 43.12, 46.64, 47.32, 48.8, 50.09, 52.86, 63.03]),
           st.floats(min_value=20.981, max_value=21.731, exclude_min=True, allow_nan=False),
           st.sampled_from([25.32, 28.94, 32.24, 32.39, 33.63, 36.0, 40.35, 51.61, 54.0, 55.5]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=118.067, max_value=127.067, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.828, max_value=0.479, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([40.25, 43.35, 44.32, 44.55, 45.54, 46.39, 48.11, 48.92, 50.91, 54.12]),
           st.floats(min_value=20.981, max_value=21.731, exclude_min=True, allow_nan=False),
           st.sampled_from([15.5, 27.78, 28.07, 29.04, 30.3, 35.56, 38.0, 39.61, 42.2, 47.0]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=118.067, max_value=127.067, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.482, max_value=1.425, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([26.15, 31.28, 38.66, 39.06, 41.73, 43.2, 43.35, 48.11, 50.21, 68.83]),
           st.floats(min_value=24.736, max_value=29.674, exclude_min=True, allow_nan=False),
           st.sampled_from([20.03, 20.7, 27.78, 28.32, 29.0, 35.33, 35.56, 42.2, 47.69, 54.0]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=126.502, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.948, max_value=5.199, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([35.49, 36.13, 40.25, 41.17, 43.2, 43.58, 43.79, 54.12, 63.07, 74.43]),
           st.sampled_from([5.01, 10.06, 10.76, 12.99, 13.53, 16.3, 17.9, 19.01, 19.23, 22.22]),
           st.sampled_from([14.0, 25.02, 35.33, 35.87, 36.0, 36.18, 36.68, 37.17, 37.83, 50.09]),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=119.511, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.202, max_value=5.788, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([31.48, 32.09, 35.49, 38.66, 40.56, 43.2, 43.92, 63.07, 63.83, 74.43]),
           st.sampled_from([6.99, 8.4, 11.7, 14.93, 15.4, 17.32, 21.06, 24.41, 24.89, 41.56]),
           st.sampled_from([15.59, 20.03, 31.33, 33.47, 35.87, 36.1, 39.61, 40.26, 42.69, 44.31]),
           st.floats(min_value=28.136, max_value=28.639, exclude_min=True, allow_nan=False),
           st.floats(min_value=128.117, max_value=135.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.202, max_value=5.788, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([39.09, 40.35, 42.92, 45.08, 46.64, 49.0, 51.08, 53.68, 59.73, 63.79]),
           st.sampled_from([-5.85, 5.07, 6.82, 7.51, 8.3, 8.57, 8.69, 13.29, 14.17, 16.74]),
           st.sampled_from([25.32, 36.0, 39.71, 40.35, 44.0, 48.1, 55.57, 58.25, 61.01, 66.0]),
           st.floats(min_value=30.656, max_value=48.81, exclude_min=True, allow_nan=False),
           st.floats(min_value=128.117, max_value=135.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.202, max_value=5.788, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=37.838, max_value=40.759, allow_nan=False),
           st.sampled_from([1.84, 8.57, 9.98, 11.94, 13.88, 14.5, 16.49, 16.93, 19.44, 26.08]),
           st.floats(min_value=62.383, max_value=74.478, allow_nan=False),
           st.floats(min_value=28.136, max_value=28.612, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=126.502, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.136, max_value=9.724, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.762, max_value=58.575, exclude_min=True, allow_nan=False),
           st.sampled_from([7.47, 9.65, 11.08, 11.92, 13.04, 16.58, 17.98, 19.96, 21.93, 29.76]),
           st.floats(min_value=62.383, max_value=74.478, allow_nan=False),
           st.floats(min_value=28.136, max_value=28.612, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=126.502, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.136, max_value=9.724, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([33.79, 39.66, 40.41, 41.65, 42.52, 50.68, 51.53, 59.17, 62.14, 63.96]),
           st.sampled_from([5.59, 6.82, 8.88, 9.41, 10.19, 13.63, 13.82, 14.66, 16.72, 19.58]),
           st.floats(min_value=62.383, max_value=74.478, allow_nan=False),
           st.floats(min_value=30.522, max_value=48.703, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=126.502, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.136, max_value=9.724, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([53.94, 60.63, 63.36, 65.67, 66.8, 73.64, 75.3, 85.64, 90.51, 118.14]),
           st.sampled_from([6.33, 12.49, 13.28, 13.71, 20.6, 22.43, 23.94, 29.09, 42.69, 48.9]),
           st.floats(min_value=74.481, max_value=84.732, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.136, max_value=46.794, exclude_min=True, allow_nan=False),
           st.floats(min_value=117.361, max_value=126.502, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.136, max_value=9.724, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.006, max_value=57.219, allow_nan=False),
           st.sampled_from([1.84, 8.84, 9.75, 10.1, 13.43, 14.92, 16.54, 20.24, 22.64, 23.14]),
           st.sampled_from([20.24, 26.93, 28.94, 31.02, 35.95, 37.97, 43.2, 50.0, 50.45, 57.0]),
           st.sampled_from([25.97, 33.09, 33.11, 35.42, 36.16, 36.97, 39.81, 49.09, 51.25, 53.13]),
           st.sampled_from([113.78, 114.37, 114.51, 118.55, 124.13, 126.92, 133.28, 135.62, 137.59, 147.89]),
           st.floats(min_value=16.081, max_value=16.881, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=57.222, max_value=71.743, exclude_min=True, allow_nan=False),
           st.sampled_from([-6.55, 9.81, 13.91, 14.32, 15.4, 17.21, 20.12, 21.94, 29.09, 30.35]),
           st.sampled_from([45.78, 47.87, 48.5, 59.18, 63.01, 65.48, 83.35, 86.0, 86.96, 93.82]),
           st.sampled_from([19.29, 35.39, 43.15, 46.64, 47.29, 51.77, 55.92, 56.93, 59.53, 63.43]),
           st.sampled_from([95.44, 105.14, 110.64, 110.71, 110.86, 114.87, 119.43, 122.65, 146.47, 163.07]),
           st.floats(min_value=16.081, max_value=16.881, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([50.83, 55.51, 58.78, 60.63, 74.38, 77.41, 79.94, 85.29, 87.68, 94.17]),
           st.sampled_from([5.27, 14.31, 19.28, 19.76, 21.7, 24.51, 24.82, 26.73, 30.35, 30.47]),
           st.sampled_from([39.0, 40.8, 44.0, 49.2, 53.0, 57.74, 63.0, 64.0, 91.0, 96.28]),
           st.sampled_from([35.66, 41.19, 42.14, 53.71, 54.31, 56.93, 59.83, 61.58, 63.43, 67.01]),
           st.floats(min_value=111.411, max_value=121.743, allow_nan=False),
           st.floats(min_value=20.087, max_value=22.247, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.38, 40.41, 41.65, 44.36, 51.33, 54.95, 59.17, 67.54, 69.0, 69.4]),
           st.sampled_from([-2.97, 2.63, 8.3, 8.88, 9.15, 9.39, 9.41, 12.94, 13.96, 19.58]),
           st.sampled_from([19.07, 29.36, 30.9, 36.0, 41.58, 43.9, 53.73, 54.0, 55.57, 90.56]),
           st.sampled_from([25.97, 30.26, 33.11, 35.95, 36.87, 37.12, 37.98, 42.44, 43.96, 62.94]),
           st.floats(min_value=121.746, max_value=122.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=20.087, max_value=22.247, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([47.74, 50.83, 56.56, 69.76, 71.19, 74.85, 77.12, 79.25, 85.64, 95.38]),
           st.sampled_from([3.97, 9.84, 14.32, 15.17, 20.8, 22.18, 23.9, 29.09, 30.36, 31.54]),
           st.sampled_from([39.1, 43.77, 62.86, 66.15, 67.63, 68.06, 70.65, 74.44, 82.68, 89.6]),
           st.sampled_from([42.95, 43.26, 47.03, 50.48, 52.88, 55.18, 60.07, 60.42, 61.63, 66.96]),
           st.floats(min_value=124.016, max_value=131.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=20.087, max_value=22.247, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([30.74, 42.52, 48.17, 48.8, 51.31, 53.94, 59.73, 61.82, 63.03, 64.31]),
           st.sampled_from([-2.97, 8.69, 8.84, 9.15, 9.41, 9.59, 10.22, 13.21, 14.21, 16.21]),
           st.sampled_from([26.93, 29.36, 36.64, 40.18, 44.58, 51.0, 51.87, 55.34, 55.57, 69.02]),
           st.sampled_from([22.54, 29.74, 31.53, 32.35, 35.88, 36.17, 36.87, 44.6, 44.62, 48.76]),
           st.sampled_from([113.91, 116.8, 118.34, 119.38, 123.8, 126.97, 128.98, 130.18, 130.35, 139.12]),
           st.floats(min_value=30.892, max_value=30.963, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([48.26, 58.6, 60.75, 63.36, 64.81, 70.25, 70.95, 77.24, 77.69, 85.35]),
           st.sampled_from([9.43, 14.04, 14.31, 16.42, 17.39, 18.77, 22.18, 24.7, 24.82, 33.28]),
           st.sampled_from([41.47, 44.99, 45.0, 45.78, 47.56, 50.95, 57.2, 59.18, 65.48, 71.67]),
           st.sampled_from([36.03, 47.29, 49.64, 51.01, 53.75, 55.18, 63.43, 63.92, 67.01, 69.51]),
           st.sampled_from([95.44, 99.71, 101.72, 102.34, 103.05, 116.6, 119.24, 134.63, 135.08, 135.63]),
           st.floats(min_value=31.251, max_value=108.708, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
