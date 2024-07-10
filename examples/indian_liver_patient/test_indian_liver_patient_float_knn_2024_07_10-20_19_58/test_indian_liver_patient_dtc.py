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
    request.cls.data['n_test'] = 111
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_indian_liver_patient_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestIndianLiverPatientProperty:

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=63.0, max_value=211.49, allow_nan=False),
           st.floats(min_value=10.0, max_value=26.49, allow_nan=False),
           st.sampled_from([19.0, 20.0, 34.0, 38.0, 47.0, 67.0, 74.0, 145.0, 247.0, 350.0]),
           st.sampled_from([2.8, 3.0, 3.6, 5.8, 6.4, 7.1, 7.2, 7.6, 8.2, 8.9]),
           st.sampled_from([0.9, 2.0, 2.3, 2.8, 3.0, 3.6, 3.7, 4.2, 4.5, 4.8]),
           st.sampled_from([0.35, 0.48, 0.52, 0.75, 0.87, 1.06, 1.1, 1.3, 1.6, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=63.0, max_value=211.49, allow_nan=False),
           st.floats(min_value=26.51, max_value=45.99, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 17.0, 28.0, 34.0, 40.0, 45.0, 46.0, 53.0, 64.0, 85.0]),
           st.sampled_from([5.1, 5.5, 5.7, 5.8, 6.3, 6.4, 7.0, 7.3, 8.0, 8.2]),
           st.sampled_from([2.1, 2.8, 3.2, 3.7, 3.8, 3.9, 4.0, 4.3, 4.7, 4.9]),
           st.sampled_from([0.37, 0.52, 0.67, 0.71, 0.92, 0.95, 1.03, 1.06, 1.7, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=211.49, allow_nan=False),
           st.floats(min_value=10.0, max_value=45.99, allow_nan=False),
           st.sampled_from([24.0, 27.0, 30.0, 34.0, 40.0, 46.0, 53.0, 57.0, 58.0, 108.0]),
           st.sampled_from([4.5, 5.5, 5.7, 5.9, 6.0, 6.7, 7.1, 7.3, 7.9, 8.2]),
           st.sampled_from([2.2, 2.6, 2.7, 2.9, 3.0, 3.8, 4.1, 4.3, 4.7, 4.9]),
           st.floats(min_value=0.3, max_value=1.249, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=187.99, allow_nan=False),
           st.floats(min_value=10.0, max_value=45.99, allow_nan=False),
           st.sampled_from([15.0, 18.0, 24.0, 32.0, 35.0, 42.0, 43.0, 70.0, 103.0, 178.0]),
           st.sampled_from([5.2, 5.7, 6.0, 6.1, 6.6, 6.7, 7.2, 7.3, 7.5, 8.1]),
           st.sampled_from([1.4, 1.9, 2.8, 2.9, 3.0, 3.1, 3.3, 3.7, 4.6, 4.9]),
           st.floats(min_value=1.251, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=188.01, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=45.99, allow_nan=False),
           st.sampled_from([11.0, 33.0, 39.0, 52.0, 57.0, 79.0, 89.0, 152.0, 202.0, 731.0]),
           st.sampled_from([4.6, 5.0, 5.8, 6.7, 7.1, 7.4, 7.7, 7.8, 8.7, 8.9]),
           st.sampled_from([0.9, 1.6, 1.8, 2.1, 2.8, 3.5, 3.7, 4.7, 4.8, 5.5]),
           st.floats(min_value=1.251, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=63.0, max_value=211.49, allow_nan=False),
           st.floats(min_value=46.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 14.0, 15.0, 20.0, 25.0, 42.0, 47.0, 64.0, 84.0, 231.0]),
           st.sampled_from([3.8, 4.6, 4.8, 5.3, 5.6, 6.1, 7.1, 7.5, 7.8, 8.3]),
           st.sampled_from([1.6, 2.1, 2.2, 2.4, 2.5, 3.1, 3.3, 4.0, 4.5, 4.6]),
           st.sampled_from([0.45, 0.52, 0.58, 0.8, 0.9, 0.92, 1.0, 1.3, 1.5, 1.9]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=25.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=211.49, allow_nan=False),
           st.floats(min_value=46.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 27.0, 29.0, 30.0, 70.0, 72.0, 83.0, 100.0, 221.0, 950.0]),
           st.sampled_from([3.8, 5.2, 6.1, 6.4, 6.5, 7.5, 8.0, 8.3, 8.4, 8.5]),
           st.sampled_from([1.5, 1.9, 2.0, 2.3, 2.6, 2.8, 3.1, 4.3, 4.4, 4.7]),
           st.sampled_from([0.3, 0.48, 0.88, 0.97, 1.0, 1.06, 1.1, 1.16, 1.27, 1.66]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.4, 0.6, 0.7, 0.8, 1.1, 1.2, 1.4, 1.6, 2.3]),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([10.0, 13.0, 14.0, 16.0, 25.0, 38.0, 44.0, 45.0, 88.0, 181.0]),
           st.sampled_from([15.0, 25.0, 36.0, 41.0, 42.0, 70.0, 90.0, 108.0, 127.0, 148.0]),
           st.sampled_from([3.9, 4.6, 5.1, 6.0, 6.1, 6.3, 6.6, 7.1, 7.9, 8.1]),
           st.sampled_from([1.8, 2.3, 2.5, 2.6, 2.7, 3.2, 3.3, 4.3, 4.9, 5.0]),
           st.floats(min_value=0.3, max_value=0.619, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=0.74, allow_nan=False),
           st.sampled_from([0.3, 1.0, 1.6, 3.0, 3.7, 4.5, 6.2, 7.2, 11.8, 12.1]),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([22.0, 29.0, 32.0, 42.0, 60.0, 75.0, 79.0, 157.0, 173.0, 390.0]),
           st.sampled_from([45.0, 73.0, 78.0, 79.0, 90.0, 95.0, 134.0, 168.0, 497.0, 576.0]),
           st.sampled_from([2.7, 4.0, 4.8, 6.3, 6.6, 6.8, 7.3, 8.3, 8.5, 9.6]),
           st.sampled_from([1.5, 2.1, 2.5, 2.6, 2.7, 3.3, 3.4, 3.6, 4.0, 4.2]),
           st.floats(min_value=0.622, max_value=1.049, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=46.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.76, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.1, 0.3, 0.5, 0.8, 0.9, 1.1, 1.2, 1.6, 3.0, 3.6]),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([19.0, 23.0, 27.0, 28.0, 31.0, 45.0, 48.0, 59.0, 61.0, 63.0]),
           st.sampled_from([13.0, 15.0, 26.0, 42.0, 43.0, 53.0, 57.0, 90.0, 108.0, 110.0]),
           st.sampled_from([3.9, 4.5, 4.6, 5.0, 5.5, 5.6, 6.1, 6.7, 7.5, 8.4]),
           st.sampled_from([1.7, 1.9, 2.3, 2.9, 3.3, 3.7, 4.1, 4.2, 4.3, 4.9]),
           st.floats(min_value=0.622, max_value=1.049, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.76, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2, 0.3, 0.5, 0.7, 0.8, 1.2, 1.4, 1.6, 3.0, 3.2]),
           st.floats(min_value=63.0, max_value=119.49, allow_nan=False),
           st.sampled_from([16.0, 19.0, 25.0, 32.0, 36.0, 45.0, 49.0, 55.0, 61.0, 64.0]),
           st.sampled_from([10.0, 20.0, 21.0, 36.0, 67.0, 82.0, 92.0, 110.0, 148.0, 231.0]),
           st.sampled_from([4.5, 5.0, 5.5, 6.1, 6.6, 7.4, 7.6, 8.2, 8.3, 9.2]),
           st.sampled_from([1.6, 1.7, 1.9, 2.2, 2.5, 2.9, 3.3, 3.7, 3.8, 4.7]),
           st.floats(min_value=0.622, max_value=1.049, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.76, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.8, 2.3, 2.9, 4.6, 7.2, 7.8, 10.8, 11.4, 12.1, 12.6]),
           st.floats(min_value=119.51, max_value=144.49, exclude_min=True, allow_nan=False),
           st.sampled_from([21.0, 39.0, 41.0, 58.0, 85.0, 86.0, 123.0, 190.0, 622.0, 875.0]),
           st.sampled_from([11.0, 23.0, 41.0, 60.0, 77.0, 126.0, 152.0, 231.0, 236.0, 330.0]),
           st.sampled_from([3.8, 4.7, 5.0, 5.3, 5.8, 6.3, 6.8, 6.9, 8.4, 9.2]),
           st.sampled_from([1.0, 1.9, 2.6, 2.8, 2.9, 3.1, 3.5, 4.2, 4.6, 5.5]),
           st.floats(min_value=0.622, max_value=1.049, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.4, 0.7, 1.6, 1.7, 4.9, 5.1, 7.6, 10.8, 14.2, 18.3]),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([56.0, 67.0, 68.0, 96.0, 114.0, 189.0, 196.0, 230.0, 233.0, 875.0]),
           st.sampled_from([32.0, 35.0, 67.0, 72.0, 86.0, 92.0, 101.0, 221.0, 401.0, 441.0]),
           st.sampled_from([3.0, 4.3, 5.3, 5.8, 6.5, 6.6, 6.7, 6.8, 8.0, 9.6]),
           st.sampled_from([1.4, 1.7, 1.8, 2.2, 2.5, 2.6, 2.7, 2.8, 3.1, 5.5]),
           st.floats(min_value=1.052, max_value=1.448, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.24, allow_nan=False),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([13.0, 21.0, 24.0, 25.0, 33.0, 37.0, 41.0, 45.0, 57.0, 64.0]),
           st.sampled_from([17.0, 18.0, 40.0, 44.0, 46.0, 56.0, 82.0, 90.0, 108.0, 231.0]),
           st.sampled_from([3.7, 5.3, 5.9, 6.1, 6.2, 6.3, 6.8, 7.5, 8.3, 8.4]),
           st.sampled_from([1.4, 1.8, 2.1, 2.8, 2.9, 3.7, 4.2, 4.4, 4.6, 4.9]),
           st.floats(min_value=1.451, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.27, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=144.49, allow_nan=False),
           st.sampled_from([31.0, 32.0, 47.0, 51.0, 75.0, 91.0, 96.0, 112.0, 118.0, 950.0]),
           st.sampled_from([28.0, 51.0, 66.0, 71.0, 100.0, 126.0, 140.0, 142.0, 178.0, 245.0]),
           st.sampled_from([3.0, 4.1, 5.2, 5.4, 5.6, 5.9, 6.4, 7.7, 8.3, 8.6]),
           st.sampled_from([0.9, 1.7, 2.0, 2.5, 2.6, 2.8, 3.5, 3.9, 4.3, 4.6]),
           st.floats(min_value=1.451, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=36.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.6, 0.8, 2.2, 2.8, 4.0, 4.2, 4.6, 10.2, 10.8, 12.6]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([18.0, 32.0, 46.0, 58.0, 130.0, 149.0, 176.0, 350.0, 794.0, 1500.0]),
           st.floats(min_value=2.7, max_value=6.24, allow_nan=False),
           st.sampled_from([1.0, 1.4, 1.5, 1.7, 1.9, 3.2, 3.7, 3.8, 4.0, 5.5]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=36.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 1.0, 1.4, 2.3, 3.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([10.0, 12.0, 14.0, 22.0, 25.0, 54.0, 70.0, 71.0, 85.0, 127.0]),
           st.floats(min_value=2.7, max_value=6.24, allow_nan=False),
           st.sampled_from([1.8, 2.3, 2.6, 3.0, 3.1, 3.2, 3.4, 4.4, 4.5, 5.0]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 3.0, 3.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([14.0, 22.0, 25.0, 35.0, 53.0, 58.0, 59.0, 85.0, 110.0, 178.0]),
           st.floats(min_value=6.26, max_value=9.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9, max_value=2.94, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([1.8, 2.3, 2.6, 4.6, 5.5, 6.2, 7.6, 7.8, 11.3, 11.7]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([15.0, 39.0, 103.0, 143.0, 150.0, 168.0, 188.0, 441.0, 497.0, 623.0]),
           st.floats(min_value=6.26, max_value=6.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.97, max_value=3.44, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.6, 2.3, 3.0, 3.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([15.0, 20.0, 25.0, 27.0, 30.0, 39.0, 70.0, 85.0, 90.0, 110.0]),
           st.floats(min_value=6.87, max_value=9.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.97, max_value=3.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.8, 1.4, 2.8, 3.3, 4.2, 5.2, 6.0, 8.4, 12.1, 14.1]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([14.0, 32.0, 60.0, 73.0, 82.0, 83.0, 140.0, 230.0, 367.0, 562.0]),
           st.floats(min_value=6.87, max_value=9.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.37, max_value=3.44, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.2, 0.5, 2.0, 3.2, 4.2, 6.4, 8.5, 10.0, 10.8, 13.7]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.floats(min_value=10.0, max_value=13.49, allow_nan=False),
           st.floats(min_value=6.26, max_value=9.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.47, max_value=5.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.4, 0.6, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4, 2.3, 3.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.floats(min_value=13.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.26, max_value=9.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.47, max_value=5.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.328, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([3.6, 3.7, 4.6, 6.0, 8.9, 9.0, 11.3, 12.6, 14.1, 18.3]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=19.49, allow_nan=False),
           st.sampled_from([37.0, 61.0, 70.0, 82.0, 86.0, 89.0, 232.0, 348.0, 367.0, 623.0]),
           st.floats(min_value=6.26, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.9, 2.4, 2.7, 3.2, 3.6, 4.1, 4.5, 4.6, 4.7, 4.8]),
           st.floats(min_value=1.331, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=57.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=0.83, allow_nan=False),
           st.sampled_from([0.1, 1.0, 1.1, 1.3, 2.8, 3.2, 3.7, 3.9, 7.0, 14.1]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=18.49, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.8, 1.9, 2.2, 2.9, 3.2, 3.3, 3.5, 4.0, 4.2, 4.3]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=57.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.86, max_value=1.33, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2, 0.4, 0.6, 0.8, 0.9, 1.1, 1.4, 1.6, 2.3, 3.0]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=18.49, allow_nan=False),
           st.floats(min_value=2.7, max_value=6.99, allow_nan=False),
           st.sampled_from([1.4, 1.6, 1.8, 2.3, 2.4, 2.6, 2.7, 3.3, 3.7, 5.0]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=57.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.86, max_value=1.33, exclude_min=True, allow_nan=False),
           st.sampled_from([0.7, 1.4, 2.6, 4.5, 5.1, 6.0, 8.8, 9.5, 11.8, 14.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=18.49, allow_nan=False),
           st.floats(min_value=7.01, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([1.9, 2.0, 2.3, 2.4, 2.6, 2.8, 3.7, 4.2, 4.3, 4.9]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=57.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.1, 1.2, 2.3, 3.6]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=18.49, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.6, 2.2, 2.4, 2.6, 2.7, 3.4, 3.6, 3.8, 4.1, 5.0]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 2.3, 3.0]),
           st.floats(min_value=144.51, max_value=166.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.4, 1.6, 1.9, 2.1, 2.9, 3.0, 3.1, 3.3, 3.5, 3.9]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=167.01, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.floats(min_value=0.9, max_value=3.88, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=167.01, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.floats(min_value=3.91, max_value=5.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=167.01, max_value=202.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.8, 2.3, 2.6, 2.8, 3.0, 3.4, 4.1, 4.3, 4.5, 4.7]),
           st.floats(min_value=0.3, max_value=0.649, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=38.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=167.01, max_value=186.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=5.83, allow_nan=False),
           st.sampled_from([1.0, 1.8, 2.2, 2.4, 2.8, 3.0, 3.4, 3.6, 3.8, 4.9]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=167.01, max_value=186.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=5.83, allow_nan=False),
           st.sampled_from([1.6, 1.8, 1.9, 2.0, 2.2, 2.3, 3.2, 4.0, 4.6, 4.7]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=167.01, max_value=186.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.86, max_value=6.59, exclude_min=True, allow_nan=False),
           st.sampled_from([1.7, 2.0, 2.4, 2.6, 3.0, 3.4, 4.4, 4.5, 4.7, 4.9]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=167.01, max_value=186.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.62, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([0.9, 1.9, 2.0, 2.8, 3.0, 3.2, 3.4, 3.5, 4.2, 4.6]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=187.01, max_value=202.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=38.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.4, 1.8, 2.9, 3.8, 4.3, 4.4, 4.5, 4.7, 4.9, 5.0]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=187.01, max_value=202.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=31.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([0.9, 1.0, 2.8, 3.1, 3.3, 3.6, 4.2, 4.5, 4.7, 4.8]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=187.01, max_value=202.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=32.01, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.8, 2.5, 2.6, 2.8, 3.6, 3.7, 4.0, 4.2, 4.3, 4.9]),
           st.floats(min_value=0.652, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=45.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=202.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.4, 1.7, 2.0, 2.1, 2.4, 2.5, 2.8, 2.9, 3.1, 5.5]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=45.51, max_value=51.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.24, allow_nan=False),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=7.74, allow_nan=False),
           st.sampled_from([1.0, 1.8, 2.4, 2.7, 3.2, 3.6, 3.7, 4.0, 4.2, 4.4]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=45.51, max_value=51.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=0.74, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.24, allow_nan=False),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.76, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([1.6, 1.7, 2.2, 2.4, 2.6, 3.0, 3.6, 3.8, 4.1, 4.4]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=45.51, max_value=51.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.76, max_value=1.33, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.24, allow_nan=False),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.76, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 1.7, 1.8, 1.9, 2.4, 2.5, 3.3, 3.9, 4.8, 5.5]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=45.51, max_value=51.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.27, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.floats(min_value=0.9, max_value=4.14, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=45.51, max_value=51.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.floats(min_value=0.27, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.floats(min_value=4.17, max_value=5.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.8, 1.8, 3.0, 3.3, 4.0, 4.9, 5.5, 8.4]),
           st.floats(min_value=144.51, max_value=190.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=6.14, allow_nan=False),
           st.sampled_from([1.9, 2.3, 2.4, 2.5, 2.7, 2.8, 3.0, 4.1, 4.6, 4.7]),
           st.floats(min_value=0.3, max_value=1.049, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([1.3, 1.6, 1.9, 2.1, 2.2, 2.7, 5.2, 9.0, 12.6, 17.1]),
           st.floats(min_value=144.51, max_value=190.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.17, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([2.3, 2.6, 3.0, 3.3, 3.6, 3.7, 4.6, 4.7, 4.8, 4.9]),
           st.floats(min_value=0.3, max_value=1.049, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.01, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.2, 0.5, 0.6, 0.7, 1.0, 1.1, 1.2, 1.6, 3.0, 3.2]),
           st.floats(min_value=144.51, max_value=190.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.17, max_value=8.28, exclude_min=True, allow_nan=False),
           st.sampled_from([1.9, 2.2, 2.3, 2.4, 2.7, 2.8, 3.1, 3.3, 4.9, 5.0]),
           st.floats(min_value=0.3, max_value=1.049, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.2, 2.3, 3.0, 3.6]),
           st.floats(min_value=191.01, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.6, 2.4, 2.6, 2.7, 3.5, 3.7, 4.0, 4.1, 4.3, 5.0]),
           st.floats(min_value=0.3, max_value=1.049, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.4, 0.6, 0.7, 1.0, 1.1, 1.2, 1.4, 1.6, 3.2]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=43.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([2.2, 2.8, 3.1, 3.4, 3.5, 4.1, 4.2, 4.5, 4.6, 5.0]),
           st.floats(min_value=1.052, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.2, 1.6, 2.3]),
           st.floats(min_value=144.51, max_value=174.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=39.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.6, 1.9, 2.4, 2.7, 2.8, 2.9, 3.1, 3.4, 4.4, 4.6]),
           st.floats(min_value=0.3, max_value=0.473, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([1.2, 2.2, 2.5, 3.6, 4.1, 4.9, 5.2, 5.5, 7.8, 8.2]),
           st.floats(min_value=144.51, max_value=174.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=39.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.0, 1.7, 1.8, 2.1, 2.5, 3.5, 3.6, 3.8, 4.4, 4.8]),
           st.floats(min_value=0.476, max_value=1.679, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.2, 0.3, 0.6, 0.8, 1.1, 1.2, 1.4, 2.3, 3.0, 3.2]),
           st.floats(min_value=144.51, max_value=174.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=40.01, max_value=51.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([2.1, 2.3, 2.7, 3.1, 3.2, 3.7, 4.0, 4.4, 4.7, 4.9]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.5, 1.0, 2.0, 2.7, 6.0, 10.0, 10.2, 11.3, 11.4, 18.3]),
           st.floats(min_value=144.51, max_value=174.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.0, 1.4, 1.5, 1.7, 1.9, 2.0, 2.9, 3.3, 3.9, 4.8]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([0.1, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 3.6]),
           st.floats(min_value=174.51, max_value=185.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.7, 2.0, 2.1, 2.3, 2.7, 3.0, 3.9, 4.1, 4.3, 4.4]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.33, allow_nan=False),
           st.sampled_from([2.1, 3.2, 4.3, 5.5, 6.2, 7.0, 7.2, 8.5, 12.6, 13.7]),
           st.floats(min_value=186.01, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.0, 1.6, 2.2, 2.3, 3.2, 3.8, 4.3, 4.6, 4.7, 4.8]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.36, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.1, 0.4, 0.5, 0.7, 0.8, 0.9, 1.2, 1.4, 3.0, 3.6]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([28.0, 30.0, 33.0, 34.0, 40.0, 43.0, 44.0, 51.0, 59.0, 108.0]),
           st.floats(min_value=2.7, max_value=8.28, allow_nan=False),
           st.sampled_from([1.4, 1.8, 2.0, 2.9, 3.2, 3.3, 3.6, 3.7, 4.2, 4.5]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=70.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.5, 0.8, 0.9, 1.2, 1.4, 2.3, 3.2, 3.6]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 16.0, 20.0, 32.0, 33.0, 67.0, 82.0, 85.0, 110.0, 127.0]),
           st.floats(min_value=8.31, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.8, 2.0, 2.8, 2.9, 3.0, 3.1, 3.3, 3.4, 4.0, 4.2]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=70.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([1.1, 2.6, 3.3, 3.7, 6.0, 6.2, 6.4, 7.0, 8.5, 10.4]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([39.0, 41.0, 67.0, 83.0, 105.0, 114.0, 149.0, 275.0, 562.0, 4929.0]),
           st.sampled_from([3.0, 3.6, 5.2, 6.2, 6.3, 8.2, 8.4, 8.9, 9.2, 9.6]),
           st.sampled_from([0.9, 1.0, 1.6, 2.3, 2.6, 3.4, 3.9, 4.1, 4.5, 4.6]),
           st.floats(min_value=0.3, max_value=1.679, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.2, 2.3, 3.6]),
           st.floats(min_value=144.51, max_value=211.49, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 15.0, 20.0, 29.0, 30.0, 31.0, 46.0, 48.0, 60.0, 152.0]),
           st.sampled_from([13.0, 15.0, 19.0, 27.0, 39.0, 52.0, 56.0, 57.0, 70.0, 148.0]),
           st.sampled_from([3.7, 4.5, 5.2, 6.2, 6.9, 7.1, 7.5, 7.6, 8.4, 8.5]),
           st.sampled_from([1.4, 2.3, 2.5, 2.6, 2.9, 3.1, 3.3, 4.3, 4.6, 5.0]),
           st.floats(min_value=1.682, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=7.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.5, 0.6, 0.7, 0.9, 1.0, 1.2, 1.6, 3.0, 3.2]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 18.0, 21.0, 24.0, 34.0, 35.0, 37.0, 44.0, 59.0, 61.0]),
           st.sampled_from([12.0, 14.0, 31.0, 35.0, 43.0, 46.0, 56.0, 67.0, 82.0, 85.0]),
           st.sampled_from([4.9, 5.7, 5.9, 6.2, 7.3, 7.5, 7.6, 7.7, 7.9, 8.1]),
           st.sampled_from([1.6, 1.8, 2.2, 2.3, 2.9, 3.6, 3.9, 4.2, 4.5, 5.0]),
           st.sampled_from([0.45, 0.5, 0.67, 0.76, 0.8, 0.95, 0.96, 1.8, 1.85, 1.9]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.1, 0.4, 0.5, 0.8, 1.0, 1.6, 2.3, 3.0, 3.2, 3.6]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=10.99, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.7, 4.6, 6.3, 6.8, 6.9, 7.2, 7.4, 7.8, 8.0, 9.2]),
           st.sampled_from([2.1, 2.4, 2.6, 2.8, 4.0, 4.1, 4.2, 4.4, 4.5, 4.7]),
           st.sampled_from([0.7, 0.75, 0.92, 1.0, 1.16, 1.3, 1.4, 1.5, 1.7, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=17.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([2.1, 2.2, 2.9, 3.0, 3.3, 4.9, 6.2, 6.4, 8.8, 13.7]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.6, 4.3, 4.4, 5.0, 5.9, 6.8, 7.1, 7.6, 8.9, 9.6]),
           st.sampled_from([1.7, 1.9, 2.4, 3.1, 3.3, 3.7, 3.8, 4.0, 4.3, 5.5]),
           st.sampled_from([0.35, 0.39, 0.47, 0.7, 0.88, 1.0, 1.1, 1.2, 1.25, 1.3]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=17.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=2.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.39, allow_nan=False),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.8, 5.1, 5.5, 5.7, 6.1, 6.3, 6.4, 6.8, 8.4, 9.2]),
           st.sampled_from([2.1, 2.2, 2.8, 3.1, 3.3, 3.4, 3.5, 3.6, 3.8, 4.7]),
           st.floats(min_value=0.3, max_value=1.448, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=17.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=2.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.42, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.8, 4.4, 5.6, 6.7, 7.3, 8.2, 8.4, 8.9, 9.2, 9.5]),
           st.sampled_from([1.0, 1.4, 2.2, 2.6, 2.7, 2.9, 3.4, 3.6, 4.5, 4.6]),
           st.floats(min_value=0.3, max_value=1.448, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=17.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=2.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.3, 0.4, 0.5, 2.0, 4.3, 7.2, 7.6, 10.2, 11.4, 11.7]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([4.5, 4.6, 5.2, 5.8, 6.1, 6.4, 6.7, 7.2, 7.5, 8.1]),
           st.sampled_from([1.0, 1.7, 2.4, 3.3, 3.7, 4.2, 4.4, 4.7, 4.8, 4.9]),
           st.floats(min_value=1.451, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.34, allow_nan=False),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([5.5, 6.1, 6.3, 6.4, 6.5, 7.2, 7.6, 7.8, 8.4, 9.2]),
           st.sampled_from([2.2, 2.4, 2.6, 2.9, 3.3, 3.7, 4.4, 4.5, 4.9, 5.0]),
           st.floats(min_value=0.3, max_value=0.559, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.floats(min_value=0.37, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([4.4, 5.2, 5.3, 5.6, 6.0, 6.6, 7.1, 7.9, 8.3, 9.2]),
           st.sampled_from([2.1, 2.8, 3.0, 3.1, 3.2, 3.6, 3.8, 4.0, 4.3, 4.5]),
           st.floats(min_value=0.3, max_value=0.559, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.04, allow_nan=False),
           st.sampled_from([2.4, 3.9, 4.5, 7.0, 7.2, 7.7, 11.3, 11.8, 13.7, 17.1]),
           st.floats(min_value=211.51, max_value=226.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=24.49, allow_nan=False),
           st.sampled_from([3.0, 5.1, 5.4, 5.5, 5.9, 6.0, 6.3, 6.9, 7.2, 7.6]),
           st.sampled_from([1.6, 1.7, 2.0, 2.1, 2.5, 2.9, 3.0, 3.4, 3.8, 4.2]),
           st.floats(min_value=0.562, max_value=1.149, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.04, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.3, 0.9, 1.0, 1.1, 1.2, 2.3, 3.0, 3.6]),
           st.floats(min_value=211.51, max_value=226.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.51, max_value=88.99, exclude_min=True, allow_nan=False),
           st.sampled_from([3.8, 3.9, 4.6, 4.9, 5.3, 6.3, 7.0, 7.1, 7.2, 7.5]),
           st.sampled_from([1.6, 2.0, 2.1, 2.2, 2.3, 2.4, 2.9, 3.4, 4.4, 5.0]),
           st.floats(min_value=0.562, max_value=1.149, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=2.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=1.04, allow_nan=False),
           st.sampled_from([2.0, 2.7, 4.6, 5.2, 7.2, 7.8, 8.2, 9.5, 12.1, 17.1]),
           st.floats(min_value=211.51, max_value=226.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([2.7, 4.4, 5.1, 5.4, 6.9, 8.0, 8.1, 8.3, 8.4, 9.6]),
           st.sampled_from([1.6, 2.1, 2.3, 2.9, 3.1, 3.3, 3.4, 4.2, 4.7, 4.8]),
           st.floats(min_value=0.562, max_value=1.149, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.07, max_value=1.43, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.6, 3.2]),
           st.floats(min_value=211.51, max_value=226.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([4.8, 5.0, 5.5, 5.6, 5.8, 6.5, 6.9, 7.8, 8.1, 8.3]),
           st.sampled_from([2.2, 2.3, 2.6, 3.0, 3.2, 3.7, 4.0, 4.1, 4.2, 4.7]),
           st.floats(min_value=0.562, max_value=1.149, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.2, 0.3, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 2.3, 3.0]),
           st.floats(min_value=211.51, max_value=226.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([4.5, 5.0, 5.2, 6.0, 6.4, 6.8, 6.9, 7.5, 7.9, 8.4]),
           st.sampled_from([1.7, 1.8, 2.2, 2.8, 3.0, 3.1, 4.0, 4.3, 4.4, 4.6]),
           st.floats(min_value=1.152, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=19.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.2, 1.1, 3.6, 4.5, 5.6, 7.0, 8.4, 12.1, 12.8, 14.1]),
           st.floats(min_value=226.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([2.8, 4.1, 4.4, 4.6, 6.0, 6.2, 6.9, 7.4, 8.4, 8.9]),
           st.sampled_from([1.0, 2.3, 2.4, 2.6, 3.2, 3.3, 3.7, 3.9, 4.5, 5.5]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=20.01, max_value=27.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.1, 0.3, 0.8, 0.9, 1.1, 1.4, 2.3, 3.0, 3.2, 3.6]),
           st.floats(min_value=226.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([5.5, 5.7, 6.4, 6.6, 6.7, 6.8, 7.2, 7.3, 8.4, 9.2]),
           st.sampled_from([2.4, 2.5, 2.7, 3.6, 3.7, 4.0, 4.1, 4.3, 4.6, 5.0]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=226.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=32.49, allow_nan=False),
           st.sampled_from([3.0, 4.7, 5.6, 6.1, 7.0, 7.7, 7.9, 8.1, 8.2, 9.5]),
           st.sampled_from([1.5, 1.8, 2.3, 2.4, 3.1, 3.5, 3.6, 3.7, 4.0, 4.9]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.floats(min_value=0.1, max_value=0.14, allow_nan=False),
           st.floats(min_value=226.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=32.51, max_value=88.99, exclude_min=True, allow_nan=False),
           st.sampled_from([4.8, 5.1, 5.3, 5.4, 5.9, 6.1, 6.9, 7.0, 7.3, 8.4]),
           st.sampled_from([1.8, 2.2, 2.7, 2.9, 3.2, 3.4, 3.7, 4.0, 4.1, 4.5]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.floats(min_value=0.17, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=226.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=88.99, allow_nan=False),
           st.sampled_from([5.0, 6.6, 6.8, 7.5, 7.6, 7.7, 7.8, 8.2, 8.3, 9.6]),
           st.sampled_from([2.2, 2.8, 3.0, 3.3, 3.6, 4.3, 4.6, 4.7, 4.8, 5.5]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([2.2, 2.3, 2.6, 3.0, 5.1, 6.2, 7.7, 10.2, 12.8, 14.1]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=89.01, max_value=103.99, exclude_min=True, allow_nan=False),
           st.sampled_from([2.8, 4.0, 4.6, 5.7, 6.3, 6.7, 7.0, 7.2, 7.4, 7.9]),
           st.sampled_from([2.6, 2.7, 3.1, 3.4, 3.6, 4.0, 4.5, 4.7, 4.9, 5.5]),
           st.floats(min_value=0.562, max_value=0.949, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=58.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 3.0, 3.2, 3.6]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=89.01, max_value=103.99, exclude_min=True, allow_nan=False),
           st.sampled_from([4.8, 4.9, 5.1, 5.4, 5.7, 6.3, 6.8, 7.1, 7.6, 8.5]),
           st.sampled_from([1.6, 1.7, 2.2, 2.3, 2.4, 3.5, 3.9, 4.2, 4.4, 4.5]),
           st.floats(min_value=0.952, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.01, max_value=65.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([1.8, 2.1, 2.2, 2.3, 2.8, 5.0, 8.9, 10.2, 10.8, 17.1]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=22.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([4.0, 4.9, 5.3, 6.0, 6.6, 7.0, 7.1, 7.3, 8.0, 8.2]),
           st.sampled_from([1.7, 1.8, 2.5, 2.7, 2.8, 3.0, 3.2, 4.4, 4.7, 4.9]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.01, max_value=65.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.1, 1.2, 2.3, 3.2]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([5.1, 5.8, 6.0, 6.2, 6.9, 7.1, 8.1, 8.3, 8.5, 9.2]),
           st.sampled_from([1.9, 2.7, 2.8, 3.0, 3.2, 3.3, 3.7, 4.0, 4.1, 5.0]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([1.0, 3.9, 4.3, 5.1, 5.5, 6.1, 7.6, 8.8, 10.8, 18.3]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=16.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([4.8, 4.9, 5.2, 5.6, 6.0, 7.0, 7.1, 8.2, 9.2, 9.6]),
           st.sampled_from([1.0, 1.9, 2.1, 2.2, 2.7, 3.1, 3.6, 4.0, 4.4, 4.6]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.1, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 1.6, 2.3, 3.2]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=17.01, max_value=18.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.7, 5.0, 5.4, 5.6, 5.9, 6.1, 6.2, 6.3, 6.7, 7.9]),
           st.sampled_from([1.6, 1.8, 1.9, 2.4, 3.1, 3.6, 3.8, 4.0, 4.3, 4.4]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.43, allow_nan=False),
           st.sampled_from([0.8, 1.0, 1.1, 1.4, 1.6, 2.6, 3.2, 3.7, 6.4, 9.0]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.51, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.8, 5.2, 5.6, 6.7, 7.4, 7.9, 8.0, 8.7, 9.2, 9.6]),
           st.sampled_from([1.4, 1.5, 1.6, 2.4, 2.8, 3.1, 3.3, 3.5, 3.7, 5.5]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=17.51, max_value=44.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.46, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 1.1, 1.6, 2.3, 3.2]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([3.8, 4.6, 4.8, 5.0, 5.9, 6.2, 6.5, 6.6, 6.9, 8.1]),
           st.sampled_from([2.5, 2.6, 2.8, 2.9, 3.3, 4.2, 4.3, 4.4, 4.7, 4.9]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.46, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([0.4, 0.9, 2.3, 2.9, 3.7, 3.9, 5.6, 6.1, 8.9, 11.3]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.0, max_value=103.99, allow_nan=False),
           st.sampled_from([2.8, 4.3, 4.4, 5.1, 5.2, 5.3, 5.7, 6.2, 6.7, 7.6]),
           st.sampled_from([1.5, 2.6, 3.1, 3.2, 3.3, 3.7, 4.0, 4.2, 4.3, 4.5]),
           st.floats(min_value=0.562, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=0.4, max_value=1.64, allow_nan=False),
           st.sampled_from([0.3, 1.0, 1.2, 1.5, 1.6, 3.2, 8.5, 9.0, 11.4, 12.1]),
           st.floats(min_value=211.51, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 47.0, 53.0, 64.0, 96.0, 114.0, 159.0, 233.0, 779.0, 2000.0]),
           st.floats(min_value=104.01, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.sampled_from([3.6, 3.8, 4.4, 5.2, 5.9, 7.2, 7.6, 7.8, 8.3, 8.5]),
           st.sampled_from([1.8, 1.9, 2.1, 2.2, 2.4, 2.8, 2.9, 3.8, 3.9, 4.1]),
           st.sampled_from([0.35, 0.46, 0.89, 0.9, 0.93, 1.09, 1.1, 1.11, 1.16, 1.66]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([17.0, 21.0, 26.0, 40.0, 42.0, 43.0, 44.0, 45.0, 54.0, 85.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([154.0, 158.0, 169.0, 182.0, 199.0, 275.0, 348.0, 358.0, 406.0, 592.0]),
           st.sampled_from([21.0, 22.0, 28.0, 46.0, 49.0, 52.0, 56.0, 63.0, 82.0, 160.0]),
           st.sampled_from([15.0, 16.0, 18.0, 21.0, 30.0, 33.0, 35.0, 43.0, 56.0, 84.0]),
           st.floats(min_value=2.7, max_value=3.94, allow_nan=False),
           st.sampled_from([1.9, 2.1, 2.4, 2.5, 2.8, 2.9, 3.0, 4.2, 4.4, 4.5]),
           st.sampled_from([0.37, 0.45, 0.71, 0.9, 0.96, 1.03, 1.18, 1.58, 1.7, 1.9]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=38.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=2.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.04, allow_nan=False),
           st.sampled_from([100.0, 123.0, 174.0, 201.0, 220.0, 236.0, 309.0, 316.0, 332.0, 380.0]),
           st.sampled_from([42.0, 46.0, 52.0, 70.0, 94.0, 110.0, 139.0, 140.0, 213.0, 779.0]),
           st.floats(min_value=10.0, max_value=164.99, allow_nan=False),
           st.floats(min_value=3.97, max_value=8.08, exclude_min=True, allow_nan=False),
           st.sampled_from([1.5, 1.9, 2.5, 2.6, 2.8, 3.4, 3.7, 4.0, 4.3, 4.8]),
           st.sampled_from([0.4, 0.62, 0.69, 0.8, 1.0, 1.06, 1.11, 1.2, 1.36, 1.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=38.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=2.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.04, allow_nan=False),
           st.sampled_from([138.0, 145.0, 161.0, 179.0, 197.0, 206.0, 211.0, 237.0, 302.0, 348.0]),
           st.sampled_from([13.0, 18.0, 33.0, 36.0, 38.0, 42.0, 46.0, 82.0, 88.0, 119.0]),
           st.floats(min_value=165.01, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.97, max_value=8.08, exclude_min=True, allow_nan=False),
           st.sampled_from([1.6, 1.9, 2.3, 2.5, 2.7, 3.0, 3.2, 3.4, 3.8, 4.7]),
           st.floats(min_value=0.3, max_value=1.098, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=38.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=2.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.04, allow_nan=False),
           st.sampled_from([63.0, 185.0, 224.0, 251.0, 331.0, 554.0, 574.0, 580.0, 680.0, 719.0]),
           st.sampled_from([32.0, 46.0, 59.0, 140.0, 141.0, 142.0, 159.0, 173.0, 198.0, 950.0]),
           st.floats(min_value=165.01, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.97, max_value=8.08, exclude_min=True, allow_nan=False),
           st.sampled_from([0.9, 2.0, 2.2, 2.6, 2.8, 3.5, 3.6, 3.9, 4.1, 4.8]),
           st.floats(min_value=1.101, max_value=2.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=38.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=2.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.04, allow_nan=False),
           st.sampled_from([164.0, 166.0, 188.0, 195.0, 199.0, 201.0, 216.0, 218.0, 300.0, 352.0]),
           st.sampled_from([12.0, 17.0, 19.0, 22.0, 27.0, 32.0, 36.0, 55.0, 60.0, 63.0]),
           st.sampled_from([10.0, 14.0, 16.0, 20.0, 23.0, 25.0, 52.0, 71.0, 108.0, 110.0]),
           st.floats(min_value=8.11, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([2.3, 2.5, 2.7, 2.9, 3.4, 3.5, 3.7, 4.4, 4.5, 4.9]),
           st.sampled_from([0.45, 0.6, 0.67, 0.7, 0.76, 0.8, 0.96, 1.0, 1.3, 1.85]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=38.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=2.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.07, max_value=1.24, exclude_min=True, allow_nan=False),
           st.sampled_from([152.0, 163.0, 173.0, 176.0, 192.0, 211.0, 226.0, 243.0, 289.0, 630.0]),
           st.sampled_from([13.0, 21.0, 32.0, 35.0, 37.0, 45.0, 52.0, 56.0, 61.0, 181.0]),
           st.sampled_from([19.0, 20.0, 30.0, 34.0, 36.0, 44.0, 56.0, 58.0, 74.0, 285.0]),
           st.floats(min_value=3.97, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.7, 1.8, 2.6, 2.8, 2.9, 3.3, 4.0, 4.1, 4.2, 4.4]),
           st.sampled_from([0.5, 0.52, 0.58, 0.67, 0.7, 0.76, 0.8, 1.06, 1.85, 1.9]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=22.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=2.16, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([180.0, 189.0, 216.0, 250.0, 466.0, 599.0, 805.0, 859.0, 962.0, 1100.0]),
           st.sampled_from([22.0, 25.0, 30.0, 61.0, 67.0, 69.0, 89.0, 118.0, 141.0, 779.0]),
           st.floats(min_value=10.0, max_value=108.49, allow_nan=False),
           st.floats(min_value=3.97, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.9, 1.4, 1.5, 1.9, 2.5, 3.1, 3.5, 3.8, 4.3, 4.6]),
           st.sampled_from([0.52, 0.62, 0.68, 0.78, 0.88, 1.09, 1.36, 1.6, 1.72, 2.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.0, max_value=22.49, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=2.16, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([140.0, 161.0, 178.0, 186.0, 201.0, 202.0, 208.0, 210.0, 352.0, 460.0]),
           st.sampled_from([11.0, 12.0, 19.0, 24.0, 26.0, 30.0, 36.0, 46.0, 61.0, 152.0]),
           st.floats(min_value=108.51, max_value=4929.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.97, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.4, 2.4, 2.5, 2.6, 2.9, 3.6, 3.8, 4.3, 4.7, 5.0]),
           st.sampled_from([0.37, 0.96, 1.16, 1.18, 1.3, 1.38, 1.4, 1.5, 1.7, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=22.51, max_value=38.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=2.16, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([155.0, 158.0, 185.0, 206.0, 218.0, 275.0, 352.0, 358.0, 630.0, 661.0]),
           st.sampled_from([10.0, 12.0, 13.0, 14.0, 18.0, 36.0, 41.0, 52.0, 55.0, 63.0]),
           st.sampled_from([13.0, 14.0, 19.0, 23.0, 24.0, 32.0, 40.0, 43.0, 59.0, 135.0]),
           st.floats(min_value=3.97, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.4, 1.9, 2.0, 2.3, 2.4, 3.3, 4.2, 4.3, 4.6, 4.9]),
           st.sampled_from([0.37, 0.45, 0.52, 0.67, 0.71, 0.76, 0.9, 0.95, 1.3, 1.85]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([143.0, 156.0, 163.0, 187.0, 198.0, 245.0, 285.0, 395.0, 405.0, 670.0]),
           st.sampled_from([20.0, 32.0, 35.0, 71.0, 78.0, 99.0, 140.0, 142.0, 157.0, 220.0]),
           st.sampled_from([14.0, 26.0, 32.0, 55.0, 152.0, 181.0, 200.0, 250.0, 630.0, 960.0]),
           st.floats(min_value=3.97, max_value=6.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9, max_value=2.84, allow_nan=False),
           st.sampled_from([0.35, 0.55, 0.64, 0.75, 0.88, 0.89, 1.1, 1.11, 1.58, 1.7]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=64.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([105.0, 114.0, 135.0, 156.0, 173.0, 178.0, 218.0, 230.0, 289.0, 302.0]),
           st.sampled_from([11.0, 14.0, 19.0, 20.0, 23.0, 29.0, 47.0, 59.0, 63.0, 160.0]),
           st.sampled_from([25.0, 29.0, 34.0, 45.0, 47.0, 57.0, 85.0, 90.0, 103.0, 127.0]),
           st.floats(min_value=3.97, max_value=6.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.87, max_value=3.59, exclude_min=True, allow_nan=False),
           st.sampled_from([0.67, 0.76, 0.8, 0.92, 0.95, 0.96, 1.18, 1.2, 1.58, 1.85]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=64.99, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([152.0, 162.0, 216.0, 232.0, 386.0, 542.0, 690.0, 699.0, 802.0, 901.0]),
           st.sampled_from([30.0, 33.0, 60.0, 79.0, 89.0, 94.0, 141.0, 142.0, 189.0, 482.0]),
           st.sampled_from([12.0, 35.0, 39.0, 51.0, 91.0, 97.0, 125.0, 127.0, 441.0, 623.0]),
           st.floats(min_value=3.97, max_value=6.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.62, max_value=5.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.6, 0.61, 0.69, 0.7, 1.1, 1.16, 1.25, 1.3, 1.51, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.01, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([187.0, 256.0, 272.0, 275.0, 498.0, 599.0, 690.0, 901.0, 1050.0, 1110.0]),
           st.sampled_from([15.0, 33.0, 56.0, 78.0, 97.0, 142.0, 148.0, 166.0, 390.0, 1680.0]),
           st.sampled_from([28.0, 47.0, 49.0, 54.0, 55.0, 80.0, 82.0, 152.0, 190.0, 602.0]),
           st.floats(min_value=3.97, max_value=6.13, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.87, max_value=5.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.4, 0.64, 0.69, 0.93, 0.95, 0.97, 1.02, 1.12, 1.34, 2.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1, max_value=1.24, allow_nan=False),
           st.sampled_from([173.0, 181.0, 194.0, 198.0, 218.0, 285.0, 462.0, 515.0, 610.0, 612.0]),
           st.sampled_from([46.0, 64.0, 99.0, 114.0, 116.0, 118.0, 189.0, 194.0, 622.0, 779.0]),
           st.sampled_from([57.0, 74.0, 88.0, 97.0, 99.0, 134.0, 188.0, 330.0, 441.0, 630.0]),
           st.floats(min_value=6.16, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.6, 2.1, 2.2, 2.6, 3.1, 3.5, 3.8, 4.1, 4.3, 4.5]),
           st.sampled_from([0.3, 0.39, 0.4, 0.53, 0.55, 0.64, 0.7, 0.75, 1.02, 1.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([17.0, 18.0, 24.0, 25.0, 28.0, 35.0, 38.0, 53.0, 55.0, 58.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.sampled_from([105.0, 191.0, 194.0, 205.0, 209.0, 216.0, 218.0, 237.0, 247.0, 310.0]),
           st.floats(min_value=10.0, max_value=11.99, allow_nan=False),
           st.sampled_from([15.0, 32.0, 45.0, 51.0, 53.0, 58.0, 74.0, 82.0, 103.0, 111.0]),
           st.sampled_from([5.0, 5.5, 5.7, 6.1, 6.3, 6.9, 7.4, 7.5, 7.9, 9.2]),
           st.sampled_from([1.9, 2.4, 2.7, 2.9, 3.2, 4.3, 4.6, 4.7, 4.9, 5.0]),
           st.sampled_from([0.37, 0.45, 0.5, 0.58, 0.71, 0.75, 0.76, 1.38, 1.4, 1.58]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([12.0, 14.0, 15.0, 16.0, 27.0, 28.0, 35.0, 53.0, 68.0, 72.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=6.04, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=146.99, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([33.0, 36.0, 92.0, 103.0, 105.0, 155.0, 181.0, 186.0, 231.0, 576.0]),
           st.floats(min_value=2.7, max_value=5.03, allow_nan=False),
           st.sampled_from([1.0, 1.6, 2.4, 2.9, 3.2, 3.3, 3.5, 3.7, 4.6, 5.5]),
           st.sampled_from([0.3, 0.35, 0.39, 0.5, 0.62, 1.06, 1.25, 1.55, 1.7, 2.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_104(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_104']['n_samples'] += 1
        self.data['tests']['test_104']['samples'].append(x_test)
        self.data['tests']['test_104']['y_expected'].append(y_expected[0])
        self.data['tests']['test_104']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([20.0, 23.0, 36.0, 38.0, 40.0, 47.0, 53.0, 64.0, 65.0, 69.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=6.04, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=146.99, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 20.0, 22.0, 33.0, 34.0, 39.0, 52.0, 67.0, 135.0, 178.0]),
           st.floats(min_value=5.06, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.8, 2.0, 2.2, 2.4, 3.3, 3.4, 3.9, 4.1, 4.9, 5.0]),
           st.sampled_from([0.37, 0.75, 0.76, 0.96, 1.1, 1.16, 1.18, 1.5, 1.7, 1.85]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([20.0, 23.0, 31.0, 38.0, 48.0, 49.0, 58.0, 60.0, 64.0, 73.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=6.07, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.0, max_value=146.99, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([41.0, 42.0, 46.0, 48.0, 58.0, 73.0, 91.0, 95.0, 116.0, 220.0]),
           st.sampled_from([5.4, 5.6, 6.1, 6.9, 7.5, 7.7, 8.1, 8.3, 8.7, 8.9]),
           st.sampled_from([1.4, 1.8, 1.9, 2.8, 2.9, 3.0, 3.6, 4.1, 4.8, 4.9]),
           st.sampled_from([0.52, 0.53, 0.6, 0.62, 0.7, 0.9, 1.0, 1.38, 1.5, 1.6]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_106(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_106']['n_samples'] += 1
        self.data['tests']['test_106']['samples'].append(x_test)
        self.data['tests']['test_106']['y_expected'].append(y_expected[0])
        self.data['tests']['test_106']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([19.0, 35.0, 37.0, 40.0, 54.0, 62.0, 63.0, 73.0, 74.0, 75.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=147.01, max_value=620.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 55.0, 67.0, 68.0, 82.0, 87.0, 88.0, 168.0, 236.0, 794.0]),
           st.sampled_from([4.7, 4.9, 5.2, 6.1, 6.3, 7.0, 7.1, 7.8, 8.2, 8.3]),
           st.sampled_from([1.5, 2.1, 2.3, 2.4, 2.8, 2.9, 3.7, 3.9, 4.4, 4.8]),
           st.sampled_from([0.39, 0.52, 0.7, 0.88, 0.89, 0.93, 1.3, 1.34, 1.8, 2.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([16.0, 22.0, 24.0, 49.0, 57.0, 61.0, 63.0, 65.0, 70.0, 72.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=621.01, max_value=665.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 18.0, 23.0, 47.0, 52.0, 74.0, 82.0, 84.0, 148.0, 231.0]),
           st.sampled_from([4.5, 4.8, 5.1, 5.7, 5.9, 6.1, 6.2, 6.8, 6.9, 7.8]),
           st.sampled_from([1.8, 2.1, 2.3, 2.4, 3.0, 3.2, 3.3, 3.8, 4.1, 4.7]),
           st.sampled_from([0.37, 0.5, 0.7, 0.71, 0.8, 1.0, 1.18, 1.3, 1.38, 1.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_108(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_108']['n_samples'] += 1
        self.data['tests']['test_108']['samples'].append(x_test)
        self.data['tests']['test_108']['y_expected'].append(y_expected[0])
        self.data['tests']['test_108']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.0, 27.0, 33.0, 37.0, 43.0, 49.0, 60.0, 63.0, 66.0, 73.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=665.51, max_value=1564.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([11.0, 33.0, 35.0, 83.0, 99.0, 114.0, 168.0, 187.0, 221.0, 844.0]),
           st.sampled_from([4.1, 5.1, 5.5, 5.7, 6.4, 6.8, 6.9, 8.4, 8.5, 8.7]),
           st.sampled_from([1.0, 1.5, 1.8, 2.2, 2.9, 3.0, 3.6, 4.5, 4.9, 5.5]),
           st.sampled_from([0.3, 0.64, 0.74, 0.78, 0.8, 0.89, 1.0, 1.1, 1.12, 1.8]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_109(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_109']['n_samples'] += 1
        self.data['tests']['test_109']['samples'].append(x_test)
        self.data['tests']['test_109']['y_expected'].append(y_expected[0])
        self.data['tests']['test_109']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.0, 17.0, 19.0, 21.0, 38.0, 39.0, 40.0, 54.0, 57.0, 70.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1565.01, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 20.0, 26.0, 27.0, 32.0, 54.0, 92.0, 108.0, 110.0, 231.0]),
           st.floats(min_value=2.7, max_value=5.88, allow_nan=False),
           st.sampled_from([1.6, 1.7, 2.0, 2.1, 2.2, 2.7, 2.8, 2.9, 3.4, 3.6]),
           st.sampled_from([0.45, 0.7, 0.76, 0.8, 0.9, 0.92, 0.95, 1.1, 1.16, 1.5]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.0, 16.0, 17.0, 36.0, 52.0, 53.0, 62.0, 65.0, 66.0, 78.0]),
           st.sampled_from([1.0, 2.0]),
           st.floats(min_value=1.67, max_value=75.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.26, max_value=19.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1565.01, max_value=2110.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.01, max_value=2000.0, exclude_min=True, allow_nan=False),
           st.sampled_from([34.0, 38.0, 46.0, 47.0, 70.0, 90.0, 95.0, 114.0, 125.0, 126.0]),
           st.floats(min_value=5.91, max_value=9.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.9, 1.5, 2.1, 2.6, 2.7, 3.0, 3.3, 3.7, 4.7, 5.5]),
           st.sampled_from([0.3, 0.6, 0.7, 0.75, 0.8, 0.87, 1.02, 1.1, 1.3, 1.38]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_111(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_111']['n_samples'] += 1
        self.data['tests']['test_111']['samples'].append(x_test)
        self.data['tests']['test_111']['y_expected'].append(y_expected[0])
        self.data['tests']['test_111']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
