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
    request.cls.data['n_test'] = 57
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_cleveland_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestClevelandProperty:

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=106.0, max_value=108.99, allow_nan=False),
           st.sampled_from([157.0, 182.0, 186.0, 196.0, 210.0, 246.0, 264.0, 270.0, 302.0, 318.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 130.0, 151.0, 159.0, 162.0, 171.0, 185.0, 188.0, 192.0, 202.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.35, max_value=1.68, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=106.0, max_value=108.99, allow_nan=False),
           st.sampled_from([169.0, 187.0, 230.0, 246.0, 248.0, 254.0, 267.0, 283.0, 299.0, 353.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([124.0, 127.0, 133.0, 138.0, 140.0, 142.0, 147.0, 157.0, 162.0, 171.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.48, max_value=0.59, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
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

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=106.0, max_value=108.99, allow_nan=False),
           st.sampled_from([199.0, 201.0, 213.0, 219.0, 239.0, 253.0, 263.0, 273.0, 283.0, 315.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([96.0, 111.0, 115.0, 142.0, 147.0, 168.0, 170.0, 174.0, 181.0, 188.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.62, max_value=0.83, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=109.01, max_value=118.6, exclude_min=True, allow_nan=False),
           st.sampled_from([182.0, 196.0, 210.0, 235.0, 243.0, 246.0, 257.0, 260.0, 269.0, 321.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([114.0, 130.0, 142.0, 150.0, 167.0, 169.0, 179.0, 185.0, 186.0, 202.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.35, max_value=1.68, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([131.0, 188.0, 205.0, 207.0, 230.0, 236.0, 267.0, 275.0, 305.0, 341.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([90.0, 111.0, 117.0, 130.0, 136.0, 144.0, 145.0, 163.0, 174.0, 181.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.71, max_value=2.07, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.51, max_value=2.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([201.0, 209.0, 215.0, 246.0, 254.0, 262.0, 264.0, 265.0, 273.0, 295.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 126.0, 131.0, 137.0, 148.0, 149.0, 155.0, 165.0, 169.0, 171.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.71, max_value=2.07, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.4, max_value=59.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([198.0, 219.0, 228.0, 232.0, 269.0, 273.0, 284.0, 290.0, 318.0, 341.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([88.0, 90.0, 105.0, 108.0, 109.0, 114.0, 134.0, 136.0, 142.0, 158.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.56, max_value=4.08, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([149.0, 172.0, 177.0, 185.0, 187.0, 228.0, 264.0, 284.0, 286.0, 311.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=81.0, max_value=83.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.6, 0.8, 2.0, 2.2, 2.4, 2.5, 2.8, 3.1, 3.2, 5.6]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
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

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([168.0, 175.0, 180.0, 199.0, 204.0, 255.0, 258.0, 260.0, 262.0, 269.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=83.51, max_value=97.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.sampled_from([0.0, 0.2, 0.3, 0.7, 0.9, 1.1, 1.4, 2.3, 3.5, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([149.0, 177.0, 187.0, 219.0, 239.0, 267.0, 275.0, 283.0, 289.0, 304.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=83.51, max_value=97.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
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

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([186.0, 192.0, 196.0, 198.0, 215.0, 219.0, 221.0, 265.0, 273.0, 321.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=83.51, max_value=97.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.64, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
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

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.sampled_from([149.0, 205.0, 230.0, 261.0, 264.0, 286.0, 288.0, 307.0, 330.0, 407.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=152.51, max_value=153.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.8, 1.5, 1.6, 1.9, 2.6, 2.8, 3.2, 3.4, 3.8]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
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

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.floats(min_value=258.0, max_value=290.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=156.01, max_value=165.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.4, 1.0, 1.3, 1.4, 1.5, 1.8, 2.0, 3.0, 3.5]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.floats(min_value=291.01, max_value=345.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=156.01, max_value=165.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.1, 1.6, 2.6, 2.8, 2.9, 3.0, 3.1, 3.4, 6.2]),
           st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=63.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=144.4, max_value=156.99, allow_nan=False),
           st.floats(min_value=291.01, max_value=345.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=156.01, max_value=165.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.2, 0.3, 0.9, 1.1, 1.2, 1.6, 1.9, 2.4, 3.5, 4.2]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=55.4, max_value=61.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=157.01, max_value=165.6, exclude_min=True, allow_nan=False),
           st.sampled_from([207.0, 236.0, 254.0, 258.0, 260.0, 289.0, 300.0, 318.0, 319.0, 409.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([111.0, 114.0, 115.0, 122.0, 125.0, 133.0, 154.0, 160.0, 181.0, 182.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.5, 0.6, 1.0, 1.4, 1.5, 1.6, 1.8, 2.8, 3.6, 5.6]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=62.01, max_value=65.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([1.0, 2.0, 3.0, 4.0]),
           st.floats(min_value=157.01, max_value=165.6, exclude_min=True, allow_nan=False),
           st.sampled_from([186.0, 197.0, 210.0, 246.0, 294.0, 304.0, 325.0, 342.0, 354.0, 564.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 111.0, 114.0, 115.0, 151.0, 162.0, 173.0, 174.0, 185.0, 190.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.1, 0.3, 0.4, 0.5, 0.9, 1.1, 1.9, 2.0, 3.0]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([39.0, 41.0, 47.0, 52.0, 54.0, 55.0, 58.0, 63.0, 65.0, 69.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([100.0, 112.0, 122.0, 123.0, 130.0, 135.0, 138.0, 144.0, 150.0, 174.0]),
           st.floats(min_value=187.2, max_value=202.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=127.8, max_value=141.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.1, 0.3, 1.2, 1.6, 1.9, 2.2, 2.8, 2.9, 4.0, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.0, 45.0, 49.0, 52.0, 56.0, 59.0, 60.0, 61.0, 62.0, 66.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([102.0, 105.0, 108.0, 120.0, 122.0, 124.0, 125.0, 129.0, 135.0, 140.0]),
           st.floats(min_value=187.2, max_value=202.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=142.01, max_value=154.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.5, 0.8, 0.9, 1.4, 1.5, 1.9, 2.3, 2.6, 3.0, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.0, 35.0, 43.0, 44.0, 47.0, 50.0, 56.0, 67.0, 71.0, 74.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([100.0, 108.0, 122.0, 128.0, 136.0, 138.0, 145.0, 156.0, 170.0, 172.0]),
           st.floats(min_value=202.51, max_value=209.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=124.6, max_value=137.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.2, 0.3, 0.7, 0.9, 1.0, 1.4, 1.6, 1.8, 3.5, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([35.0, 40.0, 48.0, 50.0, 52.0, 57.0, 58.0, 63.0, 68.0, 69.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([100.0, 108.0, 112.0, 114.0, 120.0, 122.0, 126.0, 146.0, 160.0, 180.0]),
           st.floats(min_value=202.51, max_value=209.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=138.01, max_value=150.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.2, 0.5, 1.0, 1.4, 1.5, 1.9, 2.0, 2.2, 3.8, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([29.0, 40.0, 47.0, 48.0, 50.0, 52.0, 58.0, 68.0, 70.0, 76.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False),
           st.sampled_from([94.0, 102.0, 106.0, 110.0, 112.0, 129.0, 135.0, 138.0, 152.0, 170.0]),
           st.floats(min_value=237.51, max_value=302.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([126.0, 148.0, 150.0, 151.0, 153.0, 163.0, 178.0, 180.0, 181.0, 194.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.1, 0.4, 0.5, 1.1, 1.2, 1.3, 1.9, 2.6, 3.0, 3.5]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([35.0, 41.0, 42.0, 46.0, 49.0, 51.0, 54.0, 55.0, 59.0, 60.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False),
           st.sampled_from([114.0, 117.0, 123.0, 124.0, 128.0, 134.0, 138.0, 142.0, 154.0, 164.0]),
           st.floats(min_value=237.51, max_value=302.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([103.0, 116.0, 117.0, 134.0, 138.0, 139.0, 144.0, 165.0, 170.0, 177.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.sampled_from([1.6, 1.8, 2.2, 2.4, 2.6, 2.9, 3.1, 3.4, 4.0, 6.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([37.0, 41.0, 48.0, 54.0, 55.0, 62.0, 63.0, 67.0, 71.0, 76.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False),
           st.sampled_from([94.0, 104.0, 122.0, 126.0, 128.0, 134.0, 145.0, 146.0, 152.0, 155.0]),
           st.floats(min_value=237.51, max_value=302.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([123.0, 125.0, 126.0, 137.0, 148.0, 163.0, 170.0, 179.0, 180.0, 202.0]),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 1.3, 2.6, 3.5, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([37.0, 42.0, 45.0, 47.0, 53.0, 54.0, 57.0, 63.0, 64.0, 71.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=2.51, max_value=2.7, exclude_min=True, allow_nan=False),
           st.sampled_from([105.0, 106.0, 125.0, 129.0, 130.0, 135.0, 138.0, 156.0, 178.0, 180.0]),
           st.floats(min_value=237.51, max_value=302.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([96.0, 115.0, 138.0, 157.0, 161.0, 165.0, 169.0, 171.0, 174.0, 190.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.3, 0.4, 0.9, 1.2, 1.6, 1.9, 2.4, 3.0, 3.5, 4.2]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([29.0, 40.0, 42.0, 51.0, 55.0, 56.0, 60.0, 62.0, 64.0, 71.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([100.0, 104.0, 108.0, 112.0, 118.0, 120.0, 132.0, 148.0, 152.0, 178.0]),
           st.sampled_from([168.0, 195.0, 208.0, 219.0, 221.0, 248.0, 256.0, 265.0, 306.0, 340.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([111.0, 114.0, 131.0, 140.0, 145.0, 148.0, 163.0, 173.0, 190.0, 192.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.2, 0.5, 1.1, 1.6, 1.8, 2.0, 2.4, 3.0, 4.2]),
           st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([39.0, 45.0, 46.0, 49.0, 50.0, 61.0, 62.0, 64.0, 68.0, 70.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([117.0, 118.0, 124.0, 128.0, 144.0, 152.0, 160.0, 164.0, 165.0, 174.0]),
           st.floats(min_value=264.0, max_value=298.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([95.0, 97.0, 105.0, 125.0, 133.0, 157.0, 162.0, 163.0, 177.0, 181.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.4, 2.0, 2.4, 2.6, 3.0, 3.1, 3.2, 3.8, 4.4, 5.6]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([29.0, 39.0, 41.0, 44.0, 50.0, 60.0, 61.0, 64.0, 68.0, 69.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([101.0, 102.0, 105.0, 112.0, 125.0, 135.0, 156.0, 160.0, 178.0, 180.0]),
           st.floats(min_value=298.51, max_value=351.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 123.0, 131.0, 144.0, 145.0, 146.0, 153.0, 162.0, 180.0, 202.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.1, 0.6, 1.0, 1.1, 1.4, 1.6, 1.8, 2.4, 4.2]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([42.0, 44.0, 47.0, 48.0, 49.0, 56.0, 58.0, 64.0, 67.0, 77.0]),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([112.0, 117.0, 120.0, 122.0, 125.0, 128.0, 144.0, 152.0, 158.0, 174.0]),
           st.sampled_from([177.0, 205.0, 233.0, 236.0, 246.0, 264.0, 268.0, 284.0, 294.0, 319.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([88.0, 96.0, 97.0, 106.0, 116.0, 120.0, 129.0, 143.0, 163.0, 171.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.4, 0.6, 1.6, 1.8, 2.2, 2.8, 3.1, 4.2, 4.4]),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([35.0, 41.0, 43.0, 54.0, 58.0, 63.0, 64.0, 68.0, 71.0, 76.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([100.0, 104.0, 115.0, 120.0, 126.0, 135.0, 140.0, 146.0, 148.0, 160.0]),
           st.sampled_from([178.0, 192.0, 197.0, 214.0, 222.0, 232.0, 253.0, 255.0, 321.0, 417.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([115.0, 131.0, 149.0, 150.0, 157.0, 163.0, 167.0, 179.0, 184.0, 187.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.08, max_value=0.09, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([38.0, 41.0, 46.0, 51.0, 57.0, 59.0, 63.0, 67.0, 69.0, 70.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([110.0, 117.0, 125.0, 132.0, 138.0, 140.0, 142.0, 154.0, 160.0, 165.0]),
           st.sampled_from([200.0, 253.0, 255.0, 260.0, 267.0, 284.0, 294.0, 318.0, 327.0, 409.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=120.2, max_value=132.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.12, max_value=1.33, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=37.0, max_value=38.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.sampled_from([174.0, 187.0, 225.0, 248.0, 249.0, 253.0, 256.0, 284.0, 309.0, 322.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.12, max_value=1.33, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.sampled_from([177.0, 195.0, 208.0, 209.0, 211.0, 265.0, 267.0, 270.0, 340.0, 360.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.12, max_value=0.16, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.sampled_from([149.0, 164.0, 166.0, 193.0, 198.0, 225.0, 241.0, 248.0, 294.0, 315.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.12, max_value=0.16, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.floats(min_value=209.6, max_value=230.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.37, max_value=1.53, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.51, max_value=1.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.floats(min_value=189.6, max_value=205.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.37, max_value=1.53, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.51, max_value=1.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.floats(min_value=205.51, max_value=210.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.37, max_value=1.53, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.01, max_value=46.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=162.0, max_value=178.99, allow_nan=False),
           st.floats(min_value=230.51, max_value=297.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.37, max_value=1.53, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([40.0, 41.0, 42.0, 48.0, 60.0, 62.0, 64.0, 66.0, 70.0, 77.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=179.01, max_value=183.2, exclude_min=True, allow_nan=False),
           st.sampled_from([172.0, 197.0, 233.0, 237.0, 248.0, 259.0, 261.0, 298.0, 318.0, 341.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=132.51, max_value=146.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.12, max_value=1.33, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.0, 40.0, 42.0, 43.0, 49.0, 52.0, 62.0, 68.0, 71.0, 74.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([94.0, 100.0, 102.0, 122.0, 128.0, 129.0, 130.0, 134.0, 142.0, 146.0]),
           st.sampled_from([199.0, 209.0, 226.0, 232.0, 236.0, 239.0, 245.0, 252.0, 262.0, 313.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.8, max_value=0.99, allow_nan=False),
           st.sampled_from([126.0, 138.0, 139.0, 151.0, 153.0, 154.0, 161.0, 162.0, 178.0, 179.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.3, 0.5, 1.0, 1.1, 1.3, 1.6, 2.4, 3.0, 3.5, 4.2]),
           st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([35.0, 38.0, 39.0, 43.0, 46.0, 49.0, 51.0, 56.0, 70.0, 77.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([108.0, 117.0, 128.0, 135.0, 140.0, 146.0, 148.0, 158.0, 164.0, 170.0]),
           st.sampled_from([177.0, 224.0, 248.0, 256.0, 275.0, 283.0, 288.0, 318.0, 330.0, 407.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=1.01, max_value=1.2, exclude_min=True, allow_nan=False),
           st.sampled_from([103.0, 106.0, 108.0, 113.0, 130.0, 136.0, 139.0, 158.0, 171.0, 177.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.1, 1.4, 1.8, 1.9, 2.0, 3.1, 3.8, 4.2, 4.4, 6.2]),
           st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([42.0, 44.0, 46.0, 47.0, 60.0, 62.0, 64.0, 67.0, 70.0, 77.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([118.0, 120.0, 124.0, 125.0, 126.0, 134.0, 140.0, 144.0, 148.0, 178.0]),
           st.sampled_from([177.0, 203.0, 204.0, 207.0, 217.0, 229.0, 274.0, 299.0, 305.0, 311.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([111.0, 115.0, 126.0, 136.0, 138.0, 152.0, 155.0, 165.0, 169.0, 173.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 0.3, 0.8, 0.9, 2.1, 2.4, 2.6, 3.1, 3.8, 4.2]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([39.0, 43.0, 51.0, 55.0, 57.0, 61.0, 62.0, 66.0, 69.0, 77.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=118.8, max_value=124.99, allow_nan=False),
           st.sampled_from([174.0, 184.0, 188.0, 204.0, 207.0, 244.0, 266.0, 275.0, 309.0, 335.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([90.0, 95.0, 97.0, 105.0, 120.0, 123.0, 124.0, 153.0, 154.0, 170.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.1, 0.2, 1.0, 1.6, 2.2, 2.4, 2.9, 3.2, 3.6, 3.8]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.01, max_value=2.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([40.0, 43.0, 44.0, 54.0, 56.0, 57.0, 60.0, 68.0, 69.0, 76.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=125.01, max_value=127.0, exclude_min=True, allow_nan=False),
           st.sampled_from([195.0, 231.0, 239.0, 266.0, 288.0, 298.0, 303.0, 306.0, 318.0, 394.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([115.0, 146.0, 148.0, 155.0, 157.0, 161.0, 165.0, 173.0, 182.0, 186.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.6, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.3, 2.6, 3.5]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.01, max_value=2.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([38.0, 46.0, 47.0, 48.0, 50.0, 51.0, 58.0, 60.0, 69.0, 70.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=135.01, max_value=148.0, exclude_min=True, allow_nan=False),
           st.sampled_from([216.0, 224.0, 225.0, 233.0, 237.0, 266.0, 269.0, 283.0, 300.0, 409.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([106.0, 120.0, 124.0, 134.0, 147.0, 153.0, 158.0, 163.0, 165.0, 181.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.4, 1.4, 1.8, 2.2, 2.9, 3.0, 3.2, 3.6, 4.4, 5.6]),
           st.floats(min_value=1.51, max_value=1.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.01, max_value=2.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([34.0, 37.0, 40.0, 47.0, 49.0, 51.0, 57.0, 61.0, 62.0, 63.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([100.0, 101.0, 102.0, 129.0, 134.0, 135.0, 136.0, 152.0, 178.0, 180.0]),
           st.floats(min_value=214.8, max_value=236.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=133.4, max_value=148.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=54.2, max_value=60.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([100.0, 112.0, 122.0, 125.0, 132.0, 134.0, 140.0, 154.0, 164.0, 165.0]),
           st.floats(min_value=237.01, max_value=302.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=133.4, max_value=148.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=60.51, max_value=63.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([94.0, 100.0, 101.0, 102.0, 110.0, 124.0, 126.0, 129.0, 130.0, 156.0]),
           st.floats(min_value=237.01, max_value=302.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=133.4, max_value=148.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=39.4, max_value=41.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([112.0, 122.0, 123.0, 125.0, 126.0, 132.0, 144.0, 148.0, 164.0, 170.0]),
           st.sampled_from([167.0, 216.0, 229.0, 233.0, 248.0, 256.0, 269.0, 276.0, 284.0, 309.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=149.01, max_value=159.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.01, max_value=49.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([100.0, 105.0, 112.0, 118.0, 122.0, 126.0, 135.0, 150.0, 160.0, 172.0]),
           st.floats(min_value=228.4, max_value=253.99, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=149.01, max_value=159.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.01, max_value=49.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([108.0, 110.0, 126.0, 132.0, 134.0, 135.0, 160.0, 170.0, 174.0, 178.0]),
           st.floats(min_value=254.01, max_value=316.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=149.01, max_value=159.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([38.0, 41.0, 52.0, 55.0, 56.0, 58.0, 60.0, 61.0, 67.0, 70.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([110.0, 117.0, 120.0, 123.0, 126.0, 135.0, 142.0, 170.0, 180.0, 200.0]),
           st.sampled_from([166.0, 193.0, 197.0, 212.0, 228.0, 255.0, 274.0, 281.0, 288.0, 407.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.floats(min_value=149.01, max_value=159.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.44, max_value=0.54, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=58.2, max_value=65.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=108.4, max_value=111.99, allow_nan=False),
           st.floats(min_value=204.8, max_value=224.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([114.0, 121.0, 142.0, 145.0, 154.0, 171.0, 172.0, 174.0, 180.0, 181.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.57, max_value=1.69, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=58.2, max_value=65.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=108.4, max_value=111.99, allow_nan=False),
           st.floats(min_value=224.51, max_value=292.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 116.0, 123.0, 124.0, 125.0, 139.0, 146.0, 161.0, 162.0, 177.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.57, max_value=1.69, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=58.2, max_value=65.49, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=112.01, max_value=129.6, exclude_min=True, allow_nan=False),
           st.sampled_from([184.0, 205.0, 206.0, 223.0, 229.0, 233.0, 241.0, 246.0, 281.0, 330.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([115.0, 120.0, 126.0, 130.0, 131.0, 133.0, 140.0, 152.0, 154.0, 195.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.57, max_value=1.69, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.51, max_value=67.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([102.0, 104.0, 106.0, 108.0, 132.0, 134.0, 145.0, 152.0, 155.0, 172.0]),
           st.sampled_from([195.0, 197.0, 208.0, 227.0, 232.0, 233.0, 243.0, 273.0, 309.0, 315.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([105.0, 133.0, 139.0, 143.0, 149.0, 152.0, 158.0, 165.0, 168.0, 202.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.57, max_value=1.69, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=0.51, max_value=0.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([38.0, 39.0, 43.0, 44.0, 49.0, 51.0, 53.0, 56.0, 68.0, 77.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=3.51, max_value=3.6, exclude_min=True, allow_nan=False),
           st.sampled_from([108.0, 110.0, 114.0, 117.0, 125.0, 128.0, 134.0, 135.0, 138.0, 164.0]),
           st.sampled_from([169.0, 172.0, 216.0, 230.0, 249.0, 258.0, 286.0, 288.0, 289.0, 353.0]),
           st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0, 1.0, 2.0]),
           st.sampled_from([90.0, 95.0, 99.0, 125.0, 142.0, 145.0, 146.0, 150.0, 173.0, 195.0]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=0.57, max_value=1.69, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0]),
           st.sampled_from([0.0, 1.0, 2.0, 3.0]),
           st.floats(min_value=1.51, max_value=1.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
