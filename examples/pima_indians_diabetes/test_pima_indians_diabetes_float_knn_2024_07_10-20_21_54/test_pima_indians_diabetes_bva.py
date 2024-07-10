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
    request.cls.data['n_test'] = 129
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_pima_indians_diabetes_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestPimaIndiansDiabetesProperty:

    @given(st.floats(min_value=6.0, max_value=7.49, allow_nan=False),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([24.0, 30.0, 46.0, 48.0, 52.0, 61.0, 62.0, 65.0, 82.0, 94.0]),
           st.sampled_from([7.0, 12.0, 19.0, 20.0, 21.0, 39.0, 41.0, 42.0, 45.0, 60.0]),
           st.sampled_from([15.0, 70.0, 71.0, 87.0, 88.0, 120.0, 165.0, 215.0, 228.0, 255.0]),
           st.floats(min_value=24.75, max_value=30.93, allow_nan=False),
           st.floats(min_value=0.5531, max_value=0.6718, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.0, max_value=7.49, allow_nan=False),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([30.0, 52.0, 54.0, 72.0, 80.0, 82.0, 85.0, 86.0, 102.0, 110.0]),
           st.sampled_from([14.0, 15.0, 18.0, 23.0, 29.0, 34.0, 35.0, 40.0, 45.0, 46.0]),
           st.sampled_from([36.0, 70.0, 79.0, 129.0, 145.0, 150.0, 180.0, 207.0, 325.0, 360.0]),
           st.floats(min_value=24.75, max_value=30.93, allow_nan=False),
           st.floats(min_value=0.6721, max_value=0.6749, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.0, max_value=7.49, allow_nan=False),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([44.0, 55.0, 60.0, 62.0, 65.0, 74.0, 75.0, 94.0, 110.0, 122.0]),
           st.sampled_from([10.0, 17.0, 21.0, 24.0, 27.0, 31.0, 36.0, 42.0, 46.0, 47.0]),
           st.sampled_from([44.0, 66.0, 86.0, 115.0, 142.0, 194.0, 196.0, 272.0, 330.0, 545.0]),
           st.floats(min_value=24.75, max_value=30.93, allow_nan=False),
           st.floats(min_value=0.6866, max_value=1.0332, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=9.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([56.0, 58.0, 64.0, 70.0, 72.0, 86.0, 90.0, 104.0, 106.0, 114.0]),
           st.sampled_from([13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 21.0, 38.0, 42.0, 43.0]),
           st.sampled_from([14.0, 135.0, 180.0, 182.0, 210.0, 245.0, 258.0, 293.0, 465.0, 510.0]),
           st.floats(min_value=24.75, max_value=30.93, allow_nan=False),
           st.sampled_from([0.183, 0.212, 0.23, 0.24, 0.51, 0.549, 0.693, 0.867, 1.191, 2.137]),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.floats(min_value=42.4, max_value=52.99, allow_nan=False),
           st.sampled_from([20.0, 21.0, 23.0, 30.0, 34.0, 43.0, 44.0, 49.0, 56.0, 99.0]),
           st.sampled_from([0.0, 58.0, 130.0, 156.0, 160.0, 175.0, 225.0, 293.0, 540.0, 600.0]),
           st.floats(min_value=30.96, max_value=31.25, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 7.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.floats(min_value=42.4, max_value=52.99, allow_nan=False),
           st.sampled_from([21.0, 23.0, 24.0, 27.0, 30.0, 38.0, 44.0, 45.0, 46.0, 49.0]),
           st.floats(min_value=25.2, max_value=31.49, allow_nan=False),
           st.floats(min_value=32.47, max_value=35.05, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.floats(min_value=42.4, max_value=52.99, allow_nan=False),
           st.sampled_from([0.0, 7.0, 11.0, 21.0, 28.0, 42.0, 44.0, 47.0, 50.0, 60.0]),
           st.floats(min_value=31.51, max_value=194.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=32.47, max_value=35.05, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0, 13.0]),
           st.floats(min_value=89.2, max_value=111.49, allow_nan=False),
           st.floats(min_value=53.01, max_value=58.9, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 20.0, 22.0, 24.0, 37.0, 38.0, 41.0, 43.0, 46.0, 50.0]),
           st.floats(min_value=29.2, max_value=36.49, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=111.51, max_value=114.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=53.01, max_value=56.8, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 13.0, 28.0, 32.0, 36.0, 47.0, 48.0, 50.0, 52.0, 54.0]),
           st.floats(min_value=29.2, max_value=36.49, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=111.51, max_value=114.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.01, max_value=74.1, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 10.0, 11.0, 12.0, 13.0, 22.0, 35.0, 41.0, 42.0, 60.0]),
           st.floats(min_value=29.2, max_value=36.49, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=21.8, max_value=21.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 10.0, 11.0, 13.0, 14.0]),
           st.floats(min_value=111.51, max_value=114.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.01, max_value=74.1, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 23.0, 27.0, 29.0, 35.0, 38.0, 40.0, 42.0, 46.0, 51.0]),
           st.floats(min_value=29.2, max_value=36.49, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=22.01, max_value=23.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.floats(min_value=82.51, max_value=90.4, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 14.0, 15.0, 21.0, 22.0, 26.0, 43.0, 44.0, 46.0, 99.0]),
           st.floats(min_value=29.2, max_value=36.49, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.floats(min_value=53.01, max_value=66.8, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 20.0, 22.0, 23.0, 27.0, 32.0, 40.0, 43.0, 50.0, 52.0]),
           st.floats(min_value=36.51, max_value=198.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.416, max_value=0.5004, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 13.0]),
           st.floats(min_value=70.8, max_value=88.49, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([11.0, 13.0, 15.0, 20.0, 22.0, 23.0, 29.0, 39.0, 41.0, 54.0]),
           st.sampled_from([16.0, 55.0, 64.0, 71.0, 122.0, 130.0, 132.0, 180.0, 235.0, 480.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=88.51, max_value=94.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([17.0, 18.0, 20.0, 24.0, 27.0, 28.0, 29.0, 32.0, 46.0, 51.0]),
           st.sampled_from([48.0, 74.0, 135.0, 140.0, 145.0, 192.0, 225.0, 231.0, 325.0, 465.0]),
           st.floats(min_value=30.96, max_value=32.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.5859, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=88.51, max_value=94.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([0.0, 8.0, 15.0, 17.0, 23.0, 28.0, 34.0, 35.0, 42.0, 47.0]),
           st.sampled_from([16.0, 36.0, 37.0, 48.0, 57.0, 61.0, 67.0, 68.0, 180.0, 231.0]),
           st.floats(min_value=36.71, max_value=38.44, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.5859, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=3.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=88.51, max_value=94.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([12.0, 17.0, 20.0, 24.0, 26.0, 30.0, 33.0, 34.0, 37.0, 51.0]),
           st.sampled_from([29.0, 105.0, 114.0, 130.0, 132.0, 144.0, 165.0, 190.0, 370.0, 478.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.5859, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=88.51, max_value=94.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([7.0, 11.0, 19.0, 21.0, 24.0, 28.0, 29.0, 32.0, 46.0, 50.0]),
           st.sampled_from([32.0, 36.0, 56.0, 58.0, 76.0, 110.0, 112.0, 155.0, 196.0, 375.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9271, max_value=1.2256, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
           st.floats(min_value=116.01, max_value=118.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([0.0, 12.0, 23.0, 27.0, 34.0, 38.0, 44.0, 45.0, 47.0, 63.0]),
           st.sampled_from([64.0, 90.0, 155.0, 159.0, 165.0, 194.0, 318.0, 360.0, 370.0, 510.0]),
           st.floats(min_value=30.96, max_value=31.58, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=116.01, max_value=118.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=55.2, max_value=68.99, allow_nan=False),
           st.sampled_from([0.0, 18.0, 26.0, 31.0, 32.0, 33.0, 42.0, 43.0, 46.0, 50.0]),
           st.sampled_from([85.0, 86.0, 125.0, 132.0, 148.0, 231.0, 240.0, 278.0, 285.0, 375.0]),
           st.floats(min_value=34.12, max_value=36.37, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 12.0, 17.0]),
           st.floats(min_value=68.0, max_value=84.99, allow_nan=False),
           st.floats(min_value=69.01, max_value=79.6, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 12.0, 14.0, 20.0, 23.0, 25.0, 26.0, 30.0, 40.0, 41.0]),
           st.sampled_from([120.0, 132.0, 182.0, 185.0, 191.0, 230.0, 277.0, 321.0, 478.0, 495.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0]),
           st.floats(min_value=85.01, max_value=90.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.01, max_value=79.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.4, max_value=5.49, allow_nan=False),
           st.sampled_from([40.0, 60.0, 64.0, 105.0, 145.0, 240.0, 270.0, 275.0, 342.0, 480.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 3.0, 4.0, 7.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
           st.floats(min_value=112.51, max_value=115.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.01, max_value=79.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.4, max_value=5.49, allow_nan=False),
           st.sampled_from([130.0, 144.0, 145.0, 146.0, 200.0, 225.0, 240.0, 321.0, 495.0, 579.0]),
           st.floats(min_value=30.96, max_value=33.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=112.51, max_value=115.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.01, max_value=79.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.4, max_value=5.49, allow_nan=False),
           st.sampled_from([43.0, 44.0, 84.0, 86.0, 155.0, 192.0, 204.0, 210.0, 215.0, 278.0]),
           st.floats(min_value=43.72, max_value=44.05, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=85.01, max_value=93.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.01, max_value=79.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=24.2, exclude_min=True, allow_nan=False),
           st.sampled_from([32.0, 73.0, 88.0, 116.0, 160.0, 192.0, 200.0, 215.0, 310.0, 744.0]),
           st.floats(min_value=30.96, max_value=33.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5007, max_value=0.8845, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0, 14.0, 17.0]),
           st.floats(min_value=95.2, max_value=118.99, allow_nan=False),
           st.sampled_from([48.0, 54.0, 58.0, 64.0, 68.0, 72.0, 78.0, 85.0, 90.0, 94.0]),
           st.sampled_from([7.0, 13.0, 15.0, 27.0, 28.0, 31.0, 32.0, 35.0, 37.0, 41.0]),
           st.sampled_from([29.0, 36.0, 64.0, 122.0, 127.0, 132.0, 168.0, 318.0, 540.0, 846.0]),
           st.floats(min_value=45.41, max_value=49.74, exclude_min=True, allow_nan=False),
           st.sampled_from([0.18, 0.23, 0.278, 0.343, 0.439, 0.452, 0.586, 0.63, 0.741, 1.136]),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=119.01, max_value=120.7, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 44.0, 48.0, 58.0, 62.0, 68.0, 76.0, 78.0, 96.0, 110.0]),
           st.sampled_from([8.0, 11.0, 14.0, 29.0, 30.0, 34.0, 41.0, 43.0, 45.0, 52.0]),
           st.sampled_from([18.0, 56.0, 70.0, 165.0, 176.0, 215.0, 231.0, 272.0, 325.0, 342.0]),
           st.floats(min_value=45.41, max_value=49.74, exclude_min=True, allow_nan=False),
           st.sampled_from([0.144, 0.19, 0.382, 0.401, 0.432, 0.464, 0.678, 0.773, 0.892, 0.93]),
           st.floats(min_value=27.0, max_value=28.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 13.0, 14.0, 15.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([30.0, 58.0, 64.0, 66.0, 78.0, 80.0, 82.0, 86.0, 98.0, 108.0]),
           st.sampled_from([0.0, 15.0, 23.0, 26.0, 28.0, 32.0, 44.0, 46.0, 49.0, 63.0]),
           st.sampled_from([130.0, 190.0, 205.0, 215.0, 225.0, 271.0, 328.0, 474.0, 495.0, 510.0]),
           st.floats(min_value=7.71, max_value=9.63, allow_nan=False),
           st.sampled_from([0.259, 0.293, 0.325, 0.328, 0.331, 0.484, 0.851, 0.905, 0.955, 1.182]),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=102.0, max_value=127.49, allow_nan=False),
           st.sampled_from([50.0, 58.0, 61.0, 64.0, 65.0, 68.0, 70.0, 74.0, 85.0, 96.0]),
           st.sampled_from([18.0, 19.0, 20.0, 24.0, 25.0, 37.0, 40.0, 41.0, 44.0, 54.0]),
           st.sampled_from([18.0, 54.0, 61.0, 76.0, 81.0, 90.0, 128.0, 132.0, 176.0, 485.0]),
           st.floats(min_value=9.66, max_value=12.99, exclude_min=True, allow_nan=False),
           st.sampled_from([0.123, 0.223, 0.251, 0.292, 0.412, 0.485, 0.491, 0.631, 0.874, 0.947]),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 12.0, 13.0, 14.0, 17.0]),
           st.floats(min_value=22.8, max_value=28.49, allow_nan=False),
           st.sampled_from([48.0, 50.0, 56.0, 60.0, 75.0, 86.0, 98.0, 100.0, 108.0, 110.0]),
           st.sampled_from([15.0, 18.0, 25.0, 26.0, 30.0, 35.0, 37.0, 39.0, 44.0, 46.0]),
           st.sampled_from([58.0, 79.0, 155.0, 156.0, 192.0, 200.0, 274.0, 285.0, 304.0, 318.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.sampled_from([0.127, 0.135, 0.261, 0.337, 0.343, 0.403, 0.743, 0.757, 0.926, 1.182]),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 12.0, 14.0]),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([48.0, 50.0, 65.0, 66.0, 68.0, 72.0, 75.0, 94.0, 110.0, 114.0]),
           st.sampled_from([13.0, 14.0, 20.0, 21.0, 25.0, 28.0, 29.0, 30.0, 45.0, 56.0]),
           st.sampled_from([90.0, 144.0, 155.0, 185.0, 200.0, 210.0, 231.0, 250.0, 474.0, 478.0]),
           st.floats(min_value=26.37, max_value=27.57, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.152, max_value=0.1704, allow_nan=False),
           st.floats(min_value=28.51, max_value=31.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 13.0]),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 30.0, 46.0, 56.0, 61.0, 65.0, 86.0, 96.0, 98.0, 108.0]),
           st.sampled_from([12.0, 18.0, 24.0, 34.0, 35.0, 37.0, 38.0, 40.0, 44.0, 48.0]),
           st.sampled_from([55.0, 72.0, 125.0, 148.0, 155.0, 158.0, 204.0, 205.0, 326.0, 680.0]),
           st.floats(min_value=32.41, max_value=39.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.152, max_value=0.1704, allow_nan=False),
           st.floats(min_value=28.51, max_value=31.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 38.0, 58.0, 75.0, 78.0, 80.0, 85.0, 90.0, 92.0, 94.0]),
           st.sampled_from([10.0, 15.0, 20.0, 24.0, 25.0, 35.0, 38.0, 42.0, 46.0, 47.0]),
           st.sampled_from([18.0, 32.0, 52.0, 60.0, 71.0, 95.0, 140.0, 204.0, 235.0, 310.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1707, max_value=0.2957, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=31.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 14.0, 17.0]),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([60.0, 65.0, 70.0, 76.0, 85.0, 88.0, 94.0, 96.0, 104.0, 110.0]),
           st.sampled_from([0.0, 14.0, 19.0, 28.0, 29.0, 32.0, 37.0, 43.0, 51.0, 99.0]),
           st.sampled_from([100.0, 155.0, 205.0, 220.0, 225.0, 249.0, 280.0, 285.0, 360.0, 510.0]),
           st.floats(min_value=26.37, max_value=27.26, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6524, max_value=0.7959, allow_nan=False),
           st.floats(min_value=42.51, max_value=50.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.4, max_value=5.49, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 62.0, 70.0, 72.0, 74.0, 76.0, 85.0, 94.0, 95.0, 100.0]),
           st.sampled_from([8.0, 16.0, 21.0, 27.0, 31.0, 33.0, 37.0, 38.0, 47.0, 52.0]),
           st.sampled_from([18.0, 54.0, 61.0, 66.0, 155.0, 204.0, 235.0, 270.0, 335.0, 440.0]),
           st.floats(min_value=30.87, max_value=38.11, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6524, max_value=0.7959, allow_nan=False),
           st.floats(min_value=42.51, max_value=50.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.4, max_value=5.49, allow_nan=False),
           st.floats(min_value=81.01, max_value=84.7, exclude_min=True, allow_nan=False),
           st.sampled_from([50.0, 52.0, 60.0, 72.0, 80.0, 86.0, 90.0, 96.0, 98.0, 104.0]),
           st.sampled_from([12.0, 14.0, 25.0, 26.0, 29.0, 31.0, 35.0, 42.0, 45.0, 99.0]),
           st.sampled_from([29.0, 74.0, 79.0, 110.0, 155.0, 194.0, 200.0, 325.0, 474.0, 480.0]),
           st.floats(min_value=30.87, max_value=38.11, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6524, max_value=0.7959, allow_nan=False),
           st.floats(min_value=42.51, max_value=50.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.51, max_value=7.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 38.0, 54.0, 62.0, 66.0, 70.0, 74.0, 75.0, 82.0, 92.0]),
           st.sampled_from([12.0, 17.0, 18.0, 22.0, 31.0, 33.0, 38.0, 43.0, 50.0, 52.0]),
           st.sampled_from([16.0, 46.0, 66.0, 71.0, 92.0, 108.0, 125.0, 272.0, 275.0, 278.0]),
           st.floats(min_value=30.87, max_value=38.11, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6524, max_value=0.7959, allow_nan=False),
           st.floats(min_value=42.51, max_value=50.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.99, allow_nan=False),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 64.0, 68.0, 74.0, 76.0, 82.0, 84.0, 85.0, 98.0, 106.0]),
           st.sampled_from([7.0, 13.0, 22.0, 25.0, 29.0, 31.0, 37.0, 43.0, 45.0, 48.0]),
           st.sampled_from([16.0, 40.0, 79.0, 88.0, 90.0, 119.0, 125.0, 168.0, 194.0, 330.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7962, max_value=1.1209, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.01, max_value=5.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([40.0, 50.0, 54.0, 58.0, 72.0, 76.0, 84.0, 92.0, 98.0, 106.0]),
           st.floats(min_value=16.0, max_value=19.99, allow_nan=False),
           st.sampled_from([74.0, 91.0, 114.0, 122.0, 150.0, 160.0, 175.0, 191.0, 245.0, 250.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7962, max_value=0.8292, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.01, max_value=5.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([44.0, 50.0, 55.0, 64.0, 65.0, 66.0, 72.0, 85.0, 86.0, 90.0]),
           st.floats(min_value=16.0, max_value=19.99, allow_nan=False),
           st.sampled_from([18.0, 52.0, 60.0, 66.0, 89.0, 112.0, 178.0, 180.0, 194.0, 270.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9616, max_value=1.2532, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.01, max_value=5.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 54.0, 68.0, 72.0, 76.0, 84.0, 85.0, 92.0, 96.0, 110.0]),
           st.floats(min_value=20.01, max_value=35.8, exclude_min=True, allow_nan=False),
           st.sampled_from([64.0, 127.0, 155.0, 160.0, 210.0, 271.0, 304.0, 325.0, 328.0, 478.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7962, max_value=1.1209, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 13.0, 15.0, 17.0]),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([48.0, 58.0, 60.0, 78.0, 84.0, 85.0, 88.0, 94.0, 102.0, 104.0]),
           st.sampled_from([12.0, 24.0, 26.0, 28.0, 34.0, 37.0, 39.0, 42.0, 46.0, 51.0]),
           st.sampled_from([58.0, 79.0, 145.0, 171.0, 175.0, 176.0, 184.0, 250.0, 328.0, 465.0]),
           st.floats(min_value=26.37, max_value=26.62, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([44.0, 52.0, 54.0, 60.0, 61.0, 72.0, 95.0, 96.0, 98.0, 108.0]),
           st.sampled_from([0.0, 7.0, 14.0, 15.0, 16.0, 18.0, 23.0, 24.0, 41.0, 45.0]),
           st.sampled_from([18.0, 22.0, 41.0, 61.0, 63.0, 75.0, 94.0, 112.0, 148.0, 170.0]),
           st.floats(min_value=27.66, max_value=35.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=28.51, max_value=29.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 3.0, 5.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0, 17.0]),
           st.floats(min_value=99.51, max_value=102.2, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 40.0, 48.0, 68.0, 70.0, 76.0, 90.0, 94.0, 100.0, 102.0]),
           st.sampled_from([14.0, 15.0, 17.0, 20.0, 31.0, 36.0, 38.0, 39.0, 42.0, 48.0]),
           st.sampled_from([91.0, 145.0, 160.0, 180.0, 258.0, 280.0, 293.0, 318.0, 325.0, 495.0]),
           st.floats(min_value=27.66, max_value=35.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=34.51, max_value=34.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.0]),
           st.floats(min_value=99.51, max_value=102.2, exclude_min=True, allow_nan=False),
           st.sampled_from([60.0, 61.0, 72.0, 80.0, 82.0, 84.0, 85.0, 95.0, 110.0, 122.0]),
           st.sampled_from([13.0, 16.0, 23.0, 27.0, 30.0, 32.0, 33.0, 34.0, 37.0, 50.0]),
           st.sampled_from([43.0, 45.0, 50.0, 52.0, 53.0, 60.0, 63.0, 155.0, 182.0, 231.0]),
           st.floats(min_value=27.66, max_value=35.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=36.51, max_value=45.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 17.0]),
           st.floats(min_value=113.01, max_value=115.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.8, max_value=90.99, allow_nan=False),
           st.sampled_from([17.0, 20.0, 24.0, 25.0, 33.0, 34.0, 41.0, 43.0, 47.0, 99.0]),
           st.sampled_from([127.0, 145.0, 192.0, 271.0, 277.0, 304.0, 318.0, 328.0, 392.0, 600.0]),
           st.floats(min_value=27.66, max_value=35.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=34.51, max_value=43.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]),
           st.floats(min_value=113.01, max_value=115.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=91.01, max_value=97.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 7.0, 11.0, 16.0, 18.0, 28.0, 29.0, 32.0, 43.0, 48.0]),
           st.sampled_from([52.0, 71.0, 87.0, 140.0, 165.0, 180.0, 183.0, 215.0, 270.0, 480.0]),
           st.floats(min_value=27.66, max_value=35.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1755, max_value=0.1998, allow_nan=False),
           st.floats(min_value=34.51, max_value=43.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.2, max_value=1.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=102.2, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 46.0, 54.0, 55.0, 65.0, 70.0, 96.0, 100.0, 106.0, 108.0]),
           st.sampled_from([20.0, 24.0, 33.0, 34.0, 35.0, 38.0, 39.0, 45.0, 50.0, 60.0]),
           st.sampled_from([38.0, 57.0, 58.0, 88.0, 122.0, 135.0, 166.0, 231.0, 310.0, 440.0]),
           st.floats(min_value=26.37, max_value=28.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.2, max_value=1.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=102.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 48.0, 58.0, 60.0, 62.0, 66.0, 72.0, 76.0, 86.0, 106.0]),
           st.sampled_from([0.0, 18.0, 19.0, 22.0, 25.0, 35.0, 38.0, 42.0, 43.0, 45.0]),
           st.sampled_from([36.0, 159.0, 171.0, 184.0, 194.0, 207.0, 271.0, 370.0, 540.0, 579.0]),
           st.floats(min_value=36.16, max_value=42.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.2, max_value=1.49, allow_nan=False),
           st.floats(min_value=113.01, max_value=115.9, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 56.0, 62.0, 75.0, 86.0, 88.0, 90.0, 106.0, 108.0, 114.0]),
           st.sampled_from([15.0, 17.0, 21.0, 26.0, 34.0, 36.0, 38.0, 40.0, 48.0, 51.0]),
           st.sampled_from([70.0, 110.0, 127.0, 156.0, 165.0, 167.0, 184.0, 237.0, 245.0, 579.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=53.6, max_value=66.99, allow_nan=False),
           st.sampled_from([17.0, 18.0, 20.0, 23.0, 26.0, 33.0, 43.0, 44.0, 48.0, 56.0]),
           st.sampled_from([99.0, 100.0, 114.0, 122.0, 135.0, 230.0, 300.0, 321.0, 579.0, 846.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2324, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=53.6, max_value=66.99, allow_nan=False),
           st.sampled_from([14.0, 20.0, 31.0, 39.0, 40.0, 47.0, 48.0, 51.0, 56.0, 63.0]),
           st.sampled_from([58.0, 145.0, 155.0, 171.0, 200.0, 215.0, 231.0, 285.0, 321.0, 579.0]),
           st.floats(min_value=26.37, max_value=27.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3622, max_value=0.4019, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=53.6, max_value=66.99, allow_nan=False),
           st.sampled_from([8.0, 10.0, 11.0, 12.0, 25.0, 26.0, 30.0, 35.0, 36.0, 39.0]),
           st.sampled_from([37.0, 43.0, 68.0, 79.0, 148.0, 190.0, 265.0, 275.0, 285.0, 335.0]),
           st.floats(min_value=30.07, max_value=37.47, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3622, max_value=0.4019, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=103.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=70.2, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 12.0, 13.0, 18.0, 22.0, 28.0, 35.0, 39.0, 44.0, 60.0]),
           st.sampled_from([41.0, 71.0, 122.0, 160.0, 178.0, 182.0, 192.0, 204.0, 415.0, 440.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2105, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=103.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=70.2, exclude_min=True, allow_nan=False),
           st.sampled_from([14.0, 17.0, 27.0, 29.0, 32.0, 35.0, 46.0, 56.0, 63.0, 99.0]),
           st.sampled_from([70.0, 99.0, 100.0, 156.0, 165.0, 168.0, 280.0, 300.0, 360.0, 510.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2526, max_value=0.3142, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=120.01, max_value=121.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=70.2, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 19.0, 30.0, 32.0, 35.0, 36.0, 46.0, 49.0, 56.0, 63.0]),
           st.sampled_from([74.0, 175.0, 192.0, 237.0, 245.0, 328.0, 370.0, 465.0, 478.0, 540.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2078, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=120.01, max_value=121.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=70.2, exclude_min=True, allow_nan=False),
           st.sampled_from([19.0, 20.0, 22.0, 23.0, 24.0, 25.0, 30.0, 32.0, 34.0, 42.0]),
           st.sampled_from([79.0, 150.0, 167.0, 231.0, 245.0, 250.0, 318.0, 360.0, 543.0, 579.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2391, max_value=0.3034, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=28.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=120.01, max_value=121.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=70.2, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 15.0, 16.0, 17.0, 18.0, 31.0, 35.0, 45.0, 46.0, 48.0]),
           st.sampled_from([92.0, 94.0, 95.0, 122.0, 140.0, 178.0, 205.0, 240.0, 255.0, 485.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2391, max_value=0.3034, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=4.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=83.01, max_value=90.8, exclude_min=True, allow_nan=False),
           st.sampled_from([19.0, 25.0, 30.0, 31.0, 35.0, 36.0, 37.0, 39.0, 40.0, 43.0]),
           st.sampled_from([41.0, 55.0, 65.0, 72.0, 83.0, 106.0, 182.0, 215.0, 293.0, 402.0]),
           st.floats(min_value=26.37, max_value=27.98, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=3.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=78.0, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 14.0, 16.0, 17.0, 24.0, 25.0, 34.0, 35.0, 39.0, 52.0]),
           st.sampled_from([25.0, 41.0, 88.0, 94.0, 182.0, 194.0, 210.0, 265.0, 335.0, 680.0]),
           st.floats(min_value=34.47, max_value=36.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=3.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=78.0, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 15.0, 17.0, 19.0, 20.0, 25.0, 31.0, 32.0, 34.0, 35.0]),
           st.sampled_from([0.0, 70.0, 96.0, 146.0, 205.0, 215.0, 304.0, 478.0, 480.0, 543.0]),
           st.floats(min_value=43.12, max_value=47.91, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=11.51, max_value=12.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.01, max_value=78.0, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 18.0, 20.0, 24.0, 26.0, 27.0, 29.0, 38.0, 44.0, 46.0]),
           st.sampled_from([48.0, 64.0, 90.0, 130.0, 155.0, 191.0, 207.0, 318.0, 325.0, 600.0]),
           st.floats(min_value=34.47, max_value=40.99, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2001, max_value=0.2722, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 54.0, 62.0, 66.0, 82.0, 88.0, 90.0, 94.0, 96.0, 100.0]),
           st.sampled_from([0.0, 12.0, 18.0, 22.0, 29.0, 30.0, 33.0, 40.0, 43.0, 46.0]),
           st.floats(min_value=96.4, max_value=120.49, allow_nan=False),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=29.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 50.0, 78.0, 85.0, 96.0, 100.0, 102.0, 104.0, 106.0, 110.0]),
           st.sampled_from([7.0, 14.0, 17.0, 26.0, 27.0, 32.0, 33.0, 34.0, 35.0, 49.0]),
           st.floats(min_value=96.4, max_value=120.49, allow_nan=False),
           st.floats(min_value=26.37, max_value=27.72, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=34.51, max_value=43.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 52.0, 56.0, 72.0, 74.0, 85.0, 86.0, 94.0, 98.0, 122.0]),
           st.sampled_from([8.0, 11.0, 19.0, 25.0, 26.0, 27.0, 36.0, 47.0, 48.0, 52.0]),
           st.floats(min_value=96.4, max_value=120.49, allow_nan=False),
           st.floats(min_value=33.16, max_value=39.94, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=34.51, max_value=43.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([40.0, 64.0, 65.0, 72.0, 75.0, 86.0, 90.0, 96.0, 100.0, 104.0]),
           st.floats(min_value=38.0, max_value=47.49, allow_nan=False),
           st.floats(min_value=120.51, max_value=141.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 50.0, 65.0, 72.0, 75.0, 76.0, 78.0, 82.0, 84.0, 110.0]),
           st.floats(min_value=38.0, max_value=47.49, allow_nan=False),
           st.floats(min_value=225.01, max_value=349.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([46.0, 56.0, 65.0, 72.0, 75.0, 82.0, 88.0, 92.0, 95.0, 108.0]),
           st.floats(min_value=47.51, max_value=57.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=120.51, max_value=265.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.51, max_value=8.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=99.51, max_value=105.1, exclude_min=True, allow_nan=False),
           st.sampled_from([52.0, 54.0, 60.0, 64.0, 75.0, 78.0, 80.0, 90.0, 104.0, 108.0]),
           st.sampled_from([12.0, 21.0, 30.0, 33.0, 38.0, 39.0, 47.0, 48.0, 49.0, 63.0]),
           st.sampled_from([58.0, 145.0, 175.0, 185.0, 194.0, 237.0, 304.0, 325.0, 370.0, 579.0]),
           st.floats(min_value=26.37, max_value=34.51, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5611, max_value=0.9328, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.51, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 5.0, 7.0, 9.0, 10.0, 13.0, 14.0, 15.0]),
           st.floats(min_value=127.51, max_value=127.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=72.99, allow_nan=False),
           st.sampled_from([12.0, 20.0, 26.0, 34.0, 35.0, 39.0, 47.0, 51.0, 56.0, 63.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.222, 0.479, 0.529, 0.588, 0.64, 0.66, 0.732, 0.757, 0.925, 0.97]),
           st.sampled_from([22.0, 28.0, 32.0, 36.0, 38.0, 45.0, 52.0, 54.0, 56.0, 62.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=129.51, max_value=131.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=56.8, max_value=70.99, allow_nan=False),
           st.sampled_from([11.0, 12.0, 18.0, 27.0, 29.0, 35.0, 41.0, 42.0, 54.0, 60.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.157, 0.161, 0.167, 0.187, 0.399, 0.514, 0.626, 0.686, 0.698, 0.947]),
           st.sampled_from([22.0, 28.0, 29.0, 33.0, 35.0, 36.0, 44.0, 52.0, 56.0, 57.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]),
           st.floats(min_value=138.01, max_value=139.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=56.8, max_value=70.99, allow_nan=False),
           st.floats(min_value=13.6, max_value=16.99, allow_nan=False),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.141, 0.205, 0.296, 0.349, 0.467, 0.52, 0.578, 0.693, 1.154, 1.224]),
           st.sampled_from([25.0, 29.0, 41.0, 43.0, 49.0, 50.0, 54.0, 56.0, 62.0, 70.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=138.01, max_value=139.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=56.8, max_value=70.99, allow_nan=False),
           st.floats(min_value=17.01, max_value=33.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.084, 0.118, 0.143, 0.209, 0.284, 0.336, 0.37, 0.388, 0.399, 0.6]),
           st.sampled_from([26.0, 29.0, 33.0, 37.0, 38.0, 40.0, 56.0, 60.0, 61.0, 68.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 5.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=129.51, max_value=132.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=71.01, max_value=71.4, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 12.0, 15.0, 24.0, 27.0, 37.0, 42.0, 47.0, 51.0, 99.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.127, 0.151, 0.331, 0.554, 0.674, 0.687, 0.719, 0.817, 1.191, 1.282]),
           st.sampled_from([24.0, 25.0, 31.0, 40.0, 46.0, 47.0, 49.0, 50.0, 51.0, 66.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=131.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=73.01, max_value=82.8, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 14.0, 22.0, 24.0, 27.0, 35.0, 36.0, 39.0, 42.0, 48.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=22.51, max_value=28.13, allow_nan=False),
           st.sampled_from([0.144, 0.159, 0.194, 0.27, 0.365, 0.433, 0.66, 0.686, 0.766, 1.159]),
           st.sampled_from([31.0, 33.0, 40.0, 41.0, 50.0, 52.0, 53.0, 57.0, 66.0, 67.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=131.1, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 48.0, 50.0, 54.0, 75.0, 90.0, 92.0, 96.0, 98.0, 110.0]),
           st.sampled_from([12.0, 13.0, 23.0, 27.0, 34.0, 36.0, 46.0, 51.0, 56.0, 63.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=28.16, max_value=28.23, exclude_min=True, allow_nan=False),
           st.sampled_from([0.247, 0.264, 0.27, 0.344, 0.514, 0.578, 0.757, 0.855, 1.213, 1.318]),
           st.sampled_from([27.0, 28.0, 29.0, 42.0, 44.0, 46.0, 56.0, 60.0, 67.0, 70.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=129.0, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 60.0, 62.0, 80.0, 94.0, 96.0, 100.0, 102.0, 104.0, 114.0]),
           st.sampled_from([21.0, 30.0, 31.0, 32.0, 36.0, 42.0, 43.0, 46.0, 48.0, 51.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=28.57, max_value=28.84, exclude_min=True, allow_nan=False),
           st.sampled_from([0.15, 0.161, 0.203, 0.212, 0.431, 0.452, 0.484, 0.711, 1.114, 1.136]),
           st.sampled_from([22.0, 35.0, 37.0, 40.0, 47.0, 51.0, 57.0, 58.0, 59.0, 61.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
           st.floats(min_value=135.01, max_value=137.1, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 30.0, 50.0, 54.0, 58.0, 70.0, 80.0, 84.0, 95.0, 96.0]),
           st.sampled_from([10.0, 15.0, 20.0, 33.0, 37.0, 39.0, 40.0, 46.0, 54.0, 60.0]),
           st.floats(min_value=106.0, max_value=132.49, allow_nan=False),
           st.floats(min_value=28.57, max_value=28.84, exclude_min=True, allow_nan=False),
           st.sampled_from([0.147, 0.229, 0.257, 0.303, 0.482, 0.56, 0.564, 0.637, 0.658, 1.268]),
           st.sampled_from([37.0, 42.0, 43.0, 51.0, 55.0, 60.0, 62.0, 64.0, 67.0, 68.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=131.1, exclude_min=True, allow_nan=False),
           st.sampled_from([50.0, 65.0, 66.0, 70.0, 74.0, 76.0, 86.0, 88.0, 94.0, 95.0]),
           st.sampled_from([8.0, 12.0, 16.0, 23.0, 25.0, 29.0, 32.0, 35.0, 50.0, 54.0]),
           st.floats(min_value=132.51, max_value=275.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.95, max_value=29.93, allow_nan=False),
           st.sampled_from([0.259, 0.26, 0.29, 0.332, 0.374, 0.482, 0.678, 0.699, 0.773, 1.174]),
           st.sampled_from([24.0, 35.0, 36.0, 54.0, 58.0, 60.0, 61.0, 63.0, 68.0, 72.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 55.0, 58.0, 66.0, 70.0, 80.0, 86.0, 95.0, 96.0, 106.0]),
           st.sampled_from([8.0, 12.0, 18.0, 21.0, 33.0, 41.0, 43.0, 46.0, 47.0, 48.0]),
           st.sampled_from([16.0, 36.0, 56.0, 78.0, 122.0, 126.0, 130.0, 158.0, 194.0, 326.0]),
           st.floats(min_value=23.95, max_value=29.93, allow_nan=False),
           st.sampled_from([0.183, 0.236, 0.342, 0.368, 0.374, 0.507, 0.586, 0.735, 0.801, 1.441]),
           st.floats(min_value=24.6, max_value=25.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.sampled_from([46.0, 56.0, 58.0, 66.0, 70.0, 78.0, 82.0, 90.0, 96.0, 100.0]),
           st.sampled_from([0.0, 13.0, 15.0, 18.0, 22.0, 23.0, 27.0, 31.0, 37.0, 50.0]),
           st.sampled_from([16.0, 43.0, 59.0, 94.0, 110.0, 115.0, 132.0, 155.0, 176.0, 255.0]),
           st.floats(min_value=17.71, max_value=22.13, allow_nan=False),
           st.sampled_from([0.078, 0.1, 0.159, 0.173, 0.269, 0.338, 0.501, 0.892, 0.944, 1.6]),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 13.0, 15.0, 17.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.sampled_from([52.0, 66.0, 70.0, 72.0, 82.0, 88.0, 92.0, 96.0, 104.0, 106.0]),
           st.sampled_from([13.0, 14.0, 18.0, 20.0, 23.0, 34.0, 43.0, 45.0, 46.0, 47.0]),
           st.sampled_from([110.0, 125.0, 160.0, 200.0, 225.0, 240.0, 280.0, 465.0, 495.0, 600.0]),
           st.floats(min_value=22.16, max_value=23.14, exclude_min=True, allow_nan=False),
           st.sampled_from([0.165, 0.263, 0.345, 0.356, 0.395, 0.443, 0.455, 0.503, 0.851, 0.956]),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.6, max_value=81.99, allow_nan=False),
           st.sampled_from([8.0, 10.0, 11.0, 20.0, 29.0, 34.0, 41.0, 44.0, 46.0, 52.0]),
           st.sampled_from([64.0, 77.0, 125.0, 182.0, 188.0, 204.0, 215.0, 255.0, 284.0, 545.0]),
           st.floats(min_value=27.11, max_value=27.17, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3335, max_value=0.3973, allow_nan=False),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.6, max_value=81.99, allow_nan=False),
           st.sampled_from([0.0, 7.0, 19.0, 25.0, 30.0, 36.0, 38.0, 40.0, 44.0, 45.0]),
           st.sampled_from([100.0, 120.0, 129.0, 132.0, 145.0, 168.0, 180.0, 207.0, 249.0, 510.0]),
           st.floats(min_value=27.46, max_value=27.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3335, max_value=0.3973, allow_nan=False),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.6, max_value=81.99, allow_nan=False),
           st.sampled_from([8.0, 10.0, 21.0, 22.0, 32.0, 35.0, 45.0, 52.0, 54.0, 60.0]),
           st.sampled_from([42.0, 65.0, 87.0, 92.0, 94.0, 126.0, 132.0, 145.0, 148.0, 335.0]),
           st.floats(min_value=27.11, max_value=27.67, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3976, max_value=0.802, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=82.01, max_value=90.0, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 18.0, 29.0, 31.0, 34.0, 38.0, 40.0, 42.0, 47.0, 52.0]),
           st.sampled_from([68.0, 78.0, 89.0, 95.0, 130.0, 145.0, 183.0, 204.0, 210.0, 285.0]),
           st.floats(min_value=27.11, max_value=27.67, exclude_min=True, allow_nan=False),
           st.sampled_from([0.089, 0.165, 0.206, 0.28, 0.482, 0.514, 0.703, 0.748, 0.917, 0.966]),
           st.floats(min_value=25.51, max_value=32.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 11.0, 13.0]),
           st.floats(min_value=145.51, max_value=156.2, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 38.0, 44.0, 58.0, 60.0, 65.0, 76.0, 86.0, 94.0, 110.0]),
           st.sampled_from([0.0, 14.0, 19.0, 20.0, 22.0, 26.0, 28.0, 37.0, 48.0, 60.0]),
           st.sampled_from([95.0, 110.0, 122.0, 135.0, 148.0, 170.0, 188.0, 194.0, 278.0, 485.0]),
           st.floats(min_value=23.95, max_value=29.93, allow_nan=False),
           st.sampled_from([0.179, 0.219, 0.231, 0.547, 0.557, 0.559, 0.614, 0.717, 0.892, 1.781]),
           st.floats(min_value=61.01, max_value=65.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 4.0, 6.0, 9.0, 10.0, 11.0, 13.0, 14.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=48.8, max_value=60.99, allow_nan=False),
           st.sampled_from([20.0, 23.0, 25.0, 26.0, 29.0, 32.0, 35.0, 42.0, 47.0, 51.0]),
           st.sampled_from([96.0, 110.0, 115.0, 135.0, 171.0, 207.0, 225.0, 250.0, 328.0, 510.0]),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.sampled_from([0.163, 0.235, 0.238, 0.26, 0.52, 0.732, 0.803, 0.817, 0.893, 1.191]),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=62.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 13.0, 14.0, 19.0, 20.0, 24.0, 25.0, 37.0, 56.0, 99.0]),
           st.sampled_from([58.0, 176.0, 207.0, 220.0, 231.0, 304.0, 318.0, 325.0, 360.0, 370.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=24.6, max_value=25.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4, max_value=0.49, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=68.01, max_value=69.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 7.0, 24.0, 33.0, 38.0, 40.0, 41.0, 42.0, 52.0, 54.0]),
           st.sampled_from([22.0, 53.0, 56.0, 68.0, 76.0, 95.0, 125.0, 231.0, 335.0, 440.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=24.6, max_value=25.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=63.4, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 19.0, 23.0, 24.0, 29.0, 30.0, 32.0, 40.0, 46.0, 54.0]),
           st.sampled_from([49.0, 52.0, 89.0, 100.0, 130.0, 183.0, 188.0, 200.0, 272.0, 545.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=24.6, max_value=25.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.49, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=63.4, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 13.0, 15.0, 17.0, 21.0, 32.0, 38.0, 41.0, 42.0, 46.0]),
           st.sampled_from([36.0, 126.0, 129.0, 144.0, 215.0, 245.0, 274.0, 465.0, 474.0, 579.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=25.51, max_value=26.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=63.4, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 8.0, 16.0, 18.0, 20.0, 24.0, 27.0, 41.0, 43.0, 50.0]),
           st.sampled_from([48.0, 49.0, 65.0, 72.0, 95.0, 130.0, 155.0, 190.0, 192.0, 194.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=25.51, max_value=26.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.49, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=73.01, max_value=82.8, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 11.0, 13.0, 21.0, 27.0, 28.0, 29.0, 32.0, 34.0, 39.0]),
           st.sampled_from([32.0, 78.0, 90.0, 116.0, 170.0, 183.0, 188.0, 231.0, 265.0, 325.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=7.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([11.0, 12.0, 16.0, 22.0, 24.0, 25.0, 31.0, 45.0, 46.0, 60.0]),
           st.floats(min_value=100.0, max_value=124.99, allow_nan=False),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=7.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 15.0, 17.0, 28.0, 30.0, 31.0, 34.0, 35.0, 39.0, 56.0]),
           st.floats(min_value=125.01, max_value=269.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9287, max_value=1.1413, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 7.0, 15.0, 17.0, 29.0, 31.0, 33.0, 35.0, 46.0, 56.0]),
           st.sampled_from([70.0, 110.0, 132.0, 182.0, 190.0, 240.0, 271.0, 277.0, 465.0, 510.0]),
           st.floats(min_value=29.96, max_value=32.32, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1416, max_value=1.3972, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 12.0, 13.0, 14.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([14.0, 26.0, 27.0, 28.0, 30.0, 34.0, 37.0, 38.0, 41.0, 42.0]),
           st.sampled_from([0.0, 88.0, 96.0, 122.0, 155.0, 165.0, 184.0, 207.0, 293.0, 304.0]),
           st.floats(min_value=41.81, max_value=46.86, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3139, max_value=0.3728, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0, 15.0]),
           st.floats(min_value=127.51, max_value=130.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([22.0, 23.0, 26.0, 29.0, 31.0, 36.0, 41.0, 42.0, 49.0, 51.0]),
           st.sampled_from([122.0, 145.0, 160.0, 175.0, 180.0, 184.0, 250.0, 274.0, 285.0, 600.0]),
           st.floats(min_value=41.81, max_value=46.86, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3731, max_value=0.7824, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0]),
           st.floats(min_value=142.51, max_value=145.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.01, max_value=73.2, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 16.0, 22.0, 23.0, 26.0, 33.0, 34.0, 38.0, 47.0, 52.0]),
           st.sampled_from([36.0, 46.0, 176.0, 183.0, 196.0, 215.0, 255.0, 291.0, 342.0, 485.0]),
           st.floats(min_value=41.81, max_value=46.86, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3731, max_value=0.7824, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 5.0, 6.0, 10.0, 12.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 54.0, 64.0, 68.0, 72.0, 80.0, 90.0, 98.0, 100.0, 108.0]),
           st.sampled_from([7.0, 17.0, 20.0, 29.0, 33.0, 36.0, 40.0, 41.0, 43.0, 48.0]),
           st.sampled_from([70.0, 99.0, 125.0, 156.0, 167.0, 168.0, 191.0, 207.0, 240.0, 600.0]),
           st.floats(min_value=29.96, max_value=33.07, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1247, max_value=0.1363, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 9.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 48.0, 50.0, 55.0, 56.0, 62.0, 72.0, 92.0, 100.0, 108.0]),
           st.sampled_from([18.0, 21.0, 32.0, 34.0, 36.0, 42.0, 46.0, 47.0, 48.0, 54.0]),
           st.sampled_from([15.0, 32.0, 44.0, 82.0, 84.0, 86.0, 120.0, 192.0, 230.0, 265.0]),
           st.floats(min_value=29.96, max_value=30.45, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([56.0, 64.0, 80.0, 82.0, 85.0, 86.0, 88.0, 94.0, 104.0, 114.0]),
           st.sampled_from([15.0, 18.0, 19.0, 26.0, 30.0, 31.0, 47.0, 48.0, 51.0, 99.0]),
           st.sampled_from([96.0, 99.0, 115.0, 176.0, 182.0, 191.0, 237.0, 277.0, 293.0, 474.0]),
           st.floats(min_value=32.47, max_value=32.58, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 38.0, 52.0, 56.0, 58.0, 70.0, 74.0, 84.0, 88.0, 106.0]),
           st.floats(min_value=24.0, max_value=29.99, allow_nan=False),
           st.sampled_from([32.0, 50.0, 73.0, 190.0, 204.0, 210.0, 240.0, 325.0, 326.0, 342.0]),
           st.floats(min_value=33.07, max_value=35.56, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=32.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_104(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_104']['n_samples'] += 1
        self.data['tests']['test_104']['samples'].append(x_test)
        self.data['tests']['test_104']['y_expected'].append(y_expected[0])
        self.data['tests']['test_104']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 11.0, 13.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([40.0, 54.0, 56.0, 74.0, 84.0, 92.0, 100.0, 104.0, 106.0, 108.0]),
           st.floats(min_value=24.0, max_value=29.99, allow_nan=False),
           st.sampled_from([74.0, 88.0, 132.0, 140.0, 155.0, 168.0, 192.0, 205.0, 543.0, 846.0]),
           st.floats(min_value=33.07, max_value=35.56, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=42.01, max_value=42.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 55.0, 60.0, 65.0, 70.0, 74.0, 94.0, 96.0, 98.0, 122.0]),
           st.floats(min_value=24.0, max_value=29.99, allow_nan=False),
           st.sampled_from([43.0, 53.0, 61.0, 63.0, 64.0, 115.0, 119.0, 126.0, 205.0, 387.0]),
           st.floats(min_value=33.07, max_value=35.56, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=44.51, max_value=51.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_106(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_106']['n_samples'] += 1
        self.data['tests']['test_106']['samples'].append(x_test)
        self.data['tests']['test_106']['y_expected'].append(y_expected[0])
        self.data['tests']['test_106']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([44.0, 46.0, 65.0, 75.0, 76.0, 78.0, 80.0, 84.0, 90.0, 108.0]),
           st.floats(min_value=30.01, max_value=43.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=35.2, max_value=43.99, allow_nan=False),
           st.floats(min_value=33.07, max_value=35.56, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 5.0, 7.0, 9.0, 10.0, 11.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([60.0, 66.0, 74.0, 78.0, 86.0, 88.0, 90.0, 98.0, 104.0, 108.0]),
           st.floats(min_value=30.01, max_value=43.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=44.01, max_value=204.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=33.07, max_value=35.56, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1366, max_value=0.1951, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_108(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_108']['n_samples'] += 1
        self.data['tests']['test_108']['samples'].append(x_test)
        self.data['tests']['test_108']['y_expected'].append(y_expected[0])
        self.data['tests']['test_108']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 4.0, 5.0, 7.0, 9.0, 10.0, 13.0, 14.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 30.0, 54.0, 56.0, 62.0, 82.0, 90.0, 92.0, 94.0, 114.0]),
           st.sampled_from([15.0, 19.0, 21.0, 27.0, 28.0, 30.0, 34.0, 47.0, 48.0, 63.0]),
           st.sampled_from([48.0, 110.0, 129.0, 156.0, 184.0, 321.0, 370.0, 478.0, 510.0, 846.0]),
           st.floats(min_value=45.56, max_value=49.86, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3591, max_value=0.4293, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_109(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_109']['n_samples'] += 1
        self.data['tests']['test_109']['samples'].append(x_test)
        self.data['tests']['test_109']['y_expected'].append(y_expected[0])
        self.data['tests']['test_109']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 7.0, 11.0, 13.0, 14.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 40.0, 56.0, 58.0, 70.0, 80.0, 90.0, 92.0, 102.0, 108.0]),
           st.sampled_from([15.0, 17.0, 19.0, 25.0, 30.0, 31.0, 34.0, 35.0, 51.0, 63.0]),
           st.floats(min_value=266.8, max_value=333.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=31.97, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.8276, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 12.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.0, max_value=79.99, allow_nan=False),
           st.sampled_from([15.0, 17.0, 18.0, 19.0, 20.0, 24.0, 37.0, 39.0, 56.0, 99.0]),
           st.floats(min_value=266.8, max_value=333.49, allow_nan=False),
           st.floats(min_value=40.07, max_value=45.47, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.8276, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_111(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_111']['n_samples'] += 1
        self.data['tests']['test_111']['samples'].append(x_test)
        self.data['tests']['test_111']['y_expected'].append(y_expected[0])
        self.data['tests']['test_111']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=80.01, max_value=88.4, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 20.0, 22.0, 25.0, 26.0, 40.0, 42.0, 45.0, 50.0, 52.0]),
           st.floats(min_value=266.8, max_value=333.49, allow_nan=False),
           st.floats(min_value=40.07, max_value=45.47, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.8276, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_112(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_112']['n_samples'] += 1
        self.data['tests']['test_112']['samples'].append(x_test)
        self.data['tests']['test_112']['y_expected'].append(y_expected[0])
        self.data['tests']['test_112']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 3.0, 4.0, 6.0, 7.0, 10.0, 11.0, 13.0, 15.0, 17.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.2, max_value=63.99, allow_nan=False),
           st.sampled_from([0.0, 14.0, 18.0, 19.0, 26.0, 31.0, 34.0, 42.0, 44.0, 49.0]),
           st.floats(min_value=333.51, max_value=436.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.8276, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_113(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_113']['n_samples'] += 1
        self.data['tests']['test_113']['samples'].append(x_test)
        self.data['tests']['test_113']['y_expected'].append(y_expected[0])
        self.data['tests']['test_113']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0]),
           st.floats(min_value=127.51, max_value=133.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.01, max_value=75.6, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 8.0, 10.0, 11.0, 20.0, 23.0, 25.0, 32.0, 39.0, 44.0]),
           st.floats(min_value=333.51, max_value=436.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.8276, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.51, max_value=40.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_114(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_114']['n_samples'] += 1
        self.data['tests']['test_114']['samples'].append(x_test)
        self.data['tests']['test_114']['y_expected'].append(y_expected[0])
        self.data['tests']['test_114']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=157.51, max_value=163.3, exclude_min=True, allow_nan=False),
           st.sampled_from([46.0, 54.0, 60.0, 65.0, 68.0, 70.0, 90.0, 106.0, 110.0, 122.0]),
           st.sampled_from([10.0, 11.0, 22.0, 26.0, 32.0, 34.0, 44.0, 47.0, 48.0, 50.0]),
           st.floats(min_value=70.0, max_value=87.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=34.2, max_value=37.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_115(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_115']['n_samples'] += 1
        self.data['tests']['test_115']['samples'].append(x_test)
        self.data['tests']['test_115']['y_expected'].append(y_expected[0])
        self.data['tests']['test_115']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 9.0, 10.0, 14.0, 17.0]),
           st.floats(min_value=186.51, max_value=189.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 56.0, 58.0, 62.0, 64.0, 72.0, 82.0, 96.0, 98.0, 108.0]),
           st.sampled_from([7.0, 18.0, 19.0, 21.0, 27.0, 28.0, 32.0, 34.0, 38.0, 49.0]),
           st.floats(min_value=70.0, max_value=87.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=34.2, max_value=37.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_116(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_116']['n_samples'] += 1
        self.data['tests']['test_116']['samples'].append(x_test)
        self.data['tests']['test_116']['y_expected'].append(y_expected[0])
        self.data['tests']['test_116']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 17.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([48.0, 54.0, 62.0, 72.0, 75.0, 84.0, 92.0, 98.0, 106.0, 108.0]),
           st.sampled_from([0.0, 12.0, 18.0, 25.0, 26.0, 34.0, 41.0, 43.0, 49.0, 51.0]),
           st.floats(min_value=87.51, max_value=195.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=33.08, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=34.2, max_value=37.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_117(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_117']['n_samples'] += 1
        self.data['tests']['test_117']['samples'].append(x_test)
        self.data['tests']['test_117']['y_expected'].append(y_expected[0])
        self.data['tests']['test_117']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 44.0, 50.0, 56.0, 61.0, 65.0, 70.0, 75.0, 90.0, 96.0]),
           st.sampled_from([8.0, 11.0, 12.0, 25.0, 26.0, 27.0, 32.0, 36.0, 40.0, 42.0]),
           st.floats(min_value=87.51, max_value=195.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=45.62, max_value=49.91, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=34.2, max_value=37.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_118(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_118']['n_samples'] += 1
        self.data['tests']['test_118']['samples'].append(x_test)
        self.data['tests']['test_118']['y_expected'].append(y_expected[0])
        self.data['tests']['test_118']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 12.0, 13.0, 14.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 60.0, 65.0, 66.0, 70.0, 84.0, 94.0, 98.0, 104.0, 110.0]),
           st.sampled_from([0.0, 7.0, 17.0, 25.0, 30.0, 32.0, 33.0, 35.0, 45.0, 49.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=37.51, max_value=41.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_119(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_119']['n_samples'] += 1
        self.data['tests']['test_119']['samples'].append(x_test)
        self.data['tests']['test_119']['y_expected'].append(y_expected[0])
        self.data['tests']['test_119']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 40.0, 48.0, 60.0, 66.0, 70.0, 74.0, 88.0, 98.0, 102.0]),
           st.sampled_from([15.0, 17.0, 19.0, 25.0, 34.0, 36.0, 37.0, 40.0, 43.0, 44.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=56.51, max_value=61.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_120(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_120']['n_samples'] += 1
        self.data['tests']['test_120']['samples'].append(x_test)
        self.data['tests']['test_120']['y_expected'].append(y_expected[0])
        self.data['tests']['test_120']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.51, max_value=8.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 44.0, 52.0, 62.0, 64.0, 66.0, 70.0, 82.0, 96.0, 98.0]),
           st.sampled_from([11.0, 15.0, 19.0, 21.0, 24.0, 27.0, 32.0, 35.0, 52.0, 54.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2463, max_value=0.2883, allow_nan=False),
           st.floats(min_value=56.51, max_value=61.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_121(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_121']['n_samples'] += 1
        self.data['tests']['test_121']['samples'].append(x_test)
        self.data['tests']['test_121']['y_expected'].append(y_expected[0])
        self.data['tests']['test_121']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 2.0, 3.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([50.0, 60.0, 64.0, 68.0, 74.0, 80.0, 84.0, 85.0, 94.0, 110.0]),
           st.sampled_from([8.0, 12.0, 16.0, 17.0, 20.0, 26.0, 27.0, 45.0, 52.0, 60.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2886, max_value=0.2909, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 32.0, 36.0, 37.0, 40.0, 43.0, 46.0, 53.0, 64.0, 66.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_122(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_122']['n_samples'] += 1
        self.data['tests']['test_122']['samples'].append(x_test)
        self.data['tests']['test_122']['y_expected'].append(y_expected[0])
        self.data['tests']['test_122']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 48.0, 54.0, 56.0, 58.0, 65.0, 68.0, 88.0, 98.0, 110.0]),
           st.sampled_from([7.0, 14.0, 15.0, 23.0, 27.0, 34.0, 47.0, 48.0, 51.0, 56.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3006, max_value=0.7244, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.4, max_value=43.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_123(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_123']['n_samples'] += 1
        self.data['tests']['test_123']['samples'].append(x_test)
        self.data['tests']['test_123']['y_expected'].append(y_expected[0])
        self.data['tests']['test_123']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 38.0, 46.0, 50.0, 52.0, 55.0, 72.0, 84.0, 85.0, 110.0]),
           st.sampled_from([7.0, 18.0, 20.0, 29.0, 31.0, 32.0, 34.0, 38.0, 43.0, 54.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3006, max_value=0.7244, exclude_min=True, allow_nan=False),
           st.floats(min_value=44.01, max_value=45.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_124(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_124']['n_samples'] += 1
        self.data['tests']['test_124']['samples'].append(x_test)
        self.data['tests']['test_124']['y_expected'].append(y_expected[0])
        self.data['tests']['test_124']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([40.0, 60.0, 64.0, 66.0, 70.0, 74.0, 75.0, 92.0, 100.0, 110.0]),
           st.sampled_from([13.0, 15.0, 19.0, 23.0, 28.0, 35.0, 37.0, 41.0, 44.0, 46.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3006, max_value=0.4717, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.01, max_value=57.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_125(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_125']['n_samples'] += 1
        self.data['tests']['test_125']['samples'].append(x_test)
        self.data['tests']['test_125']['y_expected'].append(y_expected[0])
        self.data['tests']['test_125']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.49, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 62.0, 74.0, 76.0, 86.0, 88.0, 94.0, 98.0, 108.0, 110.0]),
           st.sampled_from([7.0, 18.0, 21.0, 24.0, 27.0, 31.0, 39.0, 43.0, 46.0, 52.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1567, max_value=1.4093, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.01, max_value=57.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_126(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_126']['n_samples'] += 1
        self.data['tests']['test_126']['samples'].append(x_test)
        self.data['tests']['test_126']['y_expected'].append(y_expected[0])
        self.data['tests']['test_126']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.51, max_value=8.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([48.0, 54.0, 56.0, 62.0, 68.0, 74.0, 80.0, 82.0, 96.0, 102.0]),
           st.sampled_from([13.0, 22.0, 25.0, 26.0, 28.0, 33.0, 35.0, 45.0, 47.0, 56.0]),
           st.floats(min_value=503.6, max_value=629.49, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3006, max_value=0.7244, exclude_min=True, allow_nan=False),
           st.floats(min_value=44.01, max_value=51.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_127(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_127']['n_samples'] += 1
        self.data['tests']['test_127']['samples'].append(x_test)
        self.data['tests']['test_127']['y_expected'].append(y_expected[0])
        self.data['tests']['test_127']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 38.0, 46.0, 56.0, 58.0, 68.0, 88.0, 94.0, 96.0, 108.0]),
           st.sampled_from([0.0, 7.0, 15.0, 19.0, 21.0, 28.0, 37.0, 40.0, 45.0, 60.0]),
           st.floats(min_value=629.51, max_value=662.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.sampled_from([0.084, 0.141, 0.17, 0.432, 0.433, 0.443, 0.472, 0.501, 0.828, 1.162]),
           st.sampled_from([25.0, 27.0, 28.0, 44.0, 46.0, 54.0, 58.0, 63.0, 68.0, 72.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_128(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_128']['n_samples'] += 1
        self.data['tests']['test_128']['samples'].append(x_test)
        self.data['tests']['test_128']['y_expected'].append(y_expected[0])
        self.data['tests']['test_128']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
           st.floats(min_value=157.51, max_value=165.8, exclude_min=True, allow_nan=False),
           st.sampled_from([52.0, 58.0, 64.0, 68.0, 75.0, 76.0, 82.0, 84.0, 85.0, 106.0]),
           st.sampled_from([0.0, 12.0, 17.0, 22.0, 25.0, 31.0, 36.0, 42.0, 47.0, 63.0]),
           st.floats(min_value=795.01, max_value=805.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.96, max_value=37.38, exclude_min=True, allow_nan=False),
           st.sampled_from([0.148, 0.15, 0.297, 0.528, 0.652, 0.732, 0.734, 1.034, 1.391, 2.137]),
           st.sampled_from([21.0, 25.0, 26.0, 34.0, 36.0, 37.0, 42.0, 49.0, 61.0, 66.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_129(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_129']['n_samples'] += 1
        self.data['tests']['test_129']['samples'].append(x_test)
        self.data['tests']['test_129']['y_expected'].append(y_expected[0])
        self.data['tests']['test_129']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
