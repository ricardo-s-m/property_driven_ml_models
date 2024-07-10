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
    request.cls.data['n_test'] = 50
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_glass_identification_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestGlassIdentificationProperty:

    @given(st.floats(min_value=1.51115, max_value=1.515994, allow_nan=False),
           st.floats(min_value=10.73, max_value=15.333, allow_nan=False),
           st.sampled_from([2.71, 3.42, 3.46, 3.5, 3.55, 3.6, 3.61, 3.62, 3.73, 3.75]),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([71.36, 71.57, 71.77, 71.78, 71.95, 72.02, 72.32, 72.73, 72.95, 73.28]),
           st.sampled_from([0.02, 0.03, 0.06, 0.18, 0.19, 0.51, 0.56, 0.61, 0.64, 0.67]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.06, 0.07, 0.09, 0.11, 0.16, 0.17, 0.19, 0.24, 0.26, 0.31]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.515994, allow_nan=False),
           st.floats(min_value=15.336, max_value=17.38, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.78, 1.74, 2.19, 2.24, 2.39, 2.41]),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([72.37, 72.38, 72.5, 72.67, 72.74, 72.76, 73.48, 74.55, 75.41]),
           st.sampled_from([0.0]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [6]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515997, max_value=1.517068, exclude_min=True, allow_nan=False),
           st.sampled_from([12.16, 13.04, 13.14, 13.24, 13.33, 13.41, 13.5, 13.65, 14.03, 14.19]),
           st.sampled_from([3.36, 3.39, 3.45, 3.52, 3.54, 3.57, 3.66, 3.76, 3.78, 3.9]),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.floats(min_value=69.81, max_value=72.723, allow_nan=False),
           st.sampled_from([0.0, 0.06, 0.11, 0.52, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.1, 0.17, 0.24, 0.37]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515997, max_value=1.517068, exclude_min=True, allow_nan=False),
           st.sampled_from([11.02, 12.87, 12.94, 13.01, 13.25, 13.34, 13.4, 13.46, 13.72, 14.86]),
           st.sampled_from([2.09, 2.85, 2.96, 3.18, 3.49, 3.54, 3.66, 3.83, 3.89, 3.9]),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.floats(min_value=72.726, max_value=72.883, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.12, 0.38, 0.55, 0.56, 0.61, 0.66, 0.68, 0.81, 1.1]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.09, 0.1, 0.12, 0.14, 0.18, 0.19, 0.21, 0.22, 0.28, 0.29]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515997, max_value=1.517068, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=12.468, allow_nan=False),
           st.floats(min_value=0.0, max_value=3.289, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.floats(min_value=72.886, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.04, 0.05, 0.31, 0.6, 0.76, 1.41, 1.46, 1.76, 2.7]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.01, 0.05, 0.07, 0.08, 0.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [7]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515997, max_value=1.517068, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.471, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=3.289, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.floats(min_value=72.886, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.1, 0.33, 0.38, 0.44, 0.54, 0.56, 0.57, 0.6, 0.62]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.08, 0.15, 0.2, 0.21, 0.22, 0.29, 0.32, 0.34, 0.35]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515997, max_value=1.517068, exclude_min=True, allow_nan=False),
           st.sampled_from([12.86, 13.04, 13.24, 13.33, 13.41, 13.42, 13.5, 13.64, 14.19, 14.32]),
           st.floats(min_value=3.292, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.floats(min_value=72.886, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.06, 0.16, 0.23, 0.52, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.1, 0.17, 0.24, 0.37]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([13.79, 14.0, 14.09, 14.15, 14.4, 14.46, 14.56, 14.99, 17.38]),
           st.floats(min_value=0.0, max_value=2.609, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([72.37, 72.38, 72.5, 72.67, 72.74, 72.76, 73.48, 74.55, 75.41]),
           st.sampled_from([0.0]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [6]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.523073, exclude_min=True, allow_nan=False),
           st.sampled_from([12.61, 12.84, 13.02, 13.39, 13.49, 13.51, 13.58, 13.64, 13.81, 14.04]),
           st.floats(min_value=2.612, max_value=3.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([71.57, 71.99, 72.36, 72.72, 72.84, 72.96, 73.2, 73.21, 73.27, 73.29]),
           st.sampled_from([0.0, 0.02, 0.12, 0.39, 0.48, 0.54, 0.56, 0.58, 0.59, 0.65]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.523076, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([13.44, 13.69, 13.88, 14.23, 14.37, 14.7, 14.75, 14.8, 14.95, 15.15]),
           st.floats(min_value=2.612, max_value=3.633, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([70.26, 70.43, 72.38, 72.61, 72.81, 72.99, 73.1, 73.11, 73.46, 73.61]),
           st.sampled_from([0.0, 0.04, 0.05, 0.08, 0.14, 0.31, 0.6, 1.41, 1.76, 2.7]),
           st.floats(min_value=5.43, max_value=10.479, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [7]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.517884, exclude_min=True, allow_nan=False),
           st.sampled_from([12.74, 12.86, 13.0, 13.39, 13.51, 13.58, 13.72, 13.81, 13.99, 14.21]),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([71.36, 71.76, 71.78, 71.81, 72.2, 72.36, 72.75, 72.97, 73.02, 73.21]),
           st.sampled_from([0.11, 0.13, 0.14, 0.48, 0.54, 0.59, 0.61, 0.64, 0.65, 0.69]),
           st.floats(min_value=5.43, max_value=8.539, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517887, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([11.45, 12.93, 12.96, 13.02, 13.41, 13.43, 13.44, 13.7, 14.25, 14.86]),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([72.19, 72.28, 72.34, 72.4, 72.53, 72.96, 73.01, 73.06, 73.21, 73.81]),
           st.sampled_from([0.19, 0.35, 0.39, 0.49, 0.57, 0.62, 0.65, 0.66, 0.68, 1.1]),
           st.floats(min_value=5.43, max_value=8.539, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.517813, exclude_min=True, allow_nan=False),
           st.sampled_from([12.16, 13.04, 13.14, 13.24, 13.33, 13.42, 13.5, 13.53, 13.65, 14.19]),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.188, allow_nan=False),
           st.sampled_from([71.36, 71.5, 71.94, 72.14, 72.48, 72.65, 72.67, 72.69, 72.77, 72.89]),
           st.sampled_from([0.0, 0.06, 0.11, 0.16, 0.23, 0.52, 0.58, 0.59, 0.6, 0.61]),
           st.floats(min_value=8.542, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.517813, exclude_min=True, allow_nan=False),
           st.sampled_from([12.57, 12.79, 12.81, 12.84, 12.99, 13.0, 13.19, 13.2, 13.72, 14.36]),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.191, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([71.78, 71.95, 72.02, 72.08, 72.2, 72.61, 72.97, 73.02, 73.15, 73.29]),
           st.sampled_from([0.02, 0.11, 0.12, 0.13, 0.48, 0.51, 0.55, 0.65, 0.67, 0.69]),
           st.floats(min_value=8.542, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517816, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=14.118, allow_nan=False),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=0.643, allow_nan=False),
           st.sampled_from([71.5, 71.79, 72.04, 72.64, 72.67, 72.69, 72.77, 72.89, 73.0, 73.01]),
           st.sampled_from([0.0, 0.06, 0.11, 0.16, 0.23, 0.52, 0.56, 0.58, 0.59, 0.61]),
           st.floats(min_value=8.542, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517816, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.121, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=0.643, allow_nan=False),
           st.sampled_from([71.95, 71.96, 72.01, 72.2, 72.75, 72.84, 73.01, 73.09, 73.15, 73.2]),
           st.sampled_from([0.14, 0.15, 0.17, 0.19, 0.39, 0.5, 0.51, 0.56, 0.58, 0.62]),
           st.floats(min_value=8.542, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517816, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.65, 12.71, 13.14, 13.19, 13.21, 13.27, 13.31, 13.49, 13.89, 14.04]),
           st.floats(min_value=3.636, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.646, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([71.35, 71.72, 71.75, 72.08, 72.61, 72.72, 73.04, 73.08, 73.21, 73.27]),
           st.sampled_from([0.0, 0.03, 0.09, 0.11, 0.18, 0.39, 0.48, 0.5, 0.56, 0.58]),
           st.floats(min_value=8.542, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.113, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.419, allow_nan=False),
           st.floats(min_value=0.0, max_value=3.863, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.168, allow_nan=False),
           st.sampled_from([71.87, 72.02, 72.4, 72.45, 72.51, 72.88, 72.89, 72.97, 73.23, 73.36]),
           st.sampled_from([0.06, 0.12, 0.16, 0.55, 0.56, 0.67, 0.68, 0.69, 0.7, 1.1]),
           st.floats(min_value=5.43, max_value=9.618, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.116, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.422, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=3.863, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.168, allow_nan=False),
           st.sampled_from([71.5, 71.79, 71.94, 72.14, 72.61, 72.64, 72.67, 72.69, 72.7, 73.01]),
           st.sampled_from([0.0, 0.06, 0.11, 0.16, 0.23, 0.56, 0.57, 0.58, 0.59, 0.61]),
           st.floats(min_value=5.43, max_value=9.618, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.116, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.57, 12.68, 12.81, 12.99, 13.39, 13.48, 13.73, 13.81, 13.99, 14.77]),
           st.floats(min_value=0.0, max_value=3.614, allow_nan=False),
           st.floats(min_value=1.171, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([71.77, 71.95, 72.01, 72.72, 72.97, 73.09, 73.11, 73.21, 73.24, 73.7]),
           st.sampled_from([0.0, 0.02, 0.15, 0.51, 0.54, 0.58, 0.6, 0.61, 0.62, 0.64]),
           st.floats(min_value=5.43, max_value=9.618, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.116, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.2, 12.3, 12.79, 12.85, 12.87, 12.9, 12.99, 13.19, 13.3, 13.34]),
           st.floats(min_value=3.617, max_value=3.863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.171, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([70.16, 72.33, 72.78, 72.81, 72.83, 72.86, 72.92, 73.06, 73.55, 74.45]),
           st.sampled_from([0.0, 0.12, 0.44, 0.45, 0.53, 0.55, 0.56, 0.58, 0.66, 0.68]),
           st.floats(min_value=5.43, max_value=9.618, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.116, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.61, 12.68, 12.86, 12.98, 13.27, 13.31, 13.64, 13.69, 14.04, 14.77]),
           st.floats(min_value=0.0, max_value=3.863, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([71.36, 71.81, 72.12, 72.72, 72.85, 73.02, 73.08, 73.11, 73.39, 73.7]),
           st.sampled_from([0.03, 0.06, 0.12, 0.17, 0.23, 0.39, 0.51, 0.56, 0.57, 0.64]),
           st.floats(min_value=9.621, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.116, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([11.45, 12.64, 12.67, 12.86, 12.9, 12.94, 13.24, 13.4, 13.7, 14.86]),
           st.floats(min_value=3.866, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([71.76, 72.4, 72.49, 72.65, 72.83, 72.84, 73.21, 73.26, 73.55, 74.45]),
           st.sampled_from([0.0, 0.06, 0.07, 0.19, 0.39, 0.44, 0.49, 0.62, 0.64, 0.73]),
           st.floats(min_value=5.43, max_value=8.593, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.08, 0.15, 0.17, 0.19, 0.22, 0.28, 0.32, 0.34, 0.35]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.16, 13.04, 13.14, 13.24, 13.33, 13.41, 13.5, 13.64, 14.19, 14.32]),
           st.floats(min_value=3.866, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.29, max_value=0.964, allow_nan=False),
           st.sampled_from([71.36, 71.5, 71.79, 72.04, 72.14, 72.67, 72.69, 72.7, 72.89, 73.0]),
           st.sampled_from([0.0, 0.11, 0.16, 0.23, 0.52, 0.56, 0.57, 0.58, 0.59, 0.6]),
           st.floats(min_value=8.596, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.1, 0.17, 0.24, 0.37]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517071, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.sampled_from([12.57, 12.69, 12.71, 12.8, 13.05, 13.29, 13.31, 13.48, 13.81, 13.99]),
           st.floats(min_value=3.866, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.967, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([71.72, 71.76, 71.99, 72.01, 72.12, 72.32, 72.75, 72.79, 73.21, 73.7]),
           st.sampled_from([0.0, 0.06, 0.12, 0.18, 0.48, 0.51, 0.55, 0.57, 0.64, 0.67]),
           st.floats(min_value=8.596, max_value=10.479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.03, 0.06, 0.07, 0.09, 0.14, 0.17, 0.19, 0.22, 0.24, 0.31]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51593, 1.51596, 1.51605, 1.51645, 1.51687, 1.51707, 1.51708, 1.51829, 1.51847, 1.52222]),
           st.floats(min_value=10.73, max_value=14.494, allow_nan=False),
           st.sampled_from([1.35, 2.88, 3.09, 3.55, 3.58, 3.61, 3.63, 3.64, 3.9, 3.98]),
           st.floats(min_value=0.29, max_value=1.378, allow_nan=False),
           st.sampled_from([71.15, 71.81, 72.33, 72.45, 72.52, 72.66, 72.88, 73.01, 73.1, 73.21]),
           st.sampled_from([0.0, 0.07, 0.16, 0.33, 0.37, 0.45, 0.6, 0.63, 0.67, 0.68]),
           st.floats(min_value=10.482, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.14, 0.18, 0.2, 0.22, 0.25, 0.29, 0.32, 0.34]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51316, 1.51321, 1.51666, 1.51969, 1.51994, 1.52043, 1.52058, 1.52119, 1.52151, 1.52171]),
           st.floats(min_value=10.73, max_value=14.494, allow_nan=False),
           st.sampled_from([0.0, 0.33, 1.61, 1.71, 1.85, 1.88, 2.68]),
           st.floats(min_value=1.381, max_value=1.418, exclude_min=True, allow_nan=False),
           st.sampled_from([70.7, 72.18, 72.25, 72.69, 72.86, 73.03, 73.39, 73.44, 73.75, 73.88]),
           st.sampled_from([0.13, 0.32, 0.33, 0.47, 0.58, 0.6, 0.76, 0.97, 1.68, 6.21]),
           st.floats(min_value=10.482, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.28, 0.51]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51115, 1.51299, 1.51829, 1.51852, 1.51888, 1.51905, 1.51916, 1.51937, 1.51969]),
           st.floats(min_value=14.497, max_value=17.38, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.78, 1.74, 2.19, 2.24, 2.39, 2.41]),
           st.floats(min_value=0.29, max_value=1.418, allow_nan=False),
           st.sampled_from([72.37, 72.38, 72.5, 72.67, 72.74, 72.76, 73.48, 74.55, 75.41]),
           st.sampled_from([0.0]),
           st.floats(min_value=10.482, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [6]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.524218, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.493, allow_nan=False),
           st.floats(min_value=0.0, max_value=2.258, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([70.48, 70.7, 72.18, 72.25, 72.69, 72.86, 73.03, 73.44, 73.75, 73.88]),
           st.sampled_from([0.13, 0.32, 0.33, 0.38, 0.47, 0.58, 0.76, 0.97, 1.68, 6.21]),
           st.sampled_from([5.87, 6.93, 6.96, 9.7, 10.09, 10.17, 11.27, 11.32, 12.24, 12.5]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.28, 0.51]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.524221, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.493, allow_nan=False),
           st.floats(min_value=0.0, max_value=2.258, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([69.81, 72.02, 72.26, 72.45, 72.72, 72.75, 72.92, 73.01, 73.1, 73.14]),
           st.sampled_from([0.16, 0.45, 0.53, 0.54, 0.61, 0.65, 0.69, 0.7, 0.72, 1.1]),
           st.sampled_from([7.9, 8.21, 8.38, 8.39, 8.52, 8.9, 9.42, 11.52, 13.44, 16.19]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.08, 0.09, 0.1, 0.18, 0.19, 0.22, 0.24, 0.34, 0.35]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51115, 1.51299, 1.51829, 1.51852, 1.51888, 1.51905, 1.51916, 1.51937, 1.51969]),
           st.floats(min_value=13.496, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=2.258, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([72.37, 72.38, 72.5, 72.67, 72.74, 72.76, 73.48, 74.55, 75.41]),
           st.floats(min_value=0.0, max_value=0.193, allow_nan=False),
           st.sampled_from([6.65, 7.59, 9.26, 9.32, 9.57, 9.77, 9.95, 10.88, 11.22]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [6]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.521573, allow_nan=False),
           st.floats(min_value=13.496, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=2.258, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([72.06, 72.26, 72.38, 72.39, 72.67, 72.96, 72.97, 73.07, 73.14, 73.21]),
           st.floats(min_value=0.196, max_value=6.21, exclude_min=True, allow_nan=False),
           st.sampled_from([8.05, 8.21, 8.28, 8.43, 8.44, 8.9, 8.96, 9.13, 10.99, 16.19]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.08, 0.09, 0.1, 0.14, 0.17, 0.18, 0.2, 0.28, 0.29, 0.34]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.521576, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.496, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=2.258, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.25, 72.38, 72.61, 72.81, 73.02, 73.28, 73.36, 73.46, 73.72, 75.18]),
           st.floats(min_value=0.196, max_value=6.21, exclude_min=True, allow_nan=False),
           st.sampled_from([5.43, 8.28, 8.4, 8.48, 8.76, 8.83, 8.93, 9.18, 9.41, 9.45]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.01, 0.05, 0.07, 0.08, 0.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [7]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.515959, allow_nan=False),
           st.sampled_from([10.73, 13.23, 13.3, 13.41, 13.44, 13.46, 13.7, 13.72, 13.78, 14.43]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([70.16, 72.26, 72.34, 72.49, 72.72, 72.88, 73.01, 73.23, 73.27, 73.36]),
           st.sampled_from([0.06, 0.33, 0.49, 0.51, 0.53, 0.57, 0.66, 0.7, 0.72, 0.73]),
           st.floats(min_value=5.43, max_value=7.804, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.228, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.515962, max_value=1.517323, exclude_min=True, allow_nan=False),
           st.sampled_from([12.61, 12.8, 12.87, 13.2, 13.38, 13.48, 13.49, 13.73, 13.9, 13.99]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.57, 71.72, 71.81, 72.08, 72.72, 72.96, 72.97, 73.11, 73.21, 73.7]),
           st.sampled_from([0.14, 0.17, 0.18, 0.19, 0.23, 0.39, 0.48, 0.55, 0.59, 0.67]),
           st.floats(min_value=5.43, max_value=7.804, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.228, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.517323, allow_nan=False),
           st.sampled_from([12.55, 12.75, 12.79, 12.87, 12.89, 12.94, 13.12, 13.2, 13.43, 13.75]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.81, max_value=72.499, allow_nan=False),
           st.sampled_from([0.06, 0.1, 0.33, 0.35, 0.57, 0.6, 0.61, 0.7, 0.81, 1.1]),
           st.floats(min_value=7.807, max_value=8.203, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.228, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.517323, allow_nan=False),
           st.sampled_from([12.16, 12.86, 13.04, 13.14, 13.24, 13.33, 13.5, 13.53, 13.64, 14.19]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.81, max_value=72.499, allow_nan=False),
           st.sampled_from([0.0, 0.11, 0.16, 0.23, 0.52, 0.56, 0.57, 0.58, 0.59, 0.61]),
           st.floats(min_value=8.206, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.228, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.517323, allow_nan=False),
           st.sampled_from([11.02, 12.3, 12.75, 12.85, 13.01, 13.1, 13.2, 13.25, 13.43, 13.8]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.501, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.08, 0.33, 0.38, 0.44, 0.6, 0.63, 0.64, 0.72, 0.73, 1.1]),
           st.floats(min_value=7.807, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.228, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.517323, allow_nan=False),
           st.floats(min_value=10.73, max_value=12.819, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.57, 71.77, 72.08, 72.61, 72.72, 72.86, 72.97, 72.98, 73.09, 73.27]),
           st.sampled_from([0.11, 0.14, 0.15, 0.18, 0.19, 0.23, 0.5, 0.57, 0.61, 0.65]),
           st.sampled_from([8.09, 8.39, 8.55, 8.56, 8.76, 9.0, 9.02, 9.03, 9.14, 10.02]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.231, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51115, max_value=1.517323, allow_nan=False),
           st.floats(min_value=12.822, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.24, 71.81, 71.96, 71.99, 72.19, 72.66, 72.86, 72.88, 73.12, 73.25]),
           st.sampled_from([0.12, 0.33, 0.38, 0.56, 0.61, 0.63, 0.64, 0.72, 0.81, 1.1]),
           st.sampled_from([7.83, 7.9, 8.05, 8.18, 8.24, 8.27, 8.38, 8.44, 8.9, 8.96]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.floats(min_value=0.231, max_value=0.51, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517326, max_value=1.517719, exclude_min=True, allow_nan=False),
           st.sampled_from([12.61, 12.68, 13.08, 13.14, 13.27, 13.48, 13.51, 13.72, 14.21, 14.77]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.76, 71.81, 72.08, 72.85, 72.95, 73.24, 73.28, 73.29, 73.39, 73.7]),
           st.sampled_from([0.09, 0.12, 0.13, 0.15, 0.19, 0.48, 0.55, 0.56, 0.6, 0.62]),
           st.sampled_from([8.07, 8.22, 8.3, 8.38, 8.52, 8.57, 8.67, 8.76, 8.83, 10.17]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.03, 0.06, 0.07, 0.1, 0.16, 0.17, 0.22, 0.24, 0.3]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517722, max_value=1.517978, exclude_min=True, allow_nan=False),
           st.sampled_from([12.16, 13.04, 13.33, 13.42, 13.53, 13.64, 13.65, 14.03, 14.19, 14.32]),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.36, 72.04, 72.14, 72.48, 72.61, 72.65, 72.77, 72.89, 73.0, 73.01]),
           st.sampled_from([0.0, 0.06, 0.11, 0.16, 0.23, 0.52, 0.57, 0.58, 0.6, 0.61]),
           st.sampled_from([8.32, 8.33, 8.38, 8.53, 8.79, 8.89, 8.93, 8.99, 9.14, 9.65]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.1, 0.17, 0.24, 0.37]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517981, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.888, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([70.16, 71.24, 72.19, 72.44, 72.51, 72.55, 72.66, 72.81, 73.01, 73.23]),
           st.sampled_from([0.07, 0.12, 0.16, 0.19, 0.33, 0.52, 0.53, 0.6, 0.63, 0.81]),
           st.floats(min_value=5.43, max_value=8.888, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.1, 0.12, 0.14, 0.15, 0.19, 0.2, 0.24, 0.32, 0.35]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517981, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.888, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([71.94, 72.14, 72.48, 72.61, 72.64, 72.65, 72.67, 72.69, 72.7, 72.77]),
           st.sampled_from([0.0, 0.11, 0.16, 0.23, 0.52, 0.57, 0.58, 0.59, 0.6, 0.61]),
           st.floats(min_value=8.891, max_value=9.419, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0, 0.09, 0.1, 0.17, 0.24, 0.37]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517981, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.73, max_value=13.888, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([70.16, 71.24, 72.18, 72.26, 72.52, 72.55, 72.81, 72.88, 73.08, 73.1]),
           st.sampled_from([0.06, 0.1, 0.19, 0.37, 0.45, 0.52, 0.58, 0.63, 0.67, 0.73]),
           st.floats(min_value=9.422, max_value=16.19, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.1, 0.14, 0.17, 0.18, 0.19, 0.21, 0.22, 0.25, 0.34, 0.35]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.517981, max_value=1.53393, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.891, max_value=17.38, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.261, max_value=4.49, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.421, max_value=3.5, exclude_min=True, allow_nan=False),
           st.sampled_from([72.37, 72.38, 72.5, 72.67, 72.74, 72.76, 73.48, 74.55, 75.41]),
           st.sampled_from([0.0]),
           st.sampled_from([6.65, 7.59, 9.26, 9.32, 9.57, 9.77, 9.95, 10.88, 11.22]),
           st.floats(min_value=0.0, max_value=0.334, allow_nan=False),
           st.sampled_from([0.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [6]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51594, 1.51596, 1.51629, 1.5166, 1.51663, 1.51707, 1.51851, 1.51872, 1.52068, 1.53125]),
           st.sampled_from([11.45, 12.35, 12.55, 12.62, 12.87, 13.1, 13.25, 13.36, 13.98, 14.25]),
           st.floats(min_value=0.0, max_value=1.339, allow_nan=False),
           st.sampled_from([0.56, 1.06, 1.08, 1.17, 1.25, 1.3, 1.48, 1.54, 1.58, 1.67]),
           st.floats(min_value=69.81, max_value=70.158, allow_nan=False),
           st.sampled_from([0.12, 0.35, 0.37, 0.44, 0.53, 0.57, 0.61, 0.67, 0.69, 1.1]),
           st.sampled_from([7.83, 8.38, 8.54, 8.81, 8.9, 9.13, 10.99, 13.3, 14.4, 14.96]),
           st.floats(min_value=0.337, max_value=3.15, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.08, 0.12, 0.14, 0.18, 0.19, 0.21, 0.24, 0.28, 0.29]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51316, 1.51321, 1.51514, 1.51666, 1.51915, 1.51969, 1.52058, 1.52119, 1.52151, 1.52369]),
           st.sampled_from([11.03, 11.56, 12.64, 12.86, 12.97, 13.0, 13.02, 13.27, 13.44, 14.01]),
           st.floats(min_value=1.342, max_value=4.49, exclude_min=True, allow_nan=False),
           st.sampled_from([1.4, 1.51, 1.56, 1.65, 1.76, 1.83, 1.86, 2.17, 3.02, 3.04]),
           st.floats(min_value=69.81, max_value=70.158, allow_nan=False),
           st.sampled_from([0.13, 0.32, 0.33, 0.38, 0.47, 0.6, 0.76, 0.97, 1.68, 6.21]),
           st.sampled_from([6.93, 6.96, 9.7, 10.09, 10.17, 11.32, 11.41, 11.62, 12.24, 12.5]),
           st.floats(min_value=0.337, max_value=3.15, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.28, 0.51]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51514, 1.51545, 1.51602, 1.51623, 1.5164, 1.51645, 1.51711, 1.51831, 1.51838, 1.52065]),
           st.sampled_from([13.69, 13.88, 14.23, 14.32, 14.36, 14.38, 14.56, 14.92, 14.94, 15.01]),
           st.floats(min_value=0.0, max_value=3.418, allow_nan=False),
           st.sampled_from([1.23, 1.31, 1.8, 1.87, 2.02, 2.06, 2.34, 2.42, 2.51, 2.54]),
           st.floats(min_value=70.161, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.04, 0.05, 0.08, 0.31, 0.6, 0.76, 1.41, 1.46, 2.7]),
           st.sampled_from([5.43, 8.28, 8.4, 8.44, 8.53, 8.62, 8.76, 9.07, 9.08, 9.18]),
           st.floats(min_value=0.337, max_value=3.15, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.01, 0.05, 0.07, 0.08, 0.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [7]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.51743, 1.51756, 1.51768, 1.51769, 1.51775, 1.51783, 1.51837, 1.51909, 1.51911, 1.51926]),
           st.sampled_from([12.71, 12.84, 13.0, 13.2, 13.29, 13.3, 13.58, 13.69, 13.99, 14.77]),
           st.floats(min_value=3.421, max_value=4.49, exclude_min=True, allow_nan=False),
           st.sampled_from([0.78, 0.82, 1.23, 1.27, 1.3, 1.32, 1.33, 1.35, 1.49, 1.54]),
           st.floats(min_value=70.161, max_value=75.41, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.06, 0.13, 0.15, 0.23, 0.5, 0.55, 0.57, 0.61, 0.64]),
           st.sampled_from([8.03, 8.22, 8.27, 8.39, 8.44, 8.57, 9.0, 9.15, 9.82, 10.17]),
           st.floats(min_value=0.337, max_value=3.15, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.06, 0.07, 0.09, 0.11, 0.16, 0.17, 0.22, 0.26, 0.3]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
