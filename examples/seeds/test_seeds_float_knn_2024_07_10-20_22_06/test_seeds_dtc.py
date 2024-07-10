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
    request.cls.data['n_test'] = 16
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'test_seeds_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestSeedsProperty:

    @given(st.floats(min_value=10.59, max_value=13.408, allow_nan=False),
           st.sampled_from([14.06, 14.09, 14.16, 14.37, 14.39, 14.4, 14.41, 14.76, 15.0, 15.46]),
           st.sampled_from([0.8658, 0.8716, 0.8728, 0.8779, 0.8811, 0.8819, 0.8871, 0.8883, 0.8951, 0.9034]),
           st.sampled_from([4.902, 5.076, 5.348, 5.388, 5.454, 5.482, 5.585, 5.832, 5.833, 5.884]),
           st.floats(min_value=2.63, max_value=3.1268, allow_nan=False),
           st.sampled_from([1.313, 1.464, 2.269, 2.462, 3.136, 3.531, 3.975, 4.102, 4.185, 4.711]),
           st.floats(min_value=4.519, max_value=4.7884, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.59, max_value=12.913, allow_nan=False),
           st.sampled_from([12.82, 12.87, 12.93, 12.95, 13.1, 13.12, 13.13, 13.27, 13.53, 13.6]),
           st.sampled_from([0.8263, 0.8266, 0.8511, 0.856, 0.859, 0.8594, 0.862, 0.8652, 0.8793, 0.8883]),
           st.sampled_from([4.899, 4.984, 5.053, 5.18, 5.224, 5.243, 5.278, 5.325, 5.333, 5.541]),
           st.floats(min_value=3.1271, max_value=4.033, exclude_min=True, allow_nan=False),
           st.sampled_from([2.3, 3.082, 3.306, 3.332, 3.521, 3.985, 4.337, 4.957, 5.209, 6.735]),
           st.floats(min_value=4.519, max_value=4.7884, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.916, max_value=13.408, exclude_min=True, allow_nan=False),
           st.sampled_from([13.75, 13.82, 13.83, 14.1, 14.17, 14.21, 14.28, 15.16, 15.25, 15.46]),
           st.sampled_from([0.8458, 0.8538, 0.8657, 0.8664, 0.8724, 0.8728, 0.8794, 0.8955, 0.9058, 0.9153]),
           st.sampled_from([5.008, 5.138, 5.504, 5.545, 5.554, 5.678, 5.701, 5.709, 5.833, 5.884]),
           st.floats(min_value=3.1271, max_value=4.033, exclude_min=True, allow_nan=False),
           st.sampled_from([1.415, 1.502, 1.717, 1.767, 2.04, 2.461, 2.7, 2.823, 2.958, 3.92]),
           st.floats(min_value=4.519, max_value=4.7884, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.59, max_value=12.709, allow_nan=False),
           st.sampled_from([12.86, 13.55, 13.75, 13.82, 13.85, 14.04, 14.09, 14.57, 14.77, 15.0]),
           st.sampled_from([0.8458, 0.8564, 0.8726, 0.8759, 0.8796, 0.8819, 0.888, 0.8911, 0.8955, 0.9]),
           st.sampled_from([5.386, 5.397, 5.412, 5.579, 5.656, 5.715, 5.741, 5.789, 5.826, 5.832]),
           st.sampled_from([2.936, 3.113, 3.158, 3.168, 3.19, 3.258, 3.298, 3.302, 3.393, 3.434]),
           st.floats(min_value=0.7651, max_value=1.53798, allow_nan=False),
           st.floats(min_value=4.7887, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.59, max_value=12.709, allow_nan=False),
           st.sampled_from([12.82, 12.83, 12.86, 13.04, 13.13, 13.2, 13.22, 13.36, 13.59, 13.92]),
           st.sampled_from([0.8081, 0.8256, 0.8291, 0.8419, 0.8481, 0.8563, 0.8575, 0.8596, 0.8795, 0.8977]),
           st.sampled_from([5.053, 5.073, 5.089, 5.175, 5.176, 5.186, 5.204, 5.279, 5.35, 5.363]),
           st.sampled_from([2.695, 2.787, 2.85, 2.897, 2.911, 2.967, 2.974, 2.981, 3.074, 3.135]),
           st.floats(min_value=1.53801, max_value=8.456, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.7887, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.712, max_value=13.408, exclude_min=True, allow_nan=False),
           st.sampled_from([13.23, 13.5, 13.57, 13.83, 14.52, 14.76, 14.84, 14.91, 15.16, 15.46]),
           st.sampled_from([0.8392, 0.8604, 0.8696, 0.8716, 0.8726, 0.8744, 0.8796, 0.8823, 0.8986, 0.9]),
           st.sampled_from([5.439, 5.479, 5.482, 5.504, 5.516, 5.541, 5.569, 5.674, 5.712, 5.826]),
           st.floats(min_value=2.63, max_value=3.1123, allow_nan=False),
           st.floats(min_value=0.7651, max_value=4.41349, allow_nan=False),
           st.floats(min_value=4.7887, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.712, max_value=13.408, exclude_min=True, allow_nan=False),
           st.sampled_from([12.79, 12.83, 12.97, 13.02, 13.04, 13.32, 13.38, 13.4, 13.52, 13.95]),
           st.sampled_from([0.8082, 0.8274, 0.8419, 0.848, 0.8491, 0.8541, 0.8594, 0.862, 0.8652, 0.8684]),
           st.sampled_from([5.009, 5.18, 5.204, 5.243, 5.263, 5.278, 5.32, 5.389, 5.408, 5.413]),
           st.floats(min_value=3.1126, max_value=4.033, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7651, max_value=4.41349, allow_nan=False),
           st.floats(min_value=4.7887, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.712, max_value=13.408, exclude_min=True, allow_nan=False),
           st.sampled_from([12.82, 12.88, 13.05, 13.15, 13.32, 13.38, 13.41, 13.46, 13.71, 13.94]),
           st.sampled_from([0.8081, 0.8253, 0.8455, 0.8473, 0.856, 0.8579, 0.8596, 0.8783, 0.8874, 0.8977]),
           st.sampled_from([5.009, 5.046, 5.105, 5.18, 5.219, 5.224, 5.25, 5.357, 5.386, 5.413]),
           st.sampled_from([2.648, 2.719, 2.763, 2.794, 2.821, 2.845, 2.849, 2.941, 2.974, 3.126]),
           st.floats(min_value=4.41352, max_value=8.456, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.7887, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.411, max_value=21.18, exclude_min=True, allow_nan=False),
           st.sampled_from([12.63, 12.86, 13.83, 13.85, 14.16, 14.17, 14.41, 14.43, 14.75, 15.46]),
           st.sampled_from([0.8458, 0.8751, 0.8779, 0.8796, 0.8799, 0.8811, 0.8819, 0.8852, 0.8944, 0.9153]),
           st.sampled_from([4.902, 5.384, 5.42, 5.658, 5.662, 5.712, 5.757, 5.789, 5.832, 5.833]),
           st.floats(min_value=2.63, max_value=3.5719, allow_nan=False),
           st.sampled_from([0.8551, 1.464, 2.04, 2.699, 2.823, 2.956, 3.128, 3.328, 4.185, 5.234]),
           st.floats(min_value=4.519, max_value=5.4174, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.411, max_value=21.18, exclude_min=True, allow_nan=False),
           st.sampled_from([15.25, 15.34, 15.76, 16.17, 16.19, 16.23, 16.32, 16.59, 16.61, 16.72]),
           st.sampled_from([0.859, 0.875, 0.8786, 0.88, 0.881, 0.8865, 0.8942, 0.8977, 0.8991, 0.8993]),
           st.floats(min_value=4.899, max_value=5.5894, allow_nan=False),
           st.floats(min_value=2.63, max_value=3.5719, allow_nan=False),
           st.sampled_from([2.068, 2.235, 2.257, 2.725, 2.858, 3.237, 3.357, 3.691, 4.675, 6.682]),
           st.floats(min_value=5.4177, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.411, max_value=21.18, exclude_min=True, allow_nan=False),
           st.sampled_from([13.67, 14.04, 14.61, 14.76, 14.77, 14.85, 14.9, 14.94, 14.99, 15.46]),
           st.sampled_from([0.8458, 0.8538, 0.8662, 0.871, 0.8716, 0.8734, 0.8744, 0.8993, 0.9034, 0.9153]),
           st.floats(min_value=5.5897, max_value=6.675, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.63, max_value=3.5719, allow_nan=False),
           st.sampled_from([1.355, 1.767, 2.04, 2.249, 2.688, 2.7, 2.704, 2.956, 4.116, 5.234]),
           st.floats(min_value=5.4177, max_value=5.5754, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.411, max_value=16.533, exclude_min=True, allow_nan=False),
           st.sampled_from([15.33, 15.73, 15.85, 15.97, 16.31, 16.5, 16.52, 16.57, 16.59, 17.03]),
           st.sampled_from([0.858, 0.8599, 0.8623, 0.8637, 0.8644, 0.8746, 0.875, 0.88, 0.885, 0.8985]),
           st.sampled_from([5.363, 5.477, 5.718, 5.89, 6.084, 6.245, 6.341, 6.369, 6.384, 6.675]),
           st.floats(min_value=3.5722, max_value=4.033, exclude_min=True, allow_nan=False),
           st.sampled_from([1.738, 2.188, 2.843, 2.962, 3.252, 3.357, 3.477, 3.824, 4.539, 5.532]),
           st.floats(min_value=4.519, max_value=5.5754, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=16.536, max_value=21.18, exclude_min=True, allow_nan=False),
           st.sampled_from([12.63, 13.84, 13.85, 14.06, 14.1, 14.18, 14.28, 14.6, 14.77, 15.0]),
           st.sampled_from([0.8604, 0.8641, 0.868, 0.8724, 0.8728, 0.8779, 0.8811, 0.884, 0.888, 0.9034]),
           st.sampled_from([5.099, 5.119, 5.351, 5.386, 5.388, 5.439, 5.554, 5.563, 5.656, 5.712]),
           st.floats(min_value=3.5722, max_value=4.033, exclude_min=True, allow_nan=False),
           st.sampled_from([0.903, 2.129, 2.259, 2.688, 2.932, 3.373, 3.975, 4.116, 4.157, 4.185]),
           st.floats(min_value=4.519, max_value=5.5754, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.23, 12.74, 13.84, 14.43, 14.46, 14.49, 14.7, 15.36, 16.12, 16.2]),
           st.sampled_from([14.29, 14.39, 14.43, 14.6, 14.68, 14.75, 14.85, 14.94, 15.0, 15.46]),
           st.sampled_from([0.8529, 0.8538, 0.8658, 0.8722, 0.8724, 0.8728, 0.8779, 0.8857, 0.9, 0.905]),
           st.sampled_from([5.008, 5.076, 5.226, 5.439, 5.516, 5.527, 5.702, 5.709, 5.717, 5.832]),
           st.sampled_from([2.879, 2.956, 2.975, 3.199, 3.272, 3.288, 3.298, 3.383, 3.434, 3.562]),
           st.floats(min_value=0.7651, max_value=2.05399, allow_nan=False),
           st.floats(min_value=5.5757, max_value=5.8853, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([15.99, 17.26, 18.14, 18.27, 18.81, 18.85, 18.95, 19.06, 20.03, 20.97]),
           st.sampled_from([14.9, 15.15, 16.09, 16.12, 16.17, 16.22, 16.32, 16.89, 16.91, 16.92]),
           st.sampled_from([0.8673, 0.8717, 0.8722, 0.878, 0.885, 0.8921, 0.8969, 0.8989, 0.9056, 0.9081]),
           st.sampled_from([5.832, 5.884, 5.92, 5.979, 6.173, 6.271, 6.303, 6.384, 6.445, 6.449]),
           st.sampled_from([3.387, 3.408, 3.465, 3.684, 3.764, 3.769, 3.785, 3.825, 3.864, 3.991]),
           st.floats(min_value=0.7651, max_value=2.05399, allow_nan=False),
           st.floats(min_value=5.8856, max_value=6.55, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([15.56, 17.26, 18.14, 18.59, 18.75, 18.81, 18.85, 18.94, 18.96, 19.46]),
           st.sampled_from([14.9, 15.33, 16.18, 16.31, 16.34, 16.49, 16.57, 16.61, 16.9, 17.23]),
           st.sampled_from([0.8452, 0.8644, 0.8648, 0.8687, 0.889, 0.8894, 0.8985, 0.8991, 0.8993, 0.9066]),
           st.sampled_from([5.791, 5.884, 5.89, 5.92, 5.927, 6.006, 6.037, 6.384, 6.549, 6.675]),
           st.sampled_from([3.408, 3.467, 3.485, 3.486, 3.566, 3.693, 3.719, 3.785, 3.825, 3.93]),
           st.floats(min_value=2.05402, max_value=8.456, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.5757, max_value=6.55, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
