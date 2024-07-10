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
    request.cls.data['n_test'] = 521
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_phoneme_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestPhonemeProperty:

    @given(st.floats(min_value=-1.7, max_value=-0.2006, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-1.0651, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2901, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.2006, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-1.0651, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2898, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.2003, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-1.0651, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=-0.2067, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.0648, max_value=1.1964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-0.2064, max_value=-0.1107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=-0.4152, allow_nan=False),
           st.floats(min_value=-1.0648, max_value=1.1964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7292, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.2064, max_value=-0.1107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4149, max_value=0.2174, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.0648, max_value=1.1964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7292, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-0.2064, max_value=-0.1107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.0648, max_value=1.1964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7289, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1104, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=-1.0648, max_value=0.4293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1104, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=0.4296, max_value=0.5323, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1104, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=0.5326, max_value=1.1964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.2174, allow_nan=False),
           st.floats(min_value=1.1967, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.2177, max_value=0.4183, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.2177, max_value=0.4183, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7567, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.4493, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-1.1971, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.4493, allow_nan=False),
           st.floats(min_value=-1.1968, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.4569, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4496, max_value=0.6894, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4572, max_value=0.5729, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4496, max_value=0.6894, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.5732, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4496, max_value=0.6894, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.6614, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7747, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3992, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.6617, max_value=0.6809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7747, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3992, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.6809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7747, allow_nan=False),
           st.floats(min_value=-0.3989, max_value=0.1619, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.6812, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.7747, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.1619, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.5019, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7744, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3641, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.5019, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7744, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3638, max_value=0.1619, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.5022, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7744, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.1619, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.4186, max_value=0.8024, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6897, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.1622, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.0361, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.7723, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6107, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4241, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.0361, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.7723, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6107, allow_nan=False),
           st.floats(min_value=-0.4238, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.0358, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.5244, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6107, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.0358, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5247, max_value=2.5289, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6107, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.0358, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5292, max_value=2.7723, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6107, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.3459, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.7723, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3231, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3462, max_value=0.4793, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.7723, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3231, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9598, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.3633, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3228, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9598, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3636, max_value=1.4788, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3228, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9598, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4791, max_value=2.7723, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3228, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.9601, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.7723, allow_nan=False),
           st.floats(min_value=-0.6104, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3228, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.4793, allow_nan=False),
           st.floats(min_value=0.8027, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7726, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=-0.2167, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.3348, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=-0.2167, allow_nan=False),
           st.floats(min_value=1.3351, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3312, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=-0.2167, allow_nan=False),
           st.floats(min_value=1.3351, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.3309, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2164, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.sampled_from([0.522, 0.575, 0.637, 0.904, 1.022, 1.058, 1.717, 1.834, 1.865, 2.371]),
           st.floats(min_value=-1.581, max_value=-0.6676, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3437, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=0.5224, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2164, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.041, 1.023, 1.057, 1.342, 1.763, 1.936, 2.136, 2.137, 2.617, 2.694]),
           st.floats(min_value=-1.581, max_value=-0.6676, allow_nan=False),
           st.floats(min_value=-0.3434, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.5227, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2164, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.316, -0.275, 0.198, 0.271, 0.578, 1.185, 1.672, 2.335, 2.516, 2.875]),
           st.floats(min_value=-1.581, max_value=-0.6676, allow_nan=False),
           st.floats(min_value=-0.3434, max_value=0.3343, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4796, max_value=1.5754, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2164, max_value=0.9664, exclude_min=True, allow_nan=False),
           st.sampled_from([0.665, 0.811, 0.848, 0.892, 1.066, 1.341, 1.554, 2.371, 2.416, 2.523]),
           st.floats(min_value=-0.6673, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.5757, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.sampled_from([-0.406, -0.383, -0.287, -0.106, 0.131, 0.34, 0.502, 0.931, 1.616, 1.637]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3343, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.121, 0.437, 0.474, 0.483, 1.638, 2.003, 2.409, 2.416, 2.444, 3.056]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.8734, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.4502, allow_nan=False),
           st.floats(min_value=0.3346, max_value=1.0328, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.146, 0.188, 0.212, 0.22, 0.333, 0.35, 0.479, 0.582, 0.593, 0.959]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.5403, allow_nan=False),
           st.floats(min_value=-0.4499, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3346, max_value=1.0328, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.126, 0.203, 0.229, 0.339, 0.368, 0.597, 0.916, 1.312, 1.929, 3.026]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.5406, max_value=0.8734, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4499, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3346, max_value=1.0328, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.072, 0.379, 0.418, 0.464, 0.604, 0.939, 0.961, 1.062, 1.167, 1.232]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.8734, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.3571, allow_nan=False),
           st.floats(min_value=1.0331, max_value=1.5019, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.175, 0.223, 0.415, 0.929, 1.109, 1.122, 1.149, 1.661, 1.851, 2.64]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.8734, allow_nan=False),
           st.floats(min_value=-0.3568, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0331, max_value=1.5019, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.185, 0.399, 0.51, 0.639, 0.89, 1.383, 1.84, 2.164, 2.329, 3.183]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.4223, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=1.5022, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.152, 0.156, 0.209, 0.233, 0.238, 0.345, 0.47, 0.502, 0.577, 1.21]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.4226, max_value=0.4368, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=1.5022, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.101, 0.292, 0.528, 0.579, 0.598, 1.051, 1.254, 1.408, 1.611, 3.05]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.4371, max_value=0.8734, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=1.5022, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.106, 0.161, 0.505, 0.861, 1.163, 1.49, 1.534, 1.799, 2.971, 3.348]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.8737, max_value=2.4883, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6506, allow_nan=False),
           st.floats(min_value=0.3346, max_value=0.4094, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2284, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.8737, max_value=2.4883, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6506, allow_nan=False),
           st.floats(min_value=0.4097, max_value=0.4368, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2287, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.8737, max_value=2.4883, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.6506, allow_nan=False),
           st.floats(min_value=0.4097, max_value=0.4368, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.144, 0.077, 0.126, 0.36, 0.435, 0.557, 0.691, 0.744, 0.769, 0.911]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.8737, max_value=2.4883, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6503, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3346, max_value=0.4368, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.105, 0.061, 0.123, 0.142, 0.324, 0.406, 0.512, 0.838, 0.886, 1.083]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=0.8737, max_value=2.4883, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.4371, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.8973, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=2.4886, max_value=2.6153, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3346, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.8973, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=2.6156, max_value=2.6388, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3346, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.8973, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=2.6391, max_value=2.7949, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3346, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.8976, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=2.4886, max_value=2.7949, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3346, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.08, 0.122, 0.257, 0.5, 0.516, 0.554, 0.66, 0.838, 1.062, 1.103]),
           st.floats(min_value=-1.327, max_value=0.9664, allow_nan=False),
           st.floats(min_value=2.7952, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3346, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.3514, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.6168, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.5566, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.3514, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6171, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.5566, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2609, allow_nan=False),
           st.floats(min_value=0.9667, max_value=0.9764, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.215, 0.284, 0.364, 0.44, 0.564, 0.75, 1.185, 2.176, 2.177, 2.489]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5563, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2612, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=0.9764, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.153, 0.292, 0.58, 0.595, 0.711, 0.947, 1.716, 1.752, 2.141, 2.617]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5563, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2554, allow_nan=False),
           st.floats(min_value=0.9767, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.418, -0.378, 0.16, 0.39, 0.449, 0.678, 0.907, 1.006, 1.064, 2.694]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5563, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2557, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.164, 0.792, 0.951, 1.456, 1.464, 1.601, 1.81, 1.84, 2.179, 2.514]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5563, max_value=-0.5096, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2557, max_value=0.2603, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=1.1688, exclude_min=True, allow_nan=False),
           st.sampled_from([0.396, 0.7, 0.85, 0.89, 1.438, 1.586, 1.895, 1.991, 2.313, 2.831]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=-0.2546, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2557, max_value=0.2603, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1691, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([0.237, 0.265, 1.024, 1.335, 1.422, 1.595, 1.622, 1.695, 1.749, 1.841]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=-0.2546, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2557, max_value=0.2603, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.617, -0.221, -0.12, 0.133, 0.176, 0.603, 0.918, 1.702, 1.935, 2.555]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2543, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=1.1178, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.618, -0.243, 0.337, 0.744, 1.12, 1.125, 1.534, 1.73, 1.886, 2.573]),
           st.floats(min_value=-1.581, max_value=-0.6261, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1181, max_value=1.1203, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.401, 0.361, 0.686, 0.707, 0.942, 1.331, 1.56, 1.639, 1.647, 2.298]),
           st.floats(min_value=-1.581, max_value=-0.6261, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=1.1203, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.9993, allow_nan=False),
           st.floats(min_value=-0.6258, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9767, max_value=1.1203, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9996, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6258, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1206, max_value=1.3318, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.554, -0.357, -0.271, -0.257, -0.151, 0.449, 1.206, 1.291, 1.738, 2.417]),
           st.floats(min_value=-1.581, max_value=-0.6517, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1206, max_value=1.3318, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.747, 0.881, 1.066, 1.122, 1.462, 1.73, 1.826, 1.982, 2.13, 2.382]),
           st.floats(min_value=-0.6514, max_value=-0.6432, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1206, max_value=1.3318, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.092, 0.155, 0.431, 0.537, 0.619, 1.34, 1.429, 1.533, 2.157, 2.479]),
           st.floats(min_value=-0.6429, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3321, max_value=1.3383, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.0613, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3321, max_value=1.3383, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0616, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2606, max_value=0.3514, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3386, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.471, -0.421, -0.291, -0.194, 0.107, 0.333, 1.353, 1.604, 1.637, 2.185]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.5093, max_value=0.7128, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.3514, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.1569, exclude_min=True, allow_nan=False),
           st.sampled_from([0.747, 0.763, 0.832, 0.845, 0.904, 0.942, 1.446, 1.553, 1.566, 2.278]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.7131, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.3514, allow_nan=False),
           st.floats(min_value=1.1572, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.643, -0.635, -0.333, 0.468, 0.657, 0.724, 0.9, 1.887, 1.897, 2.17]),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.7131, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.9049, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.4457, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2177, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.9049, allow_nan=False),
           st.floats(min_value=-0.4454, max_value=-0.3982, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2177, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.9049, allow_nan=False),
           st.floats(min_value=-0.3979, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2177, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.6579, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2174, max_value=0.5733, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.0014, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6582, max_value=1.8093, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2174, max_value=0.5733, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.0014, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8096, max_value=1.9049, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2174, max_value=0.5733, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0017, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6582, max_value=1.9049, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2174, max_value=0.5733, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.9049, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.5736, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9052, max_value=2.3354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4917, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.5953, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9052, max_value=2.3354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.4914, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.5956, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9052, max_value=2.3354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.4914, max_value=-0.2486, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.5956, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9052, max_value=2.3354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2483, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.3563, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.0763, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3357, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.88, -0.436, -0.424, 0.536, 1.31, 1.316, 1.399, 1.467, 1.941, 2.116]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3566, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.0763, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3357, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.481, -0.397, -0.281, -0.25, -0.138, -0.135, 0.553, 0.731, 1.297, 1.838]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0766, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3357, max_value=2.5348, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.935, -0.406, -0.38, -0.324, 0.634, 0.647, 1.199, 1.516, 1.592, 1.862]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0766, max_value=1.4374, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5351, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.689, -0.45, -0.419, -0.404, -0.142, -0.085, 0.273, 0.365, 0.416, 0.49]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=1.9124, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.3832, allow_nan=False),
           st.sampled_from([-0.248, -0.225, -0.164, 0.099, 0.155, 0.159, 0.357, 1.049, 1.145, 1.379]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=1.9124, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-0.3829, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.1587, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=1.9124, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-0.3829, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1584, max_value=0.3119, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_104(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_104']['n_samples'] += 1
        self.data['tests']['test_104']['samples'].append(x_test)
        self.data['tests']['test_104']['y_expected'].append(y_expected[0])
        self.data['tests']['test_104']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=1.9124, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-0.3829, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3122, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9127, max_value=3.1014, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.853, -0.651, -0.529, -0.316, -0.201, 0.124, 0.18, 0.715, 0.843, 1.877]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_106(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_106']['n_samples'] += 1
        self.data['tests']['test_106']['samples'].append(x_test)
        self.data['tests']['test_106']['y_expected'].append(y_expected[0])
        self.data['tests']['test_106']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.1017, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.3856, allow_nan=False),
           st.sampled_from([-0.484, -0.408, -0.406, -0.167, -0.111, 0.537, 0.892, 1.1, 1.503, 1.779]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.1017, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.8893, allow_nan=False),
           st.floats(min_value=-0.3853, max_value=-0.2966, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.407, -0.401, -0.387, -0.283, -0.262, -0.209, -0.165, -0.047, 0.124, 1.117]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_108(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_108']['n_samples'] += 1
        self.data['tests']['test_108']['samples'].append(x_test)
        self.data['tests']['test_108']['y_expected'].append(y_expected[0])
        self.data['tests']['test_108']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.6113, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=2.3599, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3421, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_109(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_109']['n_samples'] += 1
        self.data['tests']['test_109']['samples'].append(x_test)
        self.data['tests']['test_109']['y_expected'].append(y_expected[0])
        self.data['tests']['test_109']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.6113, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=2.3599, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.3418, max_value=0.3884, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.6113, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=2.3599, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3887, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_111(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_111']['n_samples'] += 1
        self.data['tests']['test_111']['samples'].append(x_test)
        self.data['tests']['test_111']['y_expected'].append(y_expected[0])
        self.data['tests']['test_111']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6116, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4376, max_value=2.3599, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.279, -0.173, 0.478, 0.714, 0.892, 0.978, 0.991, 0.997, 1.358, 1.525]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_112(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_112']['n_samples'] += 1
        self.data['tests']['test_112']['samples'].append(x_test)
        self.data['tests']['test_112']['y_expected'].append(y_expected[0])
        self.data['tests']['test_112']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3517, max_value=0.7383, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3602, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.sampled_from([-0.974, -0.81, -0.804, -0.687, -0.451, -0.228, -0.162, 1.255, 1.286, 1.302]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_113(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_113']['n_samples'] += 1
        self.data['tests']['test_113']['samples'].append(x_test)
        self.data['tests']['test_113']['y_expected'].append(y_expected[0])
        self.data['tests']['test_113']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.9559, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3253, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_114(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_114']['n_samples'] += 1
        self.data['tests']['test_114']['samples'].append(x_test)
        self.data['tests']['test_114']['y_expected'].append(y_expected[0])
        self.data['tests']['test_114']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.5879, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9562, max_value=1.0309, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3253, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_115(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_115']['n_samples'] += 1
        self.data['tests']['test_115']['samples'].append(x_test)
        self.data['tests']['test_115']['y_expected'].append(y_expected[0])
        self.data['tests']['test_115']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.5882, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9562, max_value=1.0309, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3253, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_116(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_116']['n_samples'] += 1
        self.data['tests']['test_116']['samples'].append(x_test)
        self.data['tests']['test_116']['y_expected'].append(y_expected[0])
        self.data['tests']['test_116']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.0309, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.3256, max_value=0.5814, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_117(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_117']['n_samples'] += 1
        self.data['tests']['test_117']['samples'].append(x_test)
        self.data['tests']['test_117']['y_expected'].append(y_expected[0])
        self.data['tests']['test_117']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.0309, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.5817, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_118(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_118']['n_samples'] += 1
        self.data['tests']['test_118']['samples'].append(x_test)
        self.data['tests']['test_118']['y_expected'].append(y_expected[0])
        self.data['tests']['test_118']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2666, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_119(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_119']['n_samples'] += 1
        self.data['tests']['test_119']['samples'].append(x_test)
        self.data['tests']['test_119']['y_expected'].append(y_expected[0])
        self.data['tests']['test_119']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2663, max_value=-0.2091, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_120(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_120']['n_samples'] += 1
        self.data['tests']['test_120']['samples'].append(x_test)
        self.data['tests']['test_120']['y_expected'].append(y_expected[0])
        self.data['tests']['test_120']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=2.5354, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=1.7443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2088, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_121(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_121']['n_samples'] += 1
        self.data['tests']['test_121']['samples'].append(x_test)
        self.data['tests']['test_121']['y_expected'].append(y_expected[0])
        self.data['tests']['test_121']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=2.1118, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7446, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2088, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_122(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_122']['n_samples'] += 1
        self.data['tests']['test_122']['samples'].append(x_test)
        self.data['tests']['test_122']['y_expected'].append(y_expected[0])
        self.data['tests']['test_122']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1121, max_value=2.5354, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7446, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2088, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_123(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_123']['n_samples'] += 1
        self.data['tests']['test_123']['samples'].append(x_test)
        self.data['tests']['test_123']['y_expected'].append(y_expected[0])
        self.data['tests']['test_123']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=0.8184, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5357, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2088, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_124(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_124']['n_samples'] += 1
        self.data['tests']['test_124']['samples'].append(x_test)
        self.data['tests']['test_124']['y_expected'].append(y_expected[0])
        self.data['tests']['test_124']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.8187, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5357, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=1.8438, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2088, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_125(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_125']['n_samples'] += 1
        self.data['tests']['test_125']['samples'].append(x_test)
        self.data['tests']['test_125']['y_expected'].append(y_expected[0])
        self.data['tests']['test_125']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8441, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2131, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_126(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_126']['n_samples'] += 1
        self.data['tests']['test_126']['samples'].append(x_test)
        self.data['tests']['test_126']['y_expected'].append(y_expected[0])
        self.data['tests']['test_126']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=0.7583, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.8428, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8441, max_value=2.1609, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2128, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_127(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_127']['n_samples'] += 1
        self.data['tests']['test_127']['samples'].append(x_test)
        self.data['tests']['test_127']['y_expected'].append(y_expected[0])
        self.data['tests']['test_127']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7586, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=1.8428, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8441, max_value=2.1609, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2128, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_128(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_128']['n_samples'] += 1
        self.data['tests']['test_128']['samples'].append(x_test)
        self.data['tests']['test_128']['y_expected'].append(y_expected[0])
        self.data['tests']['test_128']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8431, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8441, max_value=2.0658, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2128, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_129(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_129']['n_samples'] += 1
        self.data['tests']['test_129']['samples'].append(x_test)
        self.data['tests']['test_129']['y_expected'].append(y_expected[0])
        self.data['tests']['test_129']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8431, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0661, max_value=2.1609, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2128, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_130(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_130']['n_samples'] += 1
        self.data['tests']['test_130']['samples'].append(x_test)
        self.data['tests']['test_130']['y_expected'].append(y_expected[0])
        self.data['tests']['test_130']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7386, max_value=1.6114, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1612, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-0.2128, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_131(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_131']['n_samples'] += 1
        self.data['tests']['test_131']['samples'].append(x_test)
        self.data['tests']['test_131']['y_expected'].append(y_expected[0])
        self.data['tests']['test_131']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6117, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.2623, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_132(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_132']['n_samples'] += 1
        self.data['tests']['test_132']['samples'].append(x_test)
        self.data['tests']['test_132']['y_expected'].append(y_expected[0])
        self.data['tests']['test_132']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6117, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9667, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0312, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.581, max_value=-0.2966, allow_nan=False),
           st.floats(min_value=0.2626, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_133(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_133']['n_samples'] += 1
        self.data['tests']['test_133']['samples'].append(x_test)
        self.data['tests']['test_133']['y_expected'].append(y_expected[0])
        self.data['tests']['test_133']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0964, allow_nan=False),
           st.sampled_from([-0.356, 0.45, 0.486, 0.851, 1.079, 1.281, 1.357, 1.421, 2.73, 3.04]),
           st.floats(min_value=-1.823, max_value=0.5809, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5624, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.446, -0.44, -0.384, -0.361, -0.171, 0.374, 0.703, 0.797, 1.321, 1.926]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_134(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_134']['n_samples'] += 1
        self.data['tests']['test_134']['samples'].append(x_test)
        self.data['tests']['test_134']['y_expected'].append(y_expected[0])
        self.data['tests']['test_134']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0964, allow_nan=False),
           st.sampled_from([0.342, 0.355, 0.505, 0.548, 0.624, 0.633, 0.77, 0.822, 1.633, 2.373]),
           st.floats(min_value=-1.823, max_value=0.5809, allow_nan=False),
           st.floats(min_value=0.5626, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.184, -0.784, -0.569, -0.537, -0.518, -0.394, -0.362, 0.619, 1.419, 1.486]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_135(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_135']['n_samples'] += 1
        self.data['tests']['test_135']['samples'].append(x_test)
        self.data['tests']['test_135']['y_expected'].append(y_expected[0])
        self.data['tests']['test_135']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0964, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3408, allow_nan=False),
           st.floats(min_value=0.5812, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.593, -0.407, -0.17, 0.543, 0.69, 0.913, 0.991, 1.171, 1.274, 1.783]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_136(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_136']['n_samples'] += 1
        self.data['tests']['test_136']['samples'].append(x_test)
        self.data['tests']['test_136']['y_expected'].append(y_expected[0])
        self.data['tests']['test_136']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0904, allow_nan=False),
           st.floats(min_value=0.3411, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5812, max_value=1.4568, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.634, -0.315, -0.266, -0.13, 0.18, 0.193, 0.275, 0.337, 0.445, 1.263]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_137(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_137']['n_samples'] += 1
        self.data['tests']['test_137']['samples'].append(x_test)
        self.data['tests']['test_137']['y_expected'].append(y_expected[0])
        self.data['tests']['test_137']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0904, allow_nan=False),
           st.floats(min_value=0.3411, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4571, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.064, -0.645, -0.541, -0.46, -0.272, -0.267, -0.264, 1.283, 1.31, 1.489]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_138(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_138']['n_samples'] += 1
        self.data['tests']['test_138']['samples'].append(x_test)
        self.data['tests']['test_138']['y_expected'].append(y_expected[0])
        self.data['tests']['test_138']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0907, max_value=0.0964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3411, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5812, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.849, -0.436, -0.144, -0.129, 0.666, 0.689, 0.723, 0.978, 1.499, 1.699]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_139(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_139']['n_samples'] += 1
        self.data['tests']['test_139']['samples'].append(x_test)
        self.data['tests']['test_139']['y_expected'].append(y_expected[0])
        self.data['tests']['test_139']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.1008, exclude_min=True, allow_nan=False),
           st.sampled_from([0.52, 0.968, 1.213, 1.395, 1.888, 2.515, 2.611, 2.712, 2.848, 2.972]),
           st.floats(min_value=-1.823, max_value=0.5303, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.671, -0.211, -0.195, 0.156, 0.214, 0.239, 0.272, 0.308, 1.15, 1.775]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_140(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_140']['n_samples'] += 1
        self.data['tests']['test_140']['samples'].append(x_test)
        self.data['tests']['test_140']['y_expected'].append(y_expected[0])
        self.data['tests']['test_140']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1011, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.599, 0.63, 0.742, 0.79, 1.249, 1.538, 1.542, 2.202, 2.258, 2.944]),
           st.floats(min_value=-1.823, max_value=0.5303, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.908, -0.756, -0.748, -0.74, -0.672, -0.397, -0.276, 0.964, 1.131, 1.867]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_141(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_141']['n_samples'] += 1
        self.data['tests']['test_141']['samples'].append(x_test)
        self.data['tests']['test_141']['y_expected'].append(y_expected[0])
        self.data['tests']['test_141']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.0974, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.5533, allow_nan=False),
           st.floats(min_value=0.5306, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.557, -0.516, 0.485, 0.794, 0.877, 0.892, 0.945, 0.958, 1.285, 1.676]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_142(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_142']['n_samples'] += 1
        self.data['tests']['test_142']['samples'].append(x_test)
        self.data['tests']['test_142']['y_expected'].append(y_expected[0])
        self.data['tests']['test_142']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0977, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4683, allow_nan=False),
           st.floats(min_value=0.5306, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.456, -0.246, 0.257, 0.815, 0.867, 1.11, 1.151, 1.814, 2.016]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_143(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_143']['n_samples'] += 1
        self.data['tests']['test_143']['samples'].append(x_test)
        self.data['tests']['test_143']['y_expected'].append(y_expected[0])
        self.data['tests']['test_143']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0977, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4686, max_value=0.5533, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5306, max_value=0.7959, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.768, -0.728, -0.394, -0.284, -0.24, 0.612, 0.776, 0.786, 1.157, 1.723]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_144(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_144']['n_samples'] += 1
        self.data['tests']['test_144']['samples'].append(x_test)
        self.data['tests']['test_144']['y_expected'].append(y_expected[0])
        self.data['tests']['test_144']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0977, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4686, max_value=0.5533, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7962, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.641, -0.432, -0.38, -0.3, 0.074, 0.188, 0.376, 1.363, 1.514, 2.016]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_145(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_145']['n_samples'] += 1
        self.data['tests']['test_145']['samples'].append(x_test)
        self.data['tests']['test_145']['y_expected'].append(y_expected[0])
        self.data['tests']['test_145']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5536, max_value=0.8703, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5306, max_value=1.5629, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.4813, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.565, -0.442, -0.421, -0.267, -0.211, -0.198, 0.681, 0.832, 0.865, 1.497]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_146(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_146']['n_samples'] += 1
        self.data['tests']['test_146']['samples'].append(x_test)
        self.data['tests']['test_146']['y_expected'].append(y_expected[0])
        self.data['tests']['test_146']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5536, max_value=0.8703, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5632, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.4813, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.453, -0.315, -0.234, -0.138, -0.12, 0.0, 0.126, 0.224, 0.237, 0.608]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_147(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_147']['n_samples'] += 1
        self.data['tests']['test_147']['samples'].append(x_test)
        self.data['tests']['test_147']['y_expected'].append(y_expected[0])
        self.data['tests']['test_147']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5536, max_value=0.8703, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5306, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4816, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.431, -0.229, -0.194, -0.13, -0.069, -0.043, 0.045, 0.201, 0.254, 0.467]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_148(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_148']['n_samples'] += 1
        self.data['tests']['test_148']['samples'].append(x_test)
        self.data['tests']['test_148']['y_expected'].append(y_expected[0])
        self.data['tests']['test_148']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0967, max_value=0.2029, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8706, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5306, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.563, -0.358, -0.356, -0.225, -0.076, 0.169, 0.292, 0.4, 0.734, 1.433]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_149(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_149']['n_samples'] += 1
        self.data['tests']['test_149']['samples'].append(x_test)
        self.data['tests']['test_149']['y_expected'].append(y_expected[0])
        self.data['tests']['test_149']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.187, 0.378, 0.38, 0.614, 0.615, 0.66, 0.712, 0.843, 0.971, 1.382]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5578, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.5297, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_150(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_150']['n_samples'] += 1
        self.data['tests']['test_150']['samples'].append(x_test)
        self.data['tests']['test_150']['y_expected'].append(y_expected[0])
        self.data['tests']['test_150']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.5264, exclude_min=True, allow_nan=False),
           st.sampled_from([0.219, 0.471, 0.597, 0.787, 0.808, 0.889, 1.039, 1.072, 1.265, 1.556]),
           st.floats(min_value=-1.823, max_value=-0.7007, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_151(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_151']['n_samples'] += 1
        self.data['tests']['test_151']['samples'].append(x_test)
        self.data['tests']['test_151']['y_expected'].append(y_expected[0])
        self.data['tests']['test_151']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.5267, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.291, 0.293, 0.314, 0.394, 0.848, 1.187, 1.297, 1.625, 1.771, 2.334]),
           st.floats(min_value=-1.823, max_value=-0.7007, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_152(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_152']['n_samples'] += 1
        self.data['tests']['test_152']['samples'].append(x_test)
        self.data['tests']['test_152']['y_expected'].append(y_expected[0])
        self.data['tests']['test_152']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.9128, exclude_min=True, allow_nan=False),
           st.sampled_from([0.294, 1.139, 1.169, 1.41, 1.445, 1.572, 1.722, 2.251, 2.299, 2.324]),
           st.floats(min_value=-0.7004, max_value=-0.5626, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_153(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_153']['n_samples'] += 1
        self.data['tests']['test_153']['samples'].append(x_test)
        self.data['tests']['test_153']['y_expected'].append(y_expected[0])
        self.data['tests']['test_153']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.9128, exclude_min=True, allow_nan=False),
           st.sampled_from([0.401, 0.583, 0.66, 0.809, 0.822, 0.834, 1.189, 1.502, 1.633, 1.66]),
           st.floats(min_value=-0.5624, max_value=-0.5616, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_154(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_154']['n_samples'] += 1
        self.data['tests']['test_154']['samples'].append(x_test)
        self.data['tests']['test_154']['y_expected'].append(y_expected[0])
        self.data['tests']['test_154']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.9128, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.829, -0.298, 0.417, 1.236, 1.333, 2.203, 2.725, 2.783, 2.927, 3.345]),
           st.floats(min_value=-0.5613, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_155(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_155']['n_samples'] += 1
        self.data['tests']['test_155']['samples'].append(x_test)
        self.data['tests']['test_155']['y_expected'].append(y_expected[0])
        self.data['tests']['test_155']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9131, max_value=0.9178, exclude_min=True, allow_nan=False),
           st.sampled_from([0.299, 0.309, 0.514, 0.791, 0.844, 0.973, 1.318, 1.38, 1.664, 1.932]),
           st.floats(min_value=-0.7004, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_156(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_156']['n_samples'] += 1
        self.data['tests']['test_156']['samples'].append(x_test)
        self.data['tests']['test_156']['y_expected'].append(y_expected[0])
        self.data['tests']['test_156']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9181, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.8518, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_157(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_157']['n_samples'] += 1
        self.data['tests']['test_157']['samples'].append(x_test)
        self.data['tests']['test_157']['y_expected'].append(y_expected[0])
        self.data['tests']['test_157']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9181, max_value=0.9428, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1496, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_158(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_158']['n_samples'] += 1
        self.data['tests']['test_158']['samples'].append(x_test)
        self.data['tests']['test_158']['y_expected'].append(y_expected[0])
        self.data['tests']['test_158']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9431, max_value=0.9563, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1496, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_159(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_159']['n_samples'] += 1
        self.data['tests']['test_159']['samples'].append(x_test)
        self.data['tests']['test_159']['y_expected'].append(y_expected[0])
        self.data['tests']['test_159']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9566, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1496, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_160(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_160']['n_samples'] += 1
        self.data['tests']['test_160']['samples'].append(x_test)
        self.data['tests']['test_160']['y_expected'].append(y_expected[0])
        self.data['tests']['test_160']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9181, max_value=1.4304, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1493, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_161(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_161']['n_samples'] += 1
        self.data['tests']['test_161']['samples'].append(x_test)
        self.data['tests']['test_161']['y_expected'].append(y_expected[0])
        self.data['tests']['test_161']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4307, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7004, max_value=-0.3836, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1493, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_162(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_162']['n_samples'] += 1
        self.data['tests']['test_162']['samples'].append(x_test)
        self.data['tests']['test_162']['y_expected'].append(y_expected[0])
        self.data['tests']['test_162']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9181, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.332, 0.339, 0.518, 0.618, 0.645, 0.76, 0.973, 2.61, 3.178, 4.378]),
           st.floats(min_value=-0.3833, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=-0.1327, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_163(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_163']['n_samples'] += 1
        self.data['tests']['test_163']['samples'].append(x_test)
        self.data['tests']['test_163']['y_expected'].append(y_expected[0])
        self.data['tests']['test_163']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.303, 0.398, 0.417, 0.455, 0.479, 0.602, 0.632, 1.607, 1.791, 2.412]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1324, max_value=-0.1316, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=-0.0746, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_164(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_164']['n_samples'] += 1
        self.data['tests']['test_164']['samples'].append(x_test)
        self.data['tests']['test_164']['y_expected'].append(y_expected[0])
        self.data['tests']['test_164']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.463, 0.554, 0.75, 0.883, 0.901, 1.249, 2.04, 2.243, 2.707, 2.82]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1324, max_value=-0.1316, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0743, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_165(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_165']['n_samples'] += 1
        self.data['tests']['test_165']['samples'].append(x_test)
        self.data['tests']['test_165']['y_expected'].append(y_expected[0])
        self.data['tests']['test_165']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.362, 0.869, 1.233, 1.243, 1.478, 1.76, 1.796, 2.458, 2.814, 3.1]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1313, max_value=0.3679, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=-0.2647, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_166(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_166']['n_samples'] += 1
        self.data['tests']['test_166']['samples'].append(x_test)
        self.data['tests']['test_166']['y_expected'].append(y_expected[0])
        self.data['tests']['test_166']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.9094, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.515, -0.391, 0.431, 0.814, 0.912, 0.961, 1.065, 1.519, 2.133, 2.812]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1313, max_value=0.3679, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2644, max_value=-0.2627, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_167(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_167']['n_samples'] += 1
        self.data['tests']['test_167']['samples'].append(x_test)
        self.data['tests']['test_167']['y_expected'].append(y_expected[0])
        self.data['tests']['test_167']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9097, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.259, 0.726, 1.11, 1.943, 2.334, 2.465, 2.479, 3.093, 3.137, 3.173]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1313, max_value=0.3679, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2644, max_value=-0.2627, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_168(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_168']['n_samples'] += 1
        self.data['tests']['test_168']['samples'].append(x_test)
        self.data['tests']['test_168']['y_expected'].append(y_expected[0])
        self.data['tests']['test_168']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.254, 0.315, 0.36, 0.545, 0.831, 1.068, 1.791, 2.054, 2.079, 2.465]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=-0.1313, max_value=0.3679, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2624, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_169(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_169']['n_samples'] += 1
        self.data['tests']['test_169']['samples'].append(x_test)
        self.data['tests']['test_169']['y_expected'].append(y_expected[0])
        self.data['tests']['test_169']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.352, 0.531, 1.018, 1.163, 1.302, 1.671, 1.688, 1.906, 2.708, 2.897]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.3682, max_value=0.3693, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_170(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_170']['n_samples'] += 1
        self.data['tests']['test_170']['samples'].append(x_test)
        self.data['tests']['test_170']['y_expected'].append(y_expected[0])
        self.data['tests']['test_170']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.5643, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.5578, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_171(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_171']['n_samples'] += 1
        self.data['tests']['test_171']['samples'].append(x_test)
        self.data['tests']['test_171']['y_expected'].append(y_expected[0])
        self.data['tests']['test_171']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5646, max_value=1.5938, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.5578, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_172(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_172']['n_samples'] += 1
        self.data['tests']['test_172']['samples'].append(x_test)
        self.data['tests']['test_172']['y_expected'].append(y_expected[0])
        self.data['tests']['test_172']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=2.1223, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.4446, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_173(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_173']['n_samples'] += 1
        self.data['tests']['test_173']['samples'].append(x_test)
        self.data['tests']['test_173']['y_expected'].append(y_expected[0])
        self.data['tests']['test_173']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.6048, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=2.1223, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4443, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=-0.2137, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_174(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_174']['n_samples'] += 1
        self.data['tests']['test_174']['samples'].append(x_test)
        self.data['tests']['test_174']['y_expected'].append(y_expected[0])
        self.data['tests']['test_174']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6051, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=2.1223, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4443, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=-0.2137, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_175(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_175']['n_samples'] += 1
        self.data['tests']['test_175']['samples'].append(x_test)
        self.data['tests']['test_175']['y_expected'].append(y_expected[0])
        self.data['tests']['test_175']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=2.1223, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4443, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_176(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_176']['n_samples'] += 1
        self.data['tests']['test_176']['samples'].append(x_test)
        self.data['tests']['test_176']['y_expected'].append(y_expected[0])
        self.data['tests']['test_176']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1226, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=0.1994, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_177(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_177']['n_samples'] += 1
        self.data['tests']['test_177']['samples'].append(x_test)
        self.data['tests']['test_177']['y_expected'].append(y_expected[0])
        self.data['tests']['test_177']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1226, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1997, max_value=0.2134, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_178(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_178']['n_samples'] += 1
        self.data['tests']['test_178']['samples'].append(x_test)
        self.data['tests']['test_178']['y_expected'].append(y_expected[0])
        self.data['tests']['test_178']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.5898, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1226, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.5941, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2137, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_179(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_179']['n_samples'] += 1
        self.data['tests']['test_179']['samples'].append(x_test)
        self.data['tests']['test_179']['y_expected'].append(y_expected[0])
        self.data['tests']['test_179']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.5901, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1226, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.5941, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2137, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_180(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_180']['n_samples'] += 1
        self.data['tests']['test_180']['samples'].append(x_test)
        self.data['tests']['test_180']['y_expected'].append(y_expected[0])
        self.data['tests']['test_180']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1226, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5938, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2137, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_181(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_181']['n_samples'] += 1
        self.data['tests']['test_181']['samples'].append(x_test)
        self.data['tests']['test_181']['y_expected'].append(y_expected[0])
        self.data['tests']['test_181']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.4487, max_value=0.4493, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_182(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_182']['n_samples'] += 1
        self.data['tests']['test_182']['samples'].append(x_test)
        self.data['tests']['test_182']['y_expected'].append(y_expected[0])
        self.data['tests']['test_182']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5941, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.4496, max_value=0.5578, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5294, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_183(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_183']['n_samples'] += 1
        self.data['tests']['test_183']['samples'].append(x_test)
        self.data['tests']['test_183']['y_expected'].append(y_expected[0])
        self.data['tests']['test_183']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.273, 0.398, 0.485, 0.878, 1.253, 1.38, 1.502, 1.817, 2.41, 2.603]),
           st.floats(min_value=-1.823, max_value=0.7564, allow_nan=False),
           st.floats(min_value=0.5581, max_value=0.5673, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.776, -0.698, -0.63, -0.491, -0.408, 0.345, 1.191, 1.273, 1.495, 2.012]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_184(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_184']['n_samples'] += 1
        self.data['tests']['test_184']['samples'].append(x_test)
        self.data['tests']['test_184']['y_expected'].append(y_expected[0])
        self.data['tests']['test_184']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.243, 0.818, 1.064, 1.178, 1.888, 2.23, 2.622, 2.745, 2.963, 3.05]),
           st.floats(min_value=-1.823, max_value=0.6374, allow_nan=False),
           st.floats(min_value=0.5676, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.624, -0.507, -0.365, -0.122, 0.127, 0.196, 0.314, 0.462, 1.107, 1.753]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_185(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_185']['n_samples'] += 1
        self.data['tests']['test_185']['samples'].append(x_test)
        self.data['tests']['test_185']['y_expected'].append(y_expected[0])
        self.data['tests']['test_185']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.812, 0.402, 0.451, 0.513, 0.711, 0.991, 1.0, 1.043, 1.233, 2.708]),
           st.floats(min_value=0.6377, max_value=0.7564, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5676, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.9, -0.488, -0.268, 0.376, 0.544, 0.568, 0.854, 0.965, 1.234, 2.337]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_186(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_186']['n_samples'] += 1
        self.data['tests']['test_186']['samples'].append(x_test)
        self.data['tests']['test_186']['y_expected'].append(y_expected[0])
        self.data['tests']['test_186']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.3853, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.082, 0.297, 0.465, 0.517, 0.568, 1.284, 1.379, 1.38, 1.392, 1.478]),
           st.floats(min_value=0.7567, max_value=0.9324, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_187(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_187']['n_samples'] += 1
        self.data['tests']['test_187']['samples'].append(x_test)
        self.data['tests']['test_187']['y_expected'].append(y_expected[0])
        self.data['tests']['test_187']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3856, max_value=0.4988, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.206, 0.606, 0.656, 0.675, 1.914, 2.006, 2.074, 2.083, 2.239, 2.983]),
           st.floats(min_value=0.7567, max_value=0.9324, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_188(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_188']['n_samples'] += 1
        self.data['tests']['test_188']['samples'].append(x_test)
        self.data['tests']['test_188']['y_expected'].append(y_expected[0])
        self.data['tests']['test_188']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.4988, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.9849, allow_nan=False),
           st.floats(min_value=0.9327, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_189(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_189']['n_samples'] += 1
        self.data['tests']['test_189']['samples'].append(x_test)
        self.data['tests']['test_189']['y_expected'].append(y_expected[0])
        self.data['tests']['test_189']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.4988, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9852, max_value=2.0258, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9327, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_190(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_190']['n_samples'] += 1
        self.data['tests']['test_190']['samples'].append(x_test)
        self.data['tests']['test_190']['y_expected'].append(y_expected[0])
        self.data['tests']['test_190']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.4988, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0261, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9327, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_191(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_191']['n_samples'] += 1
        self.data['tests']['test_191']['samples'].append(x_test)
        self.data['tests']['test_191']['y_expected'].append(y_expected[0])
        self.data['tests']['test_191']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4991, max_value=0.6174, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.497, 0.588, 0.679, 0.808, 0.837, 1.046, 1.247, 1.512, 2.367, 2.709]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_192(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_192']['n_samples'] += 1
        self.data['tests']['test_192']['samples'].append(x_test)
        self.data['tests']['test_192']['y_expected'].append(y_expected[0])
        self.data['tests']['test_192']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6177, max_value=0.7304, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.314, -0.269, 0.368, 1.067, 1.798, 2.195, 2.719, 2.94, 3.093, 3.327]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2551, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_193(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_193']['n_samples'] += 1
        self.data['tests']['test_193']['samples'].append(x_test)
        self.data['tests']['test_193']['y_expected'].append(y_expected[0])
        self.data['tests']['test_193']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.7304, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.324, 0.349, 0.631, 0.705, 0.727, 1.045, 1.078, 1.678, 1.716, 2.394]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2548, max_value=0.2879, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_194(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_194']['n_samples'] += 1
        self.data['tests']['test_194']['samples'].append(x_test)
        self.data['tests']['test_194']['y_expected'].append(y_expected[0])
        self.data['tests']['test_194']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.7304, exclude_min=True, allow_nan=False),
           st.sampled_from([0.396, 0.471, 0.56, 0.643, 1.289, 1.391, 1.513, 1.68, 2.025, 2.368]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2882, max_value=0.3093, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_195(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_195']['n_samples'] += 1
        self.data['tests']['test_195']['samples'].append(x_test)
        self.data['tests']['test_195']['y_expected'].append(y_expected[0])
        self.data['tests']['test_195']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.7304, exclude_min=True, allow_nan=False),
           st.sampled_from([0.376, 0.987, 1.352, 1.424, 2.227, 2.235, 2.253, 2.637, 2.874, 2.903]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3096, max_value=0.6158, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_196(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_196']['n_samples'] += 1
        self.data['tests']['test_196']['samples'].append(x_test)
        self.data['tests']['test_196']['y_expected'].append(y_expected[0])
        self.data['tests']['test_196']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=2.8163, allow_nan=False),
           st.floats(min_value=0.7567, max_value=0.8103, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.0268, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.1256, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_197(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_197']['n_samples'] += 1
        self.data['tests']['test_197']['samples'].append(x_test)
        self.data['tests']['test_197']['y_expected'].append(y_expected[0])
        self.data['tests']['test_197']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=2.8163, allow_nan=False),
           st.floats(min_value=0.7567, max_value=0.8103, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0271, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.1256, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_198(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_198']['n_samples'] += 1
        self.data['tests']['test_198']['samples'].append(x_test)
        self.data['tests']['test_198']['y_expected'].append(y_expected[0])
        self.data['tests']['test_198']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=2.8163, allow_nan=False),
           st.floats(min_value=0.8106, max_value=1.0464, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.1256, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_199(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_199']['n_samples'] += 1
        self.data['tests']['test_199']['samples'].append(x_test)
        self.data['tests']['test_199']['y_expected'].append(y_expected[0])
        self.data['tests']['test_199']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=2.8163, allow_nan=False),
           st.floats(min_value=0.7567, max_value=1.0464, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1253, max_value=-0.0531, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_200(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_200']['n_samples'] += 1
        self.data['tests']['test_200']['samples'].append(x_test)
        self.data['tests']['test_200']['y_expected'].append(y_expected[0])
        self.data['tests']['test_200']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.8166, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7567, max_value=1.0464, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.0531, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_201(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_201']['n_samples'] += 1
        self.data['tests']['test_201']['samples'].append(x_test)
        self.data['tests']['test_201']['y_expected'].append(y_expected[0])
        self.data['tests']['test_201']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=0.7684, exclude_min=True, allow_nan=False),
           st.sampled_from([0.357, 0.372, 0.483, 0.539, 0.791, 0.862, 0.897, 1.542, 2.154, 2.64]),
           st.floats(min_value=0.7567, max_value=0.8888, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0528, max_value=0.6158, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_202(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_202']['n_samples'] += 1
        self.data['tests']['test_202']['samples'].append(x_test)
        self.data['tests']['test_202']['y_expected'].append(y_expected[0])
        self.data['tests']['test_202']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=0.7684, exclude_min=True, allow_nan=False),
           st.sampled_from([0.79, 0.962, 1.075, 1.091, 1.1, 1.335, 1.935, 2.027, 2.421, 3.202]),
           st.floats(min_value=0.8891, max_value=1.0464, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0528, max_value=0.6158, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_203(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_203']['n_samples'] += 1
        self.data['tests']['test_203']['samples'].append(x_test)
        self.data['tests']['test_203']['y_expected'].append(y_expected[0])
        self.data['tests']['test_203']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7687, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.sampled_from([0.327, 0.728, 0.853, 0.959, 0.968, 1.123, 1.561, 1.99, 2.448, 2.51]),
           st.floats(min_value=0.7567, max_value=1.0464, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0528, max_value=0.6158, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_204(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_204']['n_samples'] += 1
        self.data['tests']['test_204']['samples'].append(x_test)
        self.data['tests']['test_204']['y_expected'].append(y_expected[0])
        self.data['tests']['test_204']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=0.9374, exclude_min=True, allow_nan=False),
           st.sampled_from([0.349, 0.514, 0.963, 1.107, 1.258, 1.748, 2.011, 2.31, 2.658, 2.69]),
           st.floats(min_value=1.0467, max_value=1.1874, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_205(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_205']['n_samples'] += 1
        self.data['tests']['test_205']['samples'].append(x_test)
        self.data['tests']['test_205']['y_expected'].append(y_expected[0])
        self.data['tests']['test_205']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9377, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.sampled_from([0.401, 0.441, 0.545, 0.552, 0.672, 0.717, 0.811, 1.473, 1.632, 2.372]),
           st.floats(min_value=1.0467, max_value=1.1874, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5133, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3118, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_206(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_206']['n_samples'] += 1
        self.data['tests']['test_206']['samples'].append(x_test)
        self.data['tests']['test_206']['y_expected'].append(y_expected[0])
        self.data['tests']['test_206']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9377, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.sampled_from([0.365, 0.38, 0.382, 0.425, 1.328, 1.344, 1.505, 1.636, 1.696, 2.611]),
           st.floats(min_value=1.0467, max_value=1.1874, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5136, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3118, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_207(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_207']['n_samples'] += 1
        self.data['tests']['test_207']['samples'].append(x_test)
        self.data['tests']['test_207']['y_expected'].append(y_expected[0])
        self.data['tests']['test_207']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9377, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.sampled_from([0.454, 0.681, 0.901, 0.904, 1.389, 1.54, 1.981, 2.396, 2.757, 3.056]),
           st.floats(min_value=1.0467, max_value=1.1874, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3121, max_value=0.6158, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_208(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_208']['n_samples'] += 1
        self.data['tests']['test_208']['samples'].append(x_test)
        self.data['tests']['test_208']['y_expected'].append(y_expected[0])
        self.data['tests']['test_208']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7307, max_value=1.4583, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.74, 0.433, 0.475, 0.496, 0.744, 1.124, 1.185, 1.193, 1.628, 2.271]),
           st.floats(min_value=1.1876, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_209(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_209']['n_samples'] += 1
        self.data['tests']['test_209']['samples'].append(x_test)
        self.data['tests']['test_209']['y_expected'].append(y_expected[0])
        self.data['tests']['test_209']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4586, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.514, 1.051, 1.073, 1.166, 1.167, 1.302, 1.58, 1.981, 1.994, 2.512]),
           st.floats(min_value=0.7567, max_value=1.4489, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_210(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_210']['n_samples'] += 1
        self.data['tests']['test_210']['samples'].append(x_test)
        self.data['tests']['test_210']['y_expected'].append(y_expected[0])
        self.data['tests']['test_210']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.7934, allow_nan=False),
           st.floats(min_value=1.4492, max_value=1.4644, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_211(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_211']['n_samples'] += 1
        self.data['tests']['test_211']['samples'].append(x_test)
        self.data['tests']['test_211']['y_expected'].append(y_expected[0])
        self.data['tests']['test_211']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.7934, allow_nan=False),
           st.floats(min_value=1.4647, max_value=1.7893, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_212(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_212']['n_samples'] += 1
        self.data['tests']['test_212']['samples'].append(x_test)
        self.data['tests']['test_212']['y_expected'].append(y_expected[0])
        self.data['tests']['test_212']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.7934, allow_nan=False),
           st.floats(min_value=1.7896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5418, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_213(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_213']['n_samples'] += 1
        self.data['tests']['test_213']['samples'].append(x_test)
        self.data['tests']['test_213']['y_expected'].append(y_expected[0])
        self.data['tests']['test_213']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.7934, allow_nan=False),
           st.floats(min_value=1.7896, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5421, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_214(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_214']['n_samples'] += 1
        self.data['tests']['test_214']['samples'].append(x_test)
        self.data['tests']['test_214']['y_expected'].append(y_expected[0])
        self.data['tests']['test_214']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.7023, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7937, max_value=2.8889, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4492, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_215(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_215']['n_samples'] += 1
        self.data['tests']['test_215']['samples'].append(x_test)
        self.data['tests']['test_215']['y_expected'].append(y_expected[0])
        self.data['tests']['test_215']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=0.7023, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.8892, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4492, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_216(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_216']['n_samples'] += 1
        self.data['tests']['test_216']['samples'].append(x_test)
        self.data['tests']['test_216']['y_expected'].append(y_expected[0])
        self.data['tests']['test_216']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7026, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7937, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4492, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.6158, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_217(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_217']['n_samples'] += 1
        self.data['tests']['test_217']['samples'].append(x_test)
        self.data['tests']['test_217']['y_expected'].append(y_expected[0])
        self.data['tests']['test_217']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.6003, allow_nan=False),
           st.floats(min_value=0.7567, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6161, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_218(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_218']['n_samples'] += 1
        self.data['tests']['test_218']['samples'].append(x_test)
        self.data['tests']['test_218']['y_expected'].append(y_expected[0])
        self.data['tests']['test_218']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6006, max_value=1.8398, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7567, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6161, max_value=0.7889, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_219(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_219']['n_samples'] += 1
        self.data['tests']['test_219']['samples'].append(x_test)
        self.data['tests']['test_219']['y_expected'].append(y_expected[0])
        self.data['tests']['test_219']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8401, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7567, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6161, max_value=0.7889, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_220(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_220']['n_samples'] += 1
        self.data['tests']['test_220']['samples'].append(x_test)
        self.data['tests']['test_220']['y_expected'].append(y_expected[0])
        self.data['tests']['test_220']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2032, max_value=4.107, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6006, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7567, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2963, max_value=0.5763, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7892, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_221(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_221']['n_samples'] += 1
        self.data['tests']['test_221']['samples'].append(x_test)
        self.data['tests']['test_221']['y_expected'].append(y_expected[0])
        self.data['tests']['test_221']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.1587, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.3043, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.364, 0.227, 0.313, 0.525, 0.549, 0.981, 1.176, 1.235, 1.433, 1.551]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_222(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_222']['n_samples'] += 1
        self.data['tests']['test_222']['samples'].append(x_test)
        self.data['tests']['test_222']['y_expected'].append(y_expected[0])
        self.data['tests']['test_222']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1584, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.0398, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1584, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_223(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_223']['n_samples'] += 1
        self.data['tests']['test_223']['samples'].append(x_test)
        self.data['tests']['test_223']['y_expected'].append(y_expected[0])
        self.data['tests']['test_223']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1584, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.0398, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1587, max_value=1.2734, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_224(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_224']['n_samples'] += 1
        self.data['tests']['test_224']['samples'].append(x_test)
        self.data['tests']['test_224']['y_expected'].append(y_expected[0])
        self.data['tests']['test_224']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1584, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.0398, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2737, max_value=1.7644, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_225(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_225']['n_samples'] += 1
        self.data['tests']['test_225']['samples'].append(x_test)
        self.data['tests']['test_225']['y_expected'].append(y_expected[0])
        self.data['tests']['test_225']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1584, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.0398, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7647, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_226(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_226']['n_samples'] += 1
        self.data['tests']['test_226']['samples'].append(x_test)
        self.data['tests']['test_226']['y_expected'].append(y_expected[0])
        self.data['tests']['test_226']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1584, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4283, allow_nan=False),
           st.floats(min_value=0.0401, max_value=0.3043, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.572, -0.323, -0.248, -0.175, -0.162, -0.137, -0.109, 0.101, 0.112, 0.224]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_227(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_227']['n_samples'] += 1
        self.data['tests']['test_227']['samples'].append(x_test)
        self.data['tests']['test_227']['y_expected'].append(y_expected[0])
        self.data['tests']['test_227']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4286, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.0721, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.834, -0.542, -0.419, -0.396, -0.36, -0.234, 0.18, 0.439, 1.016, 1.155]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_228(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_228']['n_samples'] += 1
        self.data['tests']['test_228']['samples'].append(x_test)
        self.data['tests']['test_228']['y_expected'].append(y_expected[0])
        self.data['tests']['test_228']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4286, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0718, max_value=0.3043, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.008, -0.857, -0.692, -0.492, -0.308, -0.244, 0.501, 0.623, 1.116, 1.481]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_229(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_229']['n_samples'] += 1
        self.data['tests']['test_229']['samples'].append(x_test)
        self.data['tests']['test_229']['y_expected'].append(y_expected[0])
        self.data['tests']['test_229']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.5854, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.67, -0.602, -0.531, -0.358, -0.223, -0.154, 0.493, 1.451, 1.462, 2.087]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_230(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_230']['n_samples'] += 1
        self.data['tests']['test_230']['samples'].append(x_test)
        self.data['tests']['test_230']['y_expected'].append(y_expected[0])
        self.data['tests']['test_230']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3903, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.4193, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5857, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2016, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_231(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_231']['n_samples'] += 1
        self.data['tests']['test_231']['samples'].append(x_test)
        self.data['tests']['test_231']['y_expected'].append(y_expected[0])
        self.data['tests']['test_231']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3903, allow_nan=False),
           st.floats(min_value=0.4196, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5857, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2016, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_232(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_232']['n_samples'] += 1
        self.data['tests']['test_232']['samples'].append(x_test)
        self.data['tests']['test_232']['y_expected'].append(y_expected[0])
        self.data['tests']['test_232']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3903, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5857, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2013, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_233(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_233']['n_samples'] += 1
        self.data['tests']['test_233']['samples'].append(x_test)
        self.data['tests']['test_233']['y_expected'].append(y_expected[0])
        self.data['tests']['test_233']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=0.3906, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5857, max_value=1.3279, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.849, -0.706, -0.445, -0.341, -0.069, 0.239, 0.391, 0.55, 0.785, 0.912]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_234(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_234']['n_samples'] += 1
        self.data['tests']['test_234']['samples'].append(x_test)
        self.data['tests']['test_234']['y_expected'].append(y_expected[0])
        self.data['tests']['test_234']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=0.3906, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3282, max_value=1.7999, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.674, -0.35, 0.342, 0.623, 0.865, 1.067, 1.108, 1.812, 1.88, 2.373]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_235(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_235']['n_samples'] += 1
        self.data['tests']['test_235']['samples'].append(x_test)
        self.data['tests']['test_235']['y_expected'].append(y_expected[0])
        self.data['tests']['test_235']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1699, allow_nan=False),
           st.floats(min_value=0.3906, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8002, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.859, -0.845, -0.144, 0.087, 0.288, 0.671, 0.987, 1.361, 1.372, 1.838]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_236(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_236']['n_samples'] += 1
        self.data['tests']['test_236']['samples'].append(x_test)
        self.data['tests']['test_236']['y_expected'].append(y_expected[0])
        self.data['tests']['test_236']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1702, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.2458, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_237(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_237']['n_samples'] += 1
        self.data['tests']['test_237']['samples'].append(x_test)
        self.data['tests']['test_237']['y_expected'].append(y_expected[0])
        self.data['tests']['test_237']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1702, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3479, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2461, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_238(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_238']['n_samples'] += 1
        self.data['tests']['test_238']['samples'].append(x_test)
        self.data['tests']['test_238']['y_expected'].append(y_expected[0])
        self.data['tests']['test_238']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1702, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3482, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3046, max_value=0.5163, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2461, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_239(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_239']['n_samples'] += 1
        self.data['tests']['test_239']['samples'].append(x_test)
        self.data['tests']['test_239']['y_expected'].append(y_expected[0])
        self.data['tests']['test_239']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1524, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5389, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6732, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_240(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_240']['n_samples'] += 1
        self.data['tests']['test_240']['samples'].append(x_test)
        self.data['tests']['test_240']['y_expected'].append(y_expected[0])
        self.data['tests']['test_240']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1527, max_value=0.1908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5389, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6732, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_241(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_241']['n_samples'] += 1
        self.data['tests']['test_241']['samples'].append(x_test)
        self.data['tests']['test_241']['y_expected'].append(y_expected[0])
        self.data['tests']['test_241']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1908, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5392, max_value=0.7974, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6732, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_242(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_242']['n_samples'] += 1
        self.data['tests']['test_242']['samples'].append(x_test)
        self.data['tests']['test_242']['y_expected'].append(y_expected[0])
        self.data['tests']['test_242']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1908, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7977, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.8352, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_243(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_243']['n_samples'] += 1
        self.data['tests']['test_243']['samples'].append(x_test)
        self.data['tests']['test_243']['y_expected'].append(y_expected[0])
        self.data['tests']['test_243']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1908, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7977, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.8349, max_value=-0.6732, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_244(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_244']['n_samples'] += 1
        self.data['tests']['test_244']['samples'].append(x_test)
        self.data['tests']['test_244']['y_expected'].append(y_expected[0])
        self.data['tests']['test_244']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1554, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5283, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=1.7429, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_245(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_245']['n_samples'] += 1
        self.data['tests']['test_245']['samples'].append(x_test)
        self.data['tests']['test_245']['y_expected'].append(y_expected[0])
        self.data['tests']['test_245']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1063, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3468, allow_nan=False),
           st.floats(min_value=0.5286, max_value=0.5948, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=0.9614, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_246(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_246']['n_samples'] += 1
        self.data['tests']['test_246']['samples'].append(x_test)
        self.data['tests']['test_246']['y_expected'].append(y_expected[0])
        self.data['tests']['test_246']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1063, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3468, allow_nan=False),
           st.floats(min_value=0.5951, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=0.9614, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_247(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_247']['n_samples'] += 1
        self.data['tests']['test_247']['samples'].append(x_test)
        self.data['tests']['test_247']['y_expected'].append(y_expected[0])
        self.data['tests']['test_247']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1063, allow_nan=False),
           st.floats(min_value=0.3471, max_value=0.4964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5286, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=0.9614, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_248(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_248']['n_samples'] += 1
        self.data['tests']['test_248']['samples'].append(x_test)
        self.data['tests']['test_248']['y_expected'].append(y_expected[0])
        self.data['tests']['test_248']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1063, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5286, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9617, max_value=1.7429, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_249(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_249']['n_samples'] += 1
        self.data['tests']['test_249']['samples'].append(x_test)
        self.data['tests']['test_249']['y_expected'].append(y_expected[0])
        self.data['tests']['test_249']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1066, max_value=0.1554, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5286, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=1.7429, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_250(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_250']['n_samples'] += 1
        self.data['tests']['test_250']['samples'].append(x_test)
        self.data['tests']['test_250']['y_expected'].append(y_expected[0])
        self.data['tests']['test_250']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1557, max_value=0.1908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=1.7429, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_251(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_251']['n_samples'] += 1
        self.data['tests']['test_251']['samples'].append(x_test)
        self.data['tests']['test_251']['y_expected'].append(y_expected[0])
        self.data['tests']['test_251']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1908, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.7363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7432, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_252(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_252']['n_samples'] += 1
        self.data['tests']['test_252']['samples'].append(x_test)
        self.data['tests']['test_252']['y_expected'].append(y_expected[0])
        self.data['tests']['test_252']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0968, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7366, max_value=0.7904, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_253(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_253']['n_samples'] += 1
        self.data['tests']['test_253']['samples'].append(x_test)
        self.data['tests']['test_253']['y_expected'].append(y_expected[0])
        self.data['tests']['test_253']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0968, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7907, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_254(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_254']['n_samples'] += 1
        self.data['tests']['test_254']['samples'].append(x_test)
        self.data['tests']['test_254']['y_expected'].append(y_expected[0])
        self.data['tests']['test_254']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0971, max_value=0.1908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7366, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5039, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_255(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_255']['n_samples'] += 1
        self.data['tests']['test_255']['samples'].append(x_test)
        self.data['tests']['test_255']['y_expected'].append(y_expected[0])
        self.data['tests']['test_255']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0874, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5042, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_256(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_256']['n_samples'] += 1
        self.data['tests']['test_256']['samples'].append(x_test)
        self.data['tests']['test_256']['y_expected'].append(y_expected[0])
        self.data['tests']['test_256']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0877, max_value=0.1133, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5589, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5042, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_257(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_257']['n_samples'] += 1
        self.data['tests']['test_257']['samples'].append(x_test)
        self.data['tests']['test_257']['y_expected'].append(y_expected[0])
        self.data['tests']['test_257']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0877, max_value=0.1133, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5592, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5042, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_258(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_258']['n_samples'] += 1
        self.data['tests']['test_258']['samples'].append(x_test)
        self.data['tests']['test_258']['y_expected'].append(y_expected[0])
        self.data['tests']['test_258']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1136, max_value=0.1908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5042, max_value=2.0554, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_259(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_259']['n_samples'] += 1
        self.data['tests']['test_259']['samples'].append(x_test)
        self.data['tests']['test_259']['y_expected'].append(y_expected[0])
        self.data['tests']['test_259']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1136, max_value=0.1354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0557, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_260(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_260']['n_samples'] += 1
        self.data['tests']['test_260']['samples'].append(x_test)
        self.data['tests']['test_260']['y_expected'].append(y_expected[0])
        self.data['tests']['test_260']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1357, max_value=0.1908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0557, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6729, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_261(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_261']['n_samples'] += 1
        self.data['tests']['test_261']['samples'].append(x_test)
        self.data['tests']['test_261']['y_expected'].append(y_expected[0])
        self.data['tests']['test_261']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1911, max_value=0.2254, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5253, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.825, -0.693, -0.506, -0.436, -0.276, -0.219, 0.56, 0.96, 1.157, 1.886]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_262(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_262']['n_samples'] += 1
        self.data['tests']['test_262']['samples'].append(x_test)
        self.data['tests']['test_262']['y_expected'].append(y_expected[0])
        self.data['tests']['test_262']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2257, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5253, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.307, -0.134, -0.069, -0.041, 0.091, 0.134, 0.178, 0.544, 0.826, 1.781]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_263(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_263']['n_samples'] += 1
        self.data['tests']['test_263']['samples'].append(x_test)
        self.data['tests']['test_263']['y_expected'].append(y_expected[0])
        self.data['tests']['test_263']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1911, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5256, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9874, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3539, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_264(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_264']['n_samples'] += 1
        self.data['tests']['test_264']['samples'].append(x_test)
        self.data['tests']['test_264']['y_expected'].append(y_expected[0])
        self.data['tests']['test_264']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1911, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5256, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9877, max_value=1.0794, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3539, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_265(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_265']['n_samples'] += 1
        self.data['tests']['test_265']['samples'].append(x_test)
        self.data['tests']['test_265']['y_expected'].append(y_expected[0])
        self.data['tests']['test_265']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1911, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5256, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0797, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3539, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_266(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_266']['n_samples'] += 1
        self.data['tests']['test_266']['samples'].append(x_test)
        self.data['tests']['test_266']['y_expected'].append(y_expected[0])
        self.data['tests']['test_266']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1911, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5256, max_value=0.9589, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3542, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_267(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_267']['n_samples'] += 1
        self.data['tests']['test_267']['samples'].append(x_test)
        self.data['tests']['test_267']['y_expected'].append(y_expected[0])
        self.data['tests']['test_267']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.9592, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7826, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_268(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_268']['n_samples'] += 1
        self.data['tests']['test_268']['samples'].append(x_test)
        self.data['tests']['test_268']['y_expected'].append(y_expected[0])
        self.data['tests']['test_268']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.9592, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.1288, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7823, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_269(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_269']['n_samples'] += 1
        self.data['tests']['test_269']['samples'].append(x_test)
        self.data['tests']['test_269']['y_expected'].append(y_expected[0])
        self.data['tests']['test_269']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.7444, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1291, max_value=2.2158, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.893, -0.539, -0.357, -0.263, 0.277, 0.374, 0.552, 0.868, 0.979, 1.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_270(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_270']['n_samples'] += 1
        self.data['tests']['test_270']['samples'].append(x_test)
        self.data['tests']['test_270']['y_expected'].append(y_expected[0])
        self.data['tests']['test_270']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.7447, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1291, max_value=2.2158, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.635, -0.623, -0.537, -0.357, 0.357, 0.713, 0.748, 0.75, 1.126, 1.38]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_271(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_271']['n_samples'] += 1
        self.data['tests']['test_271']['samples'].append(x_test)
        self.data['tests']['test_271']['y_expected'].append(y_expected[0])
        self.data['tests']['test_271']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.2161, max_value=2.4719, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.596, -0.46, -0.322, 0.076, 0.205, 0.278, 0.333, 0.826, 1.114, 1.2]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_272(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_272']['n_samples'] += 1
        self.data['tests']['test_272']['samples'].append(x_test)
        self.data['tests']['test_272']['y_expected'].append(y_expected[0])
        self.data['tests']['test_272']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4964, allow_nan=False),
           st.floats(min_value=0.5166, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.4722, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.55, -0.537, -0.398, 0.721, 0.839, 1.264, 1.457, 1.487, 1.578, 1.817]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_273(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_273']['n_samples'] += 1
        self.data['tests']['test_273']['samples'].append(x_test)
        self.data['tests']['test_273']['y_expected'].append(y_expected[0])
        self.data['tests']['test_273']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7679, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_274(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_274']['n_samples'] += 1
        self.data['tests']['test_274']['samples'].append(x_test)
        self.data['tests']['test_274']['y_expected'].append(y_expected[0])
        self.data['tests']['test_274']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.9757, allow_nan=False),
           st.floats(min_value=0.7682, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_275(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_275']['n_samples'] += 1
        self.data['tests']['test_275']['samples'].append(x_test)
        self.data['tests']['test_275']['y_expected'].append(y_expected[0])
        self.data['tests']['test_275']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5123, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.9754, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7682, max_value=1.9069, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_276(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_276']['n_samples'] += 1
        self.data['tests']['test_276']['samples'].append(x_test)
        self.data['tests']['test_276']['y_expected'].append(y_expected[0])
        self.data['tests']['test_276']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5123, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.9754, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9072, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_277(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_277']['n_samples'] += 1
        self.data['tests']['test_277']['samples'].append(x_test)
        self.data['tests']['test_277']['y_expected'].append(y_expected[0])
        self.data['tests']['test_277']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.1107, allow_nan=False),
           st.floats(min_value=0.5126, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.9754, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7682, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_278(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_278']['n_samples'] += 1
        self.data['tests']['test_278']['samples'].append(x_test)
        self.data['tests']['test_278']['y_expected'].append(y_expected[0])
        self.data['tests']['test_278']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1104, max_value=-0.0016, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5126, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.9754, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7682, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_279(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_279']['n_samples'] += 1
        self.data['tests']['test_279']['samples'].append(x_test)
        self.data['tests']['test_279']['y_expected'].append(y_expected[0])
        self.data['tests']['test_279']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.0013, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5126, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.9754, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7682, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7711, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_280(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_280']['n_samples'] += 1
        self.data['tests']['test_280']['samples'].append(x_test)
        self.data['tests']['test_280']['y_expected'].append(y_expected[0])
        self.data['tests']['test_280']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.4526, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_281(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_281']['n_samples'] += 1
        self.data['tests']['test_281']['samples'].append(x_test)
        self.data['tests']['test_281']['y_expected'].append(y_expected[0])
        self.data['tests']['test_281']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4523, max_value=0.9249, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=0.8999, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_282(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_282']['n_samples'] += 1
        self.data['tests']['test_282']['samples'].append(x_test)
        self.data['tests']['test_282']['y_expected'].append(y_expected[0])
        self.data['tests']['test_282']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4523, max_value=0.6313, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9002, max_value=1.0338, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_283(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_283']['n_samples'] += 1
        self.data['tests']['test_283']['samples'].append(x_test)
        self.data['tests']['test_283']['y_expected'].append(y_expected[0])
        self.data['tests']['test_283']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6316, max_value=0.9249, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9002, max_value=1.0338, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_284(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_284']['n_samples'] += 1
        self.data['tests']['test_284']['samples'].append(x_test)
        self.data['tests']['test_284']['y_expected'].append(y_expected[0])
        self.data['tests']['test_284']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4523, max_value=0.9249, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0341, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_285(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_285']['n_samples'] += 1
        self.data['tests']['test_285']['samples'].append(x_test)
        self.data['tests']['test_285']['y_expected'].append(y_expected[0])
        self.data['tests']['test_285']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9252, max_value=1.1384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5708, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_286(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_286']['n_samples'] += 1
        self.data['tests']['test_286']['samples'].append(x_test)
        self.data['tests']['test_286']['y_expected'].append(y_expected[0])
        self.data['tests']['test_286']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1387, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.5708, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_287(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_287']['n_samples'] += 1
        self.data['tests']['test_287']['samples'].append(x_test)
        self.data['tests']['test_287']['y_expected'].append(y_expected[0])
        self.data['tests']['test_287']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9252, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5711, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_288(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_288']['n_samples'] += 1
        self.data['tests']['test_288']['samples'].append(x_test)
        self.data['tests']['test_288']['y_expected'].append(y_expected[0])
        self.data['tests']['test_288']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5342, max_value=0.5463, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9252, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5711, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_289(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_289']['n_samples'] += 1
        self.data['tests']['test_289']['samples'].append(x_test)
        self.data['tests']['test_289']['y_expected'].append(y_expected[0])
        self.data['tests']['test_289']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1428, allow_nan=False),
           st.floats(min_value=0.5466, max_value=0.5499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_290(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_290']['n_samples'] += 1
        self.data['tests']['test_290']['samples'].append(x_test)
        self.data['tests']['test_290']['y_expected'].append(y_expected[0])
        self.data['tests']['test_290']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1431, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5466, max_value=0.5499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_291(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_291']['n_samples'] += 1
        self.data['tests']['test_291']['samples'].append(x_test)
        self.data['tests']['test_291']['y_expected'].append(y_expected[0])
        self.data['tests']['test_291']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5502, max_value=0.9188, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.3813, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_292(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_292']['n_samples'] += 1
        self.data['tests']['test_292']['samples'].append(x_test)
        self.data['tests']['test_292']['y_expected'].append(y_expected[0])
        self.data['tests']['test_292']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.9191, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.3813, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_293(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_293']['n_samples'] += 1
        self.data['tests']['test_293']['samples'].append(x_test)
        self.data['tests']['test_293']['y_expected'].append(y_expected[0])
        self.data['tests']['test_293']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5502, max_value=0.5684, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_294(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_294']['n_samples'] += 1
        self.data['tests']['test_294']['samples'].append(x_test)
        self.data['tests']['test_294']['y_expected'].append(y_expected[0])
        self.data['tests']['test_294']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.3948, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7268, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_295(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_295']['n_samples'] += 1
        self.data['tests']['test_295']['samples'].append(x_test)
        self.data['tests']['test_295']['y_expected'].append(y_expected[0])
        self.data['tests']['test_295']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.5854, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3951, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7268, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_296(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_296']['n_samples'] += 1
        self.data['tests']['test_296']['samples'].append(x_test)
        self.data['tests']['test_296']['y_expected'].append(y_expected[0])
        self.data['tests']['test_296']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5857, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3951, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7268, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_297(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_297']['n_samples'] += 1
        self.data['tests']['test_297']['samples'].append(x_test)
        self.data['tests']['test_297']['y_expected'].append(y_expected[0])
        self.data['tests']['test_297']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.5804, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_298(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_298']['n_samples'] += 1
        self.data['tests']['test_298']['samples'].append(x_test)
        self.data['tests']['test_298']['y_expected'].append(y_expected[0])
        self.data['tests']['test_298']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5807, max_value=0.5914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_299(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_299']['n_samples'] += 1
        self.data['tests']['test_299']['samples'].append(x_test)
        self.data['tests']['test_299']['y_expected'].append(y_expected[0])
        self.data['tests']['test_299']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5917, max_value=0.5984, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_300(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_300']['n_samples'] += 1
        self.data['tests']['test_300']['samples'].append(x_test)
        self.data['tests']['test_300']['y_expected'].append(y_expected[0])
        self.data['tests']['test_300']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1388, allow_nan=False),
           st.floats(min_value=0.5987, max_value=0.6434, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.6909, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_301(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_301']['n_samples'] += 1
        self.data['tests']['test_301']['samples'].append(x_test)
        self.data['tests']['test_301']['y_expected'].append(y_expected[0])
        self.data['tests']['test_301']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1388, allow_nan=False),
           st.floats(min_value=0.5987, max_value=0.6434, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6912, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_302(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_302']['n_samples'] += 1
        self.data['tests']['test_302']['samples'].append(x_test)
        self.data['tests']['test_302']['y_expected'].append(y_expected[0])
        self.data['tests']['test_302']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1391, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5987, max_value=0.6434, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_303(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_303']['n_samples'] += 1
        self.data['tests']['test_303']['samples'].append(x_test)
        self.data['tests']['test_303']['y_expected'].append(y_expected[0])
        self.data['tests']['test_303']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.6437, max_value=0.6588, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_304(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_304']['n_samples'] += 1
        self.data['tests']['test_304']['samples'].append(x_test)
        self.data['tests']['test_304']['y_expected'].append(y_expected[0])
        self.data['tests']['test_304']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.6591, max_value=0.7229, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_305(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_305']['n_samples'] += 1
        self.data['tests']['test_305']['samples'].append(x_test)
        self.data['tests']['test_305']['y_expected'].append(y_expected[0])
        self.data['tests']['test_305']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7232, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.2892, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_306(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_306']['n_samples'] += 1
        self.data['tests']['test_306']['samples'].append(x_test)
        self.data['tests']['test_306']['y_expected'].append(y_expected[0])
        self.data['tests']['test_306']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2464, allow_nan=False),
           st.floats(min_value=0.7232, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2889, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_307(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_307']['n_samples'] += 1
        self.data['tests']['test_307']['samples'].append(x_test)
        self.data['tests']['test_307']['y_expected'].append(y_expected[0])
        self.data['tests']['test_307']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2467, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7232, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7271, max_value=1.4643, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2889, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_308(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_308']['n_samples'] += 1
        self.data['tests']['test_308']['samples'].append(x_test)
        self.data['tests']['test_308']['y_expected'].append(y_expected[0])
        self.data['tests']['test_308']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4646, max_value=1.6548, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_309(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_309']['n_samples'] += 1
        self.data['tests']['test_309']['samples'].append(x_test)
        self.data['tests']['test_309']['y_expected'].append(y_expected[0])
        self.data['tests']['test_309']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6551, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.2511, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_310(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_310']['n_samples'] += 1
        self.data['tests']['test_310']['samples'].append(x_test)
        self.data['tests']['test_310']['y_expected'].append(y_expected[0])
        self.data['tests']['test_310']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2209, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6551, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2508, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_311(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_311']['n_samples'] += 1
        self.data['tests']['test_311']['samples'].append(x_test)
        self.data['tests']['test_311']['y_expected'].append(y_expected[0])
        self.data['tests']['test_311']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2212, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6551, max_value=1.8259, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2508, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_312(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_312']['n_samples'] += 1
        self.data['tests']['test_312']['samples'].append(x_test)
        self.data['tests']['test_312']['y_expected'].append(y_expected[0])
        self.data['tests']['test_312']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2212, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5687, max_value=0.7663, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8262, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2508, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_313(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_313']['n_samples'] += 1
        self.data['tests']['test_313']['samples'].append(x_test)
        self.data['tests']['test_313']['y_expected'].append(y_expected[0])
        self.data['tests']['test_313']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3816, max_value=0.4534, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_314(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_314']['n_samples'] += 1
        self.data['tests']['test_314']['samples'].append(x_test)
        self.data['tests']['test_314']['y_expected'].append(y_expected[0])
        self.data['tests']['test_314']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4537, max_value=0.7853, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_315(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_315']['n_samples'] += 1
        self.data['tests']['test_315']['samples'].append(x_test)
        self.data['tests']['test_315']['y_expected'].append(y_expected[0])
        self.data['tests']['test_315']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7856, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0173, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.2971, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_316(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_316']['n_samples'] += 1
        self.data['tests']['test_316']['samples'].append(x_test)
        self.data['tests']['test_316']['y_expected'].append(y_expected[0])
        self.data['tests']['test_316']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7856, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0176, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.2971, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_317(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_317']['n_samples'] += 1
        self.data['tests']['test_317']['samples'].append(x_test)
        self.data['tests']['test_317']['y_expected'].append(y_expected[0])
        self.data['tests']['test_317']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2044, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7856, max_value=0.8229, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2968, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_318(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_318']['n_samples'] += 1
        self.data['tests']['test_318']['samples'].append(x_test)
        self.data['tests']['test_318']['y_expected'].append(y_expected[0])
        self.data['tests']['test_318']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2047, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7856, max_value=0.8229, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2968, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_319(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_319']['n_samples'] += 1
        self.data['tests']['test_319']['samples'].append(x_test)
        self.data['tests']['test_319']['y_expected'].append(y_expected[0])
        self.data['tests']['test_319']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8232, max_value=0.9559, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2968, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_320(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_320']['n_samples'] += 1
        self.data['tests']['test_320']['samples'].append(x_test)
        self.data['tests']['test_320']['y_expected'].append(y_expected[0])
        self.data['tests']['test_320']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.7666, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9562, max_value=0.9624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2968, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_321(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_321']['n_samples'] += 1
        self.data['tests']['test_321']['samples'].append(x_test)
        self.data['tests']['test_321']['y_expected'].append(y_expected[0])
        self.data['tests']['test_321']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.5502, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9627, max_value=1.2084, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.3654, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_322(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_322']['n_samples'] += 1
        self.data['tests']['test_322']['samples'].append(x_test)
        self.data['tests']['test_322']['y_expected'].append(y_expected[0])
        self.data['tests']['test_322']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2234, allow_nan=False),
           st.floats(min_value=0.5502, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2087, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.3654, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_323(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_323']['n_samples'] += 1
        self.data['tests']['test_323']['samples'].append(x_test)
        self.data['tests']['test_323']['y_expected'].append(y_expected[0])
        self.data['tests']['test_323']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2237, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5502, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2087, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.3654, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_324(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_324']['n_samples'] += 1
        self.data['tests']['test_324']['samples'].append(x_test)
        self.data['tests']['test_324']['y_expected'].append(y_expected[0])
        self.data['tests']['test_324']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2013, allow_nan=False),
           st.floats(min_value=0.5502, max_value=0.7419, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9627, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3657, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_325(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_325']['n_samples'] += 1
        self.data['tests']['test_325']['samples'].append(x_test)
        self.data['tests']['test_325']['y_expected'].append(y_expected[0])
        self.data['tests']['test_325']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2013, allow_nan=False),
           st.floats(min_value=0.7422, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9627, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3657, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_326(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_326']['n_samples'] += 1
        self.data['tests']['test_326']['samples'].append(x_test)
        self.data['tests']['test_326']['y_expected'].append(y_expected[0])
        self.data['tests']['test_326']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2016, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5502, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9627, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3657, max_value=1.9293, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_327(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_327']['n_samples'] += 1
        self.data['tests']['test_327']['samples'].append(x_test)
        self.data['tests']['test_327']['y_expected'].append(y_expected[0])
        self.data['tests']['test_327']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.0988, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.9433, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_328(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_328']['n_samples'] += 1
        self.data['tests']['test_328']['samples'].append(x_test)
        self.data['tests']['test_328']['y_expected'].append(y_expected[0])
        self.data['tests']['test_328']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0991, max_value=0.1463, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.9433, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_329(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_329']['n_samples'] += 1
        self.data['tests']['test_329']['samples'].append(x_test)
        self.data['tests']['test_329']['y_expected'].append(y_expected[0])
        self.data['tests']['test_329']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1466, max_value=0.1479, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.9433, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_330(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_330']['n_samples'] += 1
        self.data['tests']['test_330']['samples'].append(x_test)
        self.data['tests']['test_330']['y_expected'].append(y_expected[0])
        self.data['tests']['test_330']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1482, max_value=0.1503, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.9433, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_331(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_331']['n_samples'] += 1
        self.data['tests']['test_331']['samples'].append(x_test)
        self.data['tests']['test_331']['y_expected'].append(y_expected[0])
        self.data['tests']['test_331']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1503, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9436, max_value=1.2368, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_332(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_332']['n_samples'] += 1
        self.data['tests']['test_332']['samples'].append(x_test)
        self.data['tests']['test_332']['y_expected'].append(y_expected[0])
        self.data['tests']['test_332']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1506, max_value=0.2169, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.7247, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_333(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_333']['n_samples'] += 1
        self.data['tests']['test_333']['samples'].append(x_test)
        self.data['tests']['test_333']['y_expected'].append(y_expected[0])
        self.data['tests']['test_333']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1506, max_value=0.2169, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=2.0166, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=-0.7247, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_334(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_334']['n_samples'] += 1
        self.data['tests']['test_334']['samples'].append(x_test)
        self.data['tests']['test_334']['y_expected'].append(y_expected[0])
        self.data['tests']['test_334']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1506, max_value=0.2169, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.0708, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7244, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_335(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_335']['n_samples'] += 1
        self.data['tests']['test_335']['samples'].append(x_test)
        self.data['tests']['test_335']['y_expected'].append(y_expected[0])
        self.data['tests']['test_335']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1506, max_value=0.1613, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=2.0711, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7244, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_336(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_336']['n_samples'] += 1
        self.data['tests']['test_336']['samples'].append(x_test)
        self.data['tests']['test_336']['y_expected'].append(y_expected[0])
        self.data['tests']['test_336']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1616, max_value=0.2169, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=2.0711, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7244, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_337(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_337']['n_samples'] += 1
        self.data['tests']['test_337']['samples'].append(x_test)
        self.data['tests']['test_337']['y_expected'].append(y_expected[0])
        self.data['tests']['test_337']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2172, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.5278, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_338(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_338']['n_samples'] += 1
        self.data['tests']['test_338']['samples'].append(x_test)
        self.data['tests']['test_338']['y_expected'].append(y_expected[0])
        self.data['tests']['test_338']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2172, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5281, max_value=0.5983, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_339(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_339']['n_samples'] += 1
        self.data['tests']['test_339']['samples'].append(x_test)
        self.data['tests']['test_339']['y_expected'].append(y_expected[0])
        self.data['tests']['test_339']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2172, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5986, max_value=1.2809, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=1.9296, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7708, max_value=1.9793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_340(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_340']['n_samples'] += 1
        self.data['tests']['test_340']['samples'].append(x_test)
        self.data['tests']['test_340']['y_expected'].append(y_expected[0])
        self.data['tests']['test_340']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=1.2812, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.9793, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_341(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_341']['n_samples'] += 1
        self.data['tests']['test_341']['samples'].append(x_test)
        self.data['tests']['test_341']['y_expected'].append(y_expected[0])
        self.data['tests']['test_341']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.3228, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.9796, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_342(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_342']['n_samples'] += 1
        self.data['tests']['test_342']['samples'].append(x_test)
        self.data['tests']['test_342']['y_expected'].append(y_expected[0])
        self.data['tests']['test_342']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.4967, max_value=0.6479, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=2.3231, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.77, -0.597, -0.486, -0.074, 0.125, 0.253, 0.542, 0.581, 0.677, 1.298]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_343(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_343']['n_samples'] += 1
        self.data['tests']['test_343']['samples'].append(x_test)
        self.data['tests']['test_343']['y_expected'].append(y_expected[0])
        self.data['tests']['test_343']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2914, allow_nan=False),
           st.floats(min_value=0.6482, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2368, allow_nan=False),
           st.floats(min_value=2.3231, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.444, -0.387, -0.24, -0.23, 0.213, 0.543, 0.634, 0.965, 1.46, 1.462]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_344(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_344']['n_samples'] += 1
        self.data['tests']['test_344']['samples'].append(x_test)
        self.data['tests']['test_344']['y_expected'].append(y_expected[0])
        self.data['tests']['test_344']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.1037, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.3363, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_345(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_345']['n_samples'] += 1
        self.data['tests']['test_345']['samples'].append(x_test)
        self.data['tests']['test_345']['y_expected'].append(y_expected[0])
        self.data['tests']['test_345']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=-0.2602, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.3366, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_346(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_346']['n_samples'] += 1
        self.data['tests']['test_346']['samples'].append(x_test)
        self.data['tests']['test_346']['y_expected'].append(y_expected[0])
        self.data['tests']['test_346']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.2599, max_value=-0.1037, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.3366, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9249, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_347(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_347']['n_samples'] += 1
        self.data['tests']['test_347']['samples'].append(x_test)
        self.data['tests']['test_347']['y_expected'].append(y_expected[0])
        self.data['tests']['test_347']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.2599, max_value=-0.1037, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.3366, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9252, max_value=0.9708, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_348(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_348']['n_samples'] += 1
        self.data['tests']['test_348']['samples'].append(x_test)
        self.data['tests']['test_348']['y_expected'].append(y_expected[0])
        self.data['tests']['test_348']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.2599, max_value=-0.1037, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.3366, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9711, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_349(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_349']['n_samples'] += 1
        self.data['tests']['test_349']['samples'].append(x_test)
        self.data['tests']['test_349']['y_expected'].append(y_expected[0])
        self.data['tests']['test_349']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1034, max_value=0.1494, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.2464, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_350(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_350']['n_samples'] += 1
        self.data['tests']['test_350']['samples'].append(x_test)
        self.data['tests']['test_350']['y_expected'].append(y_expected[0])
        self.data['tests']['test_350']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1034, max_value=0.1494, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2467, max_value=1.2548, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_351(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_351']['n_samples'] += 1
        self.data['tests']['test_351']['samples'].append(x_test)
        self.data['tests']['test_351']['y_expected'].append(y_expected[0])
        self.data['tests']['test_351']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1034, max_value=0.1494, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2551, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9683, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_352(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_352']['n_samples'] += 1
        self.data['tests']['test_352']['samples'].append(x_test)
        self.data['tests']['test_352']['y_expected'].append(y_expected[0])
        self.data['tests']['test_352']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1034, max_value=0.1494, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2551, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9686, max_value=0.9908, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_353(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_353']['n_samples'] += 1
        self.data['tests']['test_353']['samples'].append(x_test)
        self.data['tests']['test_353']['y_expected'].append(y_expected[0])
        self.data['tests']['test_353']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.1034, max_value=0.1494, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2551, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9911, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_354(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_354']['n_samples'] += 1
        self.data['tests']['test_354']['samples'].append(x_test)
        self.data['tests']['test_354']['y_expected'].append(y_expected[0])
        self.data['tests']['test_354']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.5883, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_355(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_355']['n_samples'] += 1
        self.data['tests']['test_355']['samples'].append(x_test)
        self.data['tests']['test_355']['y_expected'].append(y_expected[0])
        self.data['tests']['test_355']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2869, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0744, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_356(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_356']['n_samples'] += 1
        self.data['tests']['test_356']['samples'].append(x_test)
        self.data['tests']['test_356']['y_expected'].append(y_expected[0])
        self.data['tests']['test_356']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2872, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0744, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_357(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_357']['n_samples'] += 1
        self.data['tests']['test_357']['samples'].append(x_test)
        self.data['tests']['test_357']['y_expected'].append(y_expected[0])
        self.data['tests']['test_357']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.1919, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.4664, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_358(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_358']['n_samples'] += 1
        self.data['tests']['test_358']['samples'].append(x_test)
        self.data['tests']['test_358']['y_expected'].append(y_expected[0])
        self.data['tests']['test_358']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1922, max_value=0.2004, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.4664, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_359(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_359']['n_samples'] += 1
        self.data['tests']['test_359']['samples'].append(x_test)
        self.data['tests']['test_359']['y_expected'].append(y_expected[0])
        self.data['tests']['test_359']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2007, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.4664, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_360(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_360']['n_samples'] += 1
        self.data['tests']['test_360']['samples'].append(x_test)
        self.data['tests']['test_360']['y_expected'].append(y_expected[0])
        self.data['tests']['test_360']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4667, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6347, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_361(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_361']['n_samples'] += 1
        self.data['tests']['test_361']['samples'].append(x_test)
        self.data['tests']['test_361']['y_expected'].append(y_expected[0])
        self.data['tests']['test_361']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4667, max_value=1.5388, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6344, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_362(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_362']['n_samples'] += 1
        self.data['tests']['test_362']['samples'].append(x_test)
        self.data['tests']['test_362']['y_expected'].append(y_expected[0])
        self.data['tests']['test_362']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5391, max_value=1.5643, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6344, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_363(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_363']['n_samples'] += 1
        self.data['tests']['test_363']['samples'].append(x_test)
        self.data['tests']['test_363']['y_expected'].append(y_expected[0])
        self.data['tests']['test_363']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=0.9063, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.5646, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0747, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6344, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_364(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_364']['n_samples'] += 1
        self.data['tests']['test_364']['samples'].append(x_test)
        self.data['tests']['test_364']['y_expected'].append(y_expected[0])
        self.data['tests']['test_364']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2829, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9066, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_365(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_365']['n_samples'] += 1
        self.data['tests']['test_365']['samples'].append(x_test)
        self.data['tests']['test_365']['y_expected'].append(y_expected[0])
        self.data['tests']['test_365']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2832, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9066, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.8029, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.7654, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_366(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_366']['n_samples'] += 1
        self.data['tests']['test_366']['samples'].append(x_test)
        self.data['tests']['test_366']['y_expected'].append(y_expected[0])
        self.data['tests']['test_366']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8032, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.7267, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_367(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_367']['n_samples'] += 1
        self.data['tests']['test_367']['samples'].append(x_test)
        self.data['tests']['test_367']['y_expected'].append(y_expected[0])
        self.data['tests']['test_367']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5886, max_value=1.1724, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8032, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7264, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_368(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_368']['n_samples'] += 1
        self.data['tests']['test_368']['samples'].append(x_test)
        self.data['tests']['test_368']['y_expected'].append(y_expected[0])
        self.data['tests']['test_368']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1727, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2371, max_value=1.7529, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8032, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.7264, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_369(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_369']['n_samples'] += 1
        self.data['tests']['test_369']['samples'].append(x_test)
        self.data['tests']['test_369']['y_expected'].append(y_expected[0])
        self.data['tests']['test_369']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.7532, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6121, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_370(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_370']['n_samples'] += 1
        self.data['tests']['test_370']['samples'].append(x_test)
        self.data['tests']['test_370']['y_expected'].append(y_expected[0])
        self.data['tests']['test_370']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2434, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.7532, max_value=2.4498, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_371(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_371']['n_samples'] += 1
        self.data['tests']['test_371']['samples'].append(x_test)
        self.data['tests']['test_371']['y_expected'].append(y_expected[0])
        self.data['tests']['test_371']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2437, max_value=0.2444, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.7532, max_value=2.4498, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_372(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_372']['n_samples'] += 1
        self.data['tests']['test_372']['samples'].append(x_test)
        self.data['tests']['test_372']['y_expected'].append(y_expected[0])
        self.data['tests']['test_372']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2447, max_value=0.2714, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.7532, max_value=1.9669, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_373(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_373']['n_samples'] += 1
        self.data['tests']['test_373']['samples'].append(x_test)
        self.data['tests']['test_373']['y_expected'].append(y_expected[0])
        self.data['tests']['test_373']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2447, max_value=0.2469, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.9672, max_value=2.1678, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_374(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_374']['n_samples'] += 1
        self.data['tests']['test_374']['samples'].append(x_test)
        self.data['tests']['test_374']['y_expected'].append(y_expected[0])
        self.data['tests']['test_374']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2472, max_value=0.2714, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.9672, max_value=2.1678, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_375(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_375']['n_samples'] += 1
        self.data['tests']['test_375']['samples'].append(x_test)
        self.data['tests']['test_375']['y_expected'].append(y_expected[0])
        self.data['tests']['test_375']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2447, max_value=0.2714, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=2.1681, max_value=2.4498, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_376(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_376']['n_samples'] += 1
        self.data['tests']['test_376']['samples'].append(x_test)
        self.data['tests']['test_376']['y_expected'].append(y_expected[0])
        self.data['tests']['test_376']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2717, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.7532, max_value=2.4498, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_377(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_377']['n_samples'] += 1
        self.data['tests']['test_377']['samples'].append(x_test)
        self.data['tests']['test_377']['y_expected'].append(y_expected[0])
        self.data['tests']['test_377']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.6708, allow_nan=False),
           st.floats(min_value=2.4501, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_378(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_378']['n_samples'] += 1
        self.data['tests']['test_378']['samples'].append(x_test)
        self.data['tests']['test_378']['y_expected'].append(y_expected[0])
        self.data['tests']['test_378']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.6708, allow_nan=False),
           st.floats(min_value=2.4501, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9181, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_379(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_379']['n_samples'] += 1
        self.data['tests']['test_379']['samples'].append(x_test)
        self.data['tests']['test_379']['y_expected'].append(y_expected[0])
        self.data['tests']['test_379']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1497, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6711, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.4501, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6118, max_value=0.7654, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_380(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_380']['n_samples'] += 1
        self.data['tests']['test_380']['samples'].append(x_test)
        self.data['tests']['test_380']['y_expected'].append(y_expected[0])
        self.data['tests']['test_380']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.1849, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2371, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7657, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_381(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_381']['n_samples'] += 1
        self.data['tests']['test_381']['samples'].append(x_test)
        self.data['tests']['test_381']['y_expected'].append(y_expected[0])
        self.data['tests']['test_381']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1852, max_value=0.1928, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2371, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7657, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_382(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_382']['n_samples'] += 1
        self.data['tests']['test_382']['samples'].append(x_test)
        self.data['tests']['test_382']['y_expected'].append(y_expected[0])
        self.data['tests']['test_382']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.1931, max_value=0.2914, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=1.2371, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7657, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_383(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_383']['n_samples'] += 1
        self.data['tests']['test_383']['samples'].append(x_test)
        self.data['tests']['test_383']['y_expected'].append(y_expected[0])
        self.data['tests']['test_383']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4984, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.7263, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.4753, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_384(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_384']['n_samples'] += 1
        self.data['tests']['test_384']['samples'].append(x_test)
        self.data['tests']['test_384']['y_expected'].append(y_expected[0])
        self.data['tests']['test_384']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3993, allow_nan=False),
           st.floats(min_value=0.7266, max_value=1.7994, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.2963, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_385(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_385']['n_samples'] += 1
        self.data['tests']['test_385']['samples'].append(x_test)
        self.data['tests']['test_385']['y_expected'].append(y_expected[0])
        self.data['tests']['test_385']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.3993, allow_nan=False),
           st.floats(min_value=0.7266, max_value=1.7994, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2966, max_value=2.4753, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_386(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_386']['n_samples'] += 1
        self.data['tests']['test_386']['samples'].append(x_test)
        self.data['tests']['test_386']['y_expected'].append(y_expected[0])
        self.data['tests']['test_386']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3996, max_value=0.4984, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7266, max_value=1.7994, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.4753, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_387(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_387']['n_samples'] += 1
        self.data['tests']['test_387']['samples'].append(x_test)
        self.data['tests']['test_387']['y_expected'].append(y_expected[0])
        self.data['tests']['test_387']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4984, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.7994, allow_nan=False),
           st.floats(min_value=2.4756, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_388(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_388']['n_samples'] += 1
        self.data['tests']['test_388']['samples'].append(x_test)
        self.data['tests']['test_388']['y_expected'].append(y_expected[0])
        self.data['tests']['test_388']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4984, allow_nan=False),
           st.floats(min_value=1.7997, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.1709, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_389(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_389']['n_samples'] += 1
        self.data['tests']['test_389']['samples'].append(x_test)
        self.data['tests']['test_389']['y_expected'].append(y_expected[0])
        self.data['tests']['test_389']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4984, allow_nan=False),
           st.floats(min_value=1.7997, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1712, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_390(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_390']['n_samples'] += 1
        self.data['tests']['test_390']['samples'].append(x_test)
        self.data['tests']['test_390']['y_expected'].append(y_expected[0])
        self.data['tests']['test_390']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2224, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7869, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.1032, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_391(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_391']['n_samples'] += 1
        self.data['tests']['test_391']['samples'].append(x_test)
        self.data['tests']['test_391']['y_expected'].append(y_expected[0])
        self.data['tests']['test_391']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.2224, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7869, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1029, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_392(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_392']['n_samples'] += 1
        self.data['tests']['test_392']['samples'].append(x_test)
        self.data['tests']['test_392']['y_expected'].append(y_expected[0])
        self.data['tests']['test_392']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.6194, allow_nan=False),
           st.floats(min_value=0.7872, max_value=1.0584, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_393(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_393']['n_samples'] += 1
        self.data['tests']['test_393']['samples'].append(x_test)
        self.data['tests']['test_393']['y_expected'].append(y_expected[0])
        self.data['tests']['test_393']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=0.8069, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.6194, allow_nan=False),
           st.floats(min_value=1.0587, max_value=2.3008, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_394(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_394']['n_samples'] += 1
        self.data['tests']['test_394']['samples'].append(x_test)
        self.data['tests']['test_394']['y_expected'].append(y_expected[0])
        self.data['tests']['test_394']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8072, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=0.6194, allow_nan=False),
           st.floats(min_value=1.0587, max_value=2.3008, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_395(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_395']['n_samples'] += 1
        self.data['tests']['test_395']['samples'].append(x_test)
        self.data['tests']['test_395']['y_expected'].append(y_expected[0])
        self.data['tests']['test_395']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6197, max_value=1.2224, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7872, max_value=2.3008, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_396(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_396']['n_samples'] += 1
        self.data['tests']['test_396']['samples'].append(x_test)
        self.data['tests']['test_396']['y_expected'].append(y_expected[0])
        self.data['tests']['test_396']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.0384, allow_nan=False),
           st.floats(min_value=2.3011, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_397(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_397']['n_samples'] += 1
        self.data['tests']['test_397']['samples'].append(x_test)
        self.data['tests']['test_397']['y_expected'].append(y_expected[0])
        self.data['tests']['test_397']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0387, max_value=1.2224, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3011, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_398(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_398']['n_samples'] += 1
        self.data['tests']['test_398']['samples'].append(x_test)
        self.data['tests']['test_398']['y_expected'].append(y_expected[0])
        self.data['tests']['test_398']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.6157, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_399(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_399']['n_samples'] += 1
        self.data['tests']['test_399']['samples'].append(x_test)
        self.data['tests']['test_399']['y_expected'].append(y_expected[0])
        self.data['tests']['test_399']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2518, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8323, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_400(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_400']['n_samples'] += 1
        self.data['tests']['test_400']['samples'].append(x_test)
        self.data['tests']['test_400']['y_expected'].append(y_expected[0])
        self.data['tests']['test_400']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.3768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2521, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8323, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_401(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_401']['n_samples'] += 1
        self.data['tests']['test_401']['samples'].append(x_test)
        self.data['tests']['test_401']['y_expected'].append(y_expected[0])
        self.data['tests']['test_401']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3771, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2521, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8323, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_402(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_402']['n_samples'] += 1
        self.data['tests']['test_402']['samples'].append(x_test)
        self.data['tests']['test_402']['y_expected'].append(y_expected[0])
        self.data['tests']['test_402']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8326, max_value=0.8624, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_403(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_403']['n_samples'] += 1
        self.data['tests']['test_403']['samples'].append(x_test)
        self.data['tests']['test_403']['y_expected'].append(y_expected[0])
        self.data['tests']['test_403']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.3624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=1.6764, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8627, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_404(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_404']['n_samples'] += 1
        self.data['tests']['test_404']['samples'].append(x_test)
        self.data['tests']['test_404']['y_expected'].append(y_expected[0])
        self.data['tests']['test_404']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.3624, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6767, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8627, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_405(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_405']['n_samples'] += 1
        self.data['tests']['test_405']['samples'].append(x_test)
        self.data['tests']['test_405']['y_expected'].append(y_expected[0])
        self.data['tests']['test_405']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3627, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8627, max_value=1.6354, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.5852, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_406(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_406']['n_samples'] += 1
        self.data['tests']['test_406']['samples'].append(x_test)
        self.data['tests']['test_406']['y_expected'].append(y_expected[0])
        self.data['tests']['test_406']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3627, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6357, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.5852, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_407(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_407']['n_samples'] += 1
        self.data['tests']['test_407']['samples'].append(x_test)
        self.data['tests']['test_407']['y_expected'].append(y_expected[0])
        self.data['tests']['test_407']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3627, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.2984, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8627, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.5849, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_408(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_408']['n_samples'] += 1
        self.data['tests']['test_408']['samples'].append(x_test)
        self.data['tests']['test_408']['y_expected'].append(y_expected[0])
        self.data['tests']['test_408']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6154, max_value=-0.3867, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_409(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_409']['n_samples'] += 1
        self.data['tests']['test_409']['samples'].append(x_test)
        self.data['tests']['test_409']['y_expected'].append(y_expected[0])
        self.data['tests']['test_409']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2987, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3864, max_value=-0.1056, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_410(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_410']['n_samples'] += 1
        self.data['tests']['test_410']['samples'].append(x_test)
        self.data['tests']['test_410']['y_expected'].append(y_expected[0])
        self.data['tests']['test_410']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=0.7349, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=0.8159, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_411(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_411']['n_samples'] += 1
        self.data['tests']['test_411']['samples'].append(x_test)
        self.data['tests']['test_411']['y_expected'].append(y_expected[0])
        self.data['tests']['test_411']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7352, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8964, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=0.8159, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_412(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_412']['n_samples'] += 1
        self.data['tests']['test_412']['samples'].append(x_test)
        self.data['tests']['test_412']['y_expected'].append(y_expected[0])
        self.data['tests']['test_412']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8964, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8162, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_413(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_413']['n_samples'] += 1
        self.data['tests']['test_413']['samples'].append(x_test)
        self.data['tests']['test_413']['y_expected'].append(y_expected[0])
        self.data['tests']['test_413']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=1.7019, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8967, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=0.9219, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_414(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_414']['n_samples'] += 1
        self.data['tests']['test_414']['samples'].append(x_test)
        self.data['tests']['test_414']['y_expected'].append(y_expected[0])
        self.data['tests']['test_414']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=1.7019, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8967, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9222, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_415(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_415']['n_samples'] += 1
        self.data['tests']['test_415']['samples'].append(x_test)
        self.data['tests']['test_415']['y_expected'].append(y_expected[0])
        self.data['tests']['test_415']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7022, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8967, max_value=1.2618, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=0.6793, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_416(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_416']['n_samples'] += 1
        self.data['tests']['test_416']['samples'].append(x_test)
        self.data['tests']['test_416']['y_expected'].append(y_expected[0])
        self.data['tests']['test_416']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7022, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8967, max_value=1.2618, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6796, max_value=0.8033, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_417(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_417']['n_samples'] += 1
        self.data['tests']['test_417']['samples'].append(x_test)
        self.data['tests']['test_417']['y_expected'].append(y_expected[0])
        self.data['tests']['test_417']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7022, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8967, max_value=1.2618, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8036, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_418(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_418']['n_samples'] += 1
        self.data['tests']['test_418']['samples'].append(x_test)
        self.data['tests']['test_418']['y_expected'].append(y_expected[0])
        self.data['tests']['test_418']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=1.0163, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7022, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2621, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_419(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_419']['n_samples'] += 1
        self.data['tests']['test_419']['samples'].append(x_test)
        self.data['tests']['test_419']['y_expected'].append(y_expected[0])
        self.data['tests']['test_419']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0166, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=1.8518, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8654, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_420(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_420']['n_samples'] += 1
        self.data['tests']['test_420']['samples'].append(x_test)
        self.data['tests']['test_420']['y_expected'].append(y_expected[0])
        self.data['tests']['test_420']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0166, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8521, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8654, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_421(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_421']['n_samples'] += 1
        self.data['tests']['test_421']['samples'].append(x_test)
        self.data['tests']['test_421']['y_expected'].append(y_expected[0])
        self.data['tests']['test_421']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4019, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0166, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8657, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_422(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_422']['n_samples'] += 1
        self.data['tests']['test_422']['samples'].append(x_test)
        self.data['tests']['test_422']['y_expected'].append(y_expected[0])
        self.data['tests']['test_422']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4022, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0166, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8657, max_value=1.0528, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_423(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_423']['n_samples'] += 1
        self.data['tests']['test_423']['samples'].append(x_test)
        self.data['tests']['test_423']['y_expected'].append(y_expected[0])
        self.data['tests']['test_423']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4022, max_value=0.4384, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0166, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2227, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0531, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.1053, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_424(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_424']['n_samples'] += 1
        self.data['tests']['test_424']['samples'].append(x_test)
        self.data['tests']['test_424']['y_expected'].append(y_expected[0])
        self.data['tests']['test_424']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=-0.4122, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_425(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_425']['n_samples'] += 1
        self.data['tests']['test_425']['samples'].append(x_test)
        self.data['tests']['test_425']['y_expected'].append(y_expected[0])
        self.data['tests']['test_425']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.0193, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.7174, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7388, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_426(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_426']['n_samples'] += 1
        self.data['tests']['test_426']['samples'].append(x_test)
        self.data['tests']['test_426']['y_expected'].append(y_expected[0])
        self.data['tests']['test_426']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0196, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.7174, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6899, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_427(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_427']['n_samples'] += 1
        self.data['tests']['test_427']['samples'].append(x_test)
        self.data['tests']['test_427']['y_expected'].append(y_expected[0])
        self.data['tests']['test_427']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0196, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.7174, allow_nan=False),
           st.floats(min_value=0.6902, max_value=0.7388, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_428(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_428']['n_samples'] += 1
        self.data['tests']['test_428']['samples'].append(x_test)
        self.data['tests']['test_428']['y_expected'].append(y_expected[0])
        self.data['tests']['test_428']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7177, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7388, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3037, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_429(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_429']['n_samples'] += 1
        self.data['tests']['test_429']['samples'].append(x_test)
        self.data['tests']['test_429']['y_expected'].append(y_expected[0])
        self.data['tests']['test_429']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7177, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7388, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3034, max_value=0.3714, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_430(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_430']['n_samples'] += 1
        self.data['tests']['test_430']['samples'].append(x_test)
        self.data['tests']['test_430']['y_expected'].append(y_expected[0])
        self.data['tests']['test_430']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7177, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.7388, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3717, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_431(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_431']['n_samples'] += 1
        self.data['tests']['test_431']['samples'].append(x_test)
        self.data['tests']['test_431']['y_expected'].append(y_expected[0])
        self.data['tests']['test_431']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=0.5843, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_432(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_432']['n_samples'] += 1
        self.data['tests']['test_432']['samples'].append(x_test)
        self.data['tests']['test_432']['y_expected'].append(y_expected[0])
        self.data['tests']['test_432']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=1.6933, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.1043, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_433(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_433']['n_samples'] += 1
        self.data['tests']['test_433']['samples'].append(x_test)
        self.data['tests']['test_433']['y_expected'].append(y_expected[0])
        self.data['tests']['test_433']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=0.8218, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6936, max_value=1.8124, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.1043, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_434(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_434']['n_samples'] += 1
        self.data['tests']['test_434']['samples'].append(x_test)
        self.data['tests']['test_434']['y_expected'].append(y_expected[0])
        self.data['tests']['test_434']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8221, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6936, max_value=1.8124, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.1043, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_435(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_435']['n_samples'] += 1
        self.data['tests']['test_435']['samples'].append(x_test)
        self.data['tests']['test_435']['y_expected'].append(y_expected[0])
        self.data['tests']['test_435']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=0.9364, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8126, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.0848, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_436(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_436']['n_samples'] += 1
        self.data['tests']['test_436']['samples'].append(x_test)
        self.data['tests']['test_436']['y_expected'].append(y_expected[0])
        self.data['tests']['test_436']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9367, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8126, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.0848, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.0278, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_437(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_437']['n_samples'] += 1
        self.data['tests']['test_437']['samples'].append(x_test)
        self.data['tests']['test_437']['y_expected'].append(y_expected[0])
        self.data['tests']['test_437']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9367, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8126, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.0848, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0281, max_value=0.5674, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_438(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_438']['n_samples'] += 1
        self.data['tests']['test_438']['samples'].append(x_test)
        self.data['tests']['test_438']['y_expected'].append(y_expected[0])
        self.data['tests']['test_438']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8126, max_value=2.3374, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0851, max_value=1.1043, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_439(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_439']['n_samples'] += 1
        self.data['tests']['test_439']['samples'].append(x_test)
        self.data['tests']['test_439']['y_expected'].append(y_expected[0])
        self.data['tests']['test_439']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5846, max_value=0.5888, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.1046, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_440(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_440']['n_samples'] += 1
        self.data['tests']['test_440']['samples'].append(x_test)
        self.data['tests']['test_440']['y_expected'].append(y_expected[0])
        self.data['tests']['test_440']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=0.4723, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5891, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.1046, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_441(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_441']['n_samples'] += 1
        self.data['tests']['test_441']['samples'].append(x_test)
        self.data['tests']['test_441']['y_expected'].append(y_expected[0])
        self.data['tests']['test_441']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4726, max_value=0.4749, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5891, max_value=0.8859, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.1046, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_442(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_442']['n_samples'] += 1
        self.data['tests']['test_442']['samples'].append(x_test)
        self.data['tests']['test_442']['y_expected'].append(y_expected[0])
        self.data['tests']['test_442']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4726, max_value=0.4749, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8862, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.1046, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_443(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_443']['n_samples'] += 1
        self.data['tests']['test_443']['samples'].append(x_test)
        self.data['tests']['test_443']['y_expected'].append(y_expected[0])
        self.data['tests']['test_443']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4752, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5891, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.1046, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5674, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_444(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_444']['n_samples'] += 1
        self.data['tests']['test_444']['samples'].append(x_test)
        self.data['tests']['test_444']['y_expected'].append(y_expected[0])
        self.data['tests']['test_444']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=0.7038, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5677, max_value=0.5969, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_445(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_445']['n_samples'] += 1
        self.data['tests']['test_445']['samples'].append(x_test)
        self.data['tests']['test_445']['y_expected'].append(y_expected[0])
        self.data['tests']['test_445']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7041, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5677, max_value=0.5969, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_446(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_446']['n_samples'] += 1
        self.data['tests']['test_446']['samples'].append(x_test)
        self.data['tests']['test_446']['y_expected'].append(y_expected[0])
        self.data['tests']['test_446']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.3239, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.7391, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5972, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_447(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_447']['n_samples'] += 1
        self.data['tests']['test_447']['samples'].append(x_test)
        self.data['tests']['test_447']['y_expected'].append(y_expected[0])
        self.data['tests']['test_447']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.3242, max_value=1.3658, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_448(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_448']['n_samples'] += 1
        self.data['tests']['test_448']['samples'].append(x_test)
        self.data['tests']['test_448']['y_expected'].append(y_expected[0])
        self.data['tests']['test_448']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.3661, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4119, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.8058, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_449(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_449']['n_samples'] += 1
        self.data['tests']['test_449']['samples'].append(x_test)
        self.data['tests']['test_449']['y_expected'].append(y_expected[0])
        self.data['tests']['test_449']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4304, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.8061, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_450(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_450']['n_samples'] += 1
        self.data['tests']['test_450']['samples'].append(x_test)
        self.data['tests']['test_450']['y_expected'].append(y_expected[0])
        self.data['tests']['test_450']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4307, max_value=0.5864, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.8061, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2786, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_451(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_451']['n_samples'] += 1
        self.data['tests']['test_451']['samples'].append(x_test)
        self.data['tests']['test_451']['y_expected'].append(y_expected[0])
        self.data['tests']['test_451']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4307, max_value=0.5864, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.8061, max_value=2.0023, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2783, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_452(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_452']['n_samples'] += 1
        self.data['tests']['test_452']['samples'].append(x_test)
        self.data['tests']['test_452']['y_expected'].append(y_expected[0])
        self.data['tests']['test_452']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=0.4604, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4307, max_value=0.5864, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=2.0026, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2783, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_453(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_453']['n_samples'] += 1
        self.data['tests']['test_453']['samples'].append(x_test)
        self.data['tests']['test_453']['y_expected'].append(y_expected[0])
        self.data['tests']['test_453']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4607, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4307, max_value=0.5864, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=2.0026, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2783, max_value=1.1858, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_454(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_454']['n_samples'] += 1
        self.data['tests']['test_454']['samples'].append(x_test)
        self.data['tests']['test_454']['y_expected'].append(y_expected[0])
        self.data['tests']['test_454']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4387, max_value=0.9433, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5867, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.8061, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_455(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_455']['n_samples'] += 1
        self.data['tests']['test_455']['samples'].append(x_test)
        self.data['tests']['test_455']['y_expected'].append(y_expected[0])
        self.data['tests']['test_455']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.9436, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5867, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.8061, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=1.1858, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_456(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_456']['n_samples'] += 1
        self.data['tests']['test_456']['samples'].append(x_test)
        self.data['tests']['test_456']['y_expected'].append(y_expected[0])
        self.data['tests']['test_456']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6058, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1861, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_457(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_457']['n_samples'] += 1
        self.data['tests']['test_457']['samples'].append(x_test)
        self.data['tests']['test_457']['y_expected'].append(y_expected[0])
        self.data['tests']['test_457']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4054, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.6061, max_value=1.2034, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1861, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_458(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_458']['n_samples'] += 1
        self.data['tests']['test_458']['samples'].append(x_test)
        self.data['tests']['test_458']['y_expected'].append(y_expected[0])
        self.data['tests']['test_458']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=0.4054, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=1.2037, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1861, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_459(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_459']['n_samples'] += 1
        self.data['tests']['test_459']['samples'].append(x_test)
        self.data['tests']['test_459']['y_expected'].append(y_expected[0])
        self.data['tests']['test_459']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4057, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=2.3374, allow_nan=False),
           st.floats(min_value=0.6061, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1861, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_460(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_460']['n_samples'] += 1
        self.data['tests']['test_460']['samples'].append(x_test)
        self.data['tests']['test_460']['y_expected'].append(y_expected[0])
        self.data['tests']['test_460']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.0364, allow_nan=False),
           st.floats(min_value=2.3377, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9158, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.655, -0.366, -0.18, -0.176, -0.133, 0.045, 0.228, 0.509, 0.551, 1.159]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_461(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_461']['n_samples'] += 1
        self.data['tests']['test_461']['samples'].append(x_test)
        self.data['tests']['test_461']['y_expected'].append(y_expected[0])
        self.data['tests']['test_461']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0367, max_value=1.4484, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.3377, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.9158, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.145, -0.552, -0.518, -0.269, 0.365, 0.562, 0.65, 0.726, 0.785, 1.264]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_462(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_462']['n_samples'] += 1
        self.data['tests']['test_462']['samples'].append(x_test)
        self.data['tests']['test_462']['y_expected'].append(y_expected[0])
        self.data['tests']['test_462']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=2.3377, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9161, max_value=0.9859, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.849, -0.707, -0.578, -0.486, -0.404, -0.153, 0.639, 0.799, 0.886, 1.03]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_463(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_463']['n_samples'] += 1
        self.data['tests']['test_463']['samples'].append(x_test)
        self.data['tests']['test_463']['y_expected'].append(y_expected[0])
        self.data['tests']['test_463']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2917, max_value=0.4784, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=2.3377, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9862, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.154, -0.047, 0.073, 0.078, 0.087, 0.199, 0.265, 0.418, 1.742, 1.807]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_464(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_464']['n_samples'] += 1
        self.data['tests']['test_464']['samples'].append(x_test)
        self.data['tests']['test_464']['y_expected'].append(y_expected[0])
        self.data['tests']['test_464']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4787, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=2.3377, max_value=2.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9862, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.833, -0.535, -0.512, -0.419, -0.385, -0.188, -0.136, 0.478, 0.998, 1.386]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_465(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_465']['n_samples'] += 1
        self.data['tests']['test_465']['samples'].append(x_test)
        self.data['tests']['test_465']['y_expected'].append(y_expected[0])
        self.data['tests']['test_465']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4787, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.327, max_value=1.4484, allow_nan=False),
           st.floats(min_value=2.4342, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9862, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.115, -0.85, -0.508, -0.374, -0.361, -0.281, -0.244, 0.755, 1.304, 1.493]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_466(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_466']['n_samples'] += 1
        self.data['tests']['test_466']['samples'].append(x_test)
        self.data['tests']['test_466']['y_expected'].append(y_expected[0])
        self.data['tests']['test_466']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-1.7, max_value=0.2753, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.4, 0.135, 0.251, 0.417, 0.996, 1.442, 1.534, 1.591, 2.275, 2.572]),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.599, -0.513, -0.469, -0.319, -0.149, -0.108, 0.048, 0.078, 0.089, 1.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_467(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_467']['n_samples'] += 1
        self.data['tests']['test_467']['samples'].append(x_test)
        self.data['tests']['test_467']['y_expected'].append(y_expected[0])
        self.data['tests']['test_467']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4043, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.6187, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.633, -0.505, -0.045, 0.1, 0.229, 0.416, 0.615, 0.712, 0.949, 2.286]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_468(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_468']['n_samples'] += 1
        self.data['tests']['test_468']['samples'].append(x_test)
        self.data['tests']['test_468']['y_expected'].append(y_expected[0])
        self.data['tests']['test_468']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4043, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6184, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.064, -0.76, -0.731, -0.577, -0.504, -0.379, -0.366, -0.189, 0.563, 0.886]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_469(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_469']['n_samples'] += 1
        self.data['tests']['test_469']['samples'].append(x_test)
        self.data['tests']['test_469']['y_expected'].append(y_expected[0])
        self.data['tests']['test_469']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.823, max_value=-0.6821, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.284, -0.677, -0.538, -0.236, 0.387, 0.922, 1.253, 1.27, 1.604, 1.999]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_470(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_470']['n_samples'] += 1
        self.data['tests']['test_470']['samples'].append(x_test)
        self.data['tests']['test_470']['y_expected'].append(y_expected[0])
        self.data['tests']['test_470']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.7733, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.3156, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_471(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_471']['n_samples'] += 1
        self.data['tests']['test_471']['samples'].append(x_test)
        self.data['tests']['test_471']['y_expected'].append(y_expected[0])
        self.data['tests']['test_471']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.7733, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.3153, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_472(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_472']['n_samples'] += 1
        self.data['tests']['test_472']['samples'].append(x_test)
        self.data['tests']['test_472']['y_expected'].append(y_expected[0])
        self.data['tests']['test_472']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7736, max_value=2.4774, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.762, -0.407, -0.263, -0.229, -0.21, -0.194, -0.148, 0.085, 0.132, 0.632]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_473(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_473']['n_samples'] += 1
        self.data['tests']['test_473']['samples'].append(x_test)
        self.data['tests']['test_473']['y_expected'].append(y_expected[0])
        self.data['tests']['test_473']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.4777, max_value=2.5114, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8144, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.96, -0.769, -0.563, -0.374, -0.077, 0.179, 0.393, 0.565, 0.814, 1.742]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_474(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_474']['n_samples'] += 1
        self.data['tests']['test_474']['samples'].append(x_test)
        self.data['tests']['test_474']['y_expected'].append(y_expected[0])
        self.data['tests']['test_474']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.4777, max_value=2.5114, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8147, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.831, -0.698, -0.421, -0.352, 0.298, 0.604, 0.639, 0.666, 0.682, 0.999]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_475(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_475']['n_samples'] += 1
        self.data['tests']['test_475']['samples'].append(x_test)
        self.data['tests']['test_475']['y_expected'].append(y_expected[0])
        self.data['tests']['test_475']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5117, max_value=2.8514, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8744, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3413, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_476(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_476']['n_samples'] += 1
        self.data['tests']['test_476']['samples'].append(x_test)
        self.data['tests']['test_476']['y_expected'].append(y_expected[0])
        self.data['tests']['test_476']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.8517, max_value=2.9343, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.8744, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3413, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_477(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_477']['n_samples'] += 1
        self.data['tests']['test_477']['samples'].append(x_test)
        self.data['tests']['test_477']['y_expected'].append(y_expected[0])
        self.data['tests']['test_477']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5117, max_value=2.9343, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8747, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3413, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_478(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_478']['n_samples'] += 1
        self.data['tests']['test_478']['samples'].append(x_test)
        self.data['tests']['test_478']['y_expected'].append(y_expected[0])
        self.data['tests']['test_478']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.9346, max_value=2.9753, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3413, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_479(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_479']['n_samples'] += 1
        self.data['tests']['test_479']['samples'].append(x_test)
        self.data['tests']['test_479']['y_expected'].append(y_expected[0])
        self.data['tests']['test_479']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.9756, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.3413, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_480(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_480']['n_samples'] += 1
        self.data['tests']['test_480']['samples'].append(x_test)
        self.data['tests']['test_480']['y_expected'].append(y_expected[0])
        self.data['tests']['test_480']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5117, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3416, max_value=0.4388, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_481(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_481']['n_samples'] += 1
        self.data['tests']['test_481']['samples'].append(x_test)
        self.data['tests']['test_481']['y_expected'].append(y_expected[0])
        self.data['tests']['test_481']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4046, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.5117, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.6818, max_value=0.6273, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4391, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_482(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_482']['n_samples'] += 1
        self.data['tests']['test_482']['samples'].append(x_test)
        self.data['tests']['test_482']['y_expected'].append(y_expected[0])
        self.data['tests']['test_482']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=2.6793, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=0.5969, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_483(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_483']['n_samples'] += 1
        self.data['tests']['test_483']['samples'].append(x_test)
        self.data['tests']['test_483']['y_expected'].append(y_expected[0])
        self.data['tests']['test_483']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6796, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.0951, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_484(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_484']['n_samples'] += 1
        self.data['tests']['test_484']['samples'].append(x_test)
        self.data['tests']['test_484']['y_expected'].append(y_expected[0])
        self.data['tests']['test_484']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6796, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0948, max_value=0.5969, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_485(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_485']['n_samples'] += 1
        self.data['tests']['test_485']['samples'].append(x_test)
        self.data['tests']['test_485']['y_expected'].append(y_expected[0])
        self.data['tests']['test_485']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.3424, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5972, max_value=1.0953, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_486(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_486']['n_samples'] += 1
        self.data['tests']['test_486']['samples'].append(x_test)
        self.data['tests']['test_486']['y_expected'].append(y_expected[0])
        self.data['tests']['test_486']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.3424, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0956, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_487(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_487']['n_samples'] += 1
        self.data['tests']['test_487']['samples'].append(x_test)
        self.data['tests']['test_487']['y_expected'].append(y_expected[0])
        self.data['tests']['test_487']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3427, max_value=0.3913, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5972, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_488(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_488']['n_samples'] += 1
        self.data['tests']['test_488']['samples'].append(x_test)
        self.data['tests']['test_488']['y_expected'].append(y_expected[0])
        self.data['tests']['test_488']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.3916, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=1.0544, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5972, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_489(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_489']['n_samples'] += 1
        self.data['tests']['test_489']['samples'].append(x_test)
        self.data['tests']['test_489']['y_expected'].append(y_expected[0])
        self.data['tests']['test_489']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0547, max_value=1.8193, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.667, -0.626, -0.52, -0.436, -0.366, -0.269, -0.147, 0.537, 1.266, 1.835]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_490(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_490']['n_samples'] += 1
        self.data['tests']['test_490']['samples'].append(x_test)
        self.data['tests']['test_490']['y_expected'].append(y_expected[0])
        self.data['tests']['test_490']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.2756, max_value=0.4863, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.8196, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.017, -1.0, -0.129, 0.304, 0.519, 0.725, 0.73, 1.464, 1.578, 1.814]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_491(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_491']['n_samples'] += 1
        self.data['tests']['test_491']['samples'].append(x_test)
        self.data['tests']['test_491']['y_expected'].append(y_expected[0])
        self.data['tests']['test_491']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8598, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2137, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_492(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_492']['n_samples'] += 1
        self.data['tests']['test_492']['samples'].append(x_test)
        self.data['tests']['test_492']['y_expected'].append(y_expected[0])
        self.data['tests']['test_492']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8601, max_value=0.9963, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2137, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_493(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_493']['n_samples'] += 1
        self.data['tests']['test_493']['samples'].append(x_test)
        self.data['tests']['test_493']['y_expected'].append(y_expected[0])
        self.data['tests']['test_493']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9966, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.2137, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_494(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_494']['n_samples'] += 1
        self.data['tests']['test_494']['samples'].append(x_test)
        self.data['tests']['test_494']['y_expected'].append(y_expected[0])
        self.data['tests']['test_494']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=0.7094, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8759, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_495(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_495']['n_samples'] += 1
        self.data['tests']['test_495']['samples'].append(x_test)
        self.data['tests']['test_495']['y_expected'].append(y_expected[0])
        self.data['tests']['test_495']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.7097, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8759, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_496(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_496']['n_samples'] += 1
        self.data['tests']['test_496']['samples'].append(x_test)
        self.data['tests']['test_496']['y_expected'].append(y_expected[0])
        self.data['tests']['test_496']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8762, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=-0.1002, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_497(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_497']['n_samples'] += 1
        self.data['tests']['test_497']['samples'].append(x_test)
        self.data['tests']['test_497']['y_expected'].append(y_expected[0])
        self.data['tests']['test_497']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8762, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.0999, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_498(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_498']['n_samples'] += 1
        self.data['tests']['test_498']['samples'].append(x_test)
        self.data['tests']['test_498']['y_expected'].append(y_expected[0])
        self.data['tests']['test_498']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6922, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8762, max_value=1.3383, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.5833, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_499(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_499']['n_samples'] += 1
        self.data['tests']['test_499']['samples'].append(x_test)
        self.data['tests']['test_499']['y_expected'].append(y_expected[0])
        self.data['tests']['test_499']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6922, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3386, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=0.5833, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_500(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_500']['n_samples'] += 1
        self.data['tests']['test_500']['samples'].append(x_test)
        self.data['tests']['test_500']['y_expected'].append(y_expected[0])
        self.data['tests']['test_500']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.6922, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8762, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5836, max_value=0.6919, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.2134, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_501(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_501']['n_samples'] += 1
        self.data['tests']['test_501']['samples'].append(x_test)
        self.data['tests']['test_501']['y_expected'].append(y_expected[0])
        self.data['tests']['test_501']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=2.6643, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=1.0238, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.814, -0.534, -0.247, -0.141, 0.634, 0.671, 1.251, 1.274, 1.429, 1.975]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_502(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_502']['n_samples'] += 1
        self.data['tests']['test_502']['samples'].append(x_test)
        self.data['tests']['test_502']['y_expected'].append(y_expected[0])
        self.data['tests']['test_502']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6646, max_value=2.7898, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=1.0238, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.709, -0.652, -0.202, -0.126, -0.089, 0.168, 0.278, 0.402, 0.513, 1.611]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_503(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_503']['n_samples'] += 1
        self.data['tests']['test_503']['samples'].append(x_test)
        self.data['tests']['test_503']['y_expected'].append(y_expected[0])
        self.data['tests']['test_503']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.7901, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=1.0238, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.009, -0.578, -0.201, -0.2, -0.171, 0.72, 0.776, 0.979, 1.886, 1.941]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_504(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_504']['n_samples'] += 1
        self.data['tests']['test_504']['samples'].append(x_test)
        self.data['tests']['test_504']['y_expected'].append(y_expected[0])
        self.data['tests']['test_504']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0241, max_value=1.6538, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4397, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_505(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_505']['n_samples'] += 1
        self.data['tests']['test_505']['samples'].append(x_test)
        self.data['tests']['test_505']['y_expected'].append(y_expected[0])
        self.data['tests']['test_505']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6541, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4397, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_506(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_506']['n_samples'] += 1
        self.data['tests']['test_506']['samples'].append(x_test)
        self.data['tests']['test_506']['y_expected'].append(y_expected[0])
        self.data['tests']['test_506']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6276, max_value=0.8509, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.0241, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4394, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_507(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_507']['n_samples'] += 1
        self.data['tests']['test_507']['samples'].append(x_test)
        self.data['tests']['test_507']['y_expected'].append(y_expected[0])
        self.data['tests']['test_507']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.7368, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=1.3124, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=1.1059, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.86, -0.776, -0.73, -0.683, -0.577, -0.421, -0.407, -0.349, 0.465, 1.254]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_508(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_508']['n_samples'] += 1
        self.data['tests']['test_508']['samples'].append(x_test)
        self.data['tests']['test_508']['y_expected'].append(y_expected[0])
        self.data['tests']['test_508']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.7371, max_value=1.9999, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=1.3124, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=1.1059, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.912, -0.221, -0.089, -0.05, 0.062, 0.256, 0.563, 1.058, 1.176, 1.611]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_509(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_509']['n_samples'] += 1
        self.data['tests']['test_509']['samples'].append(x_test)
        self.data['tests']['test_509']['y_expected'].append(y_expected[0])
        self.data['tests']['test_509']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.9999, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=1.3124, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.1062, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.384, 0.619, 0.639, 0.739, 0.837, 1.074, 1.273, 1.451, 1.734, 2.337]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_510(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_510']['n_samples'] += 1
        self.data['tests']['test_510']['samples'].append(x_test)
        self.data['tests']['test_510']['y_expected'].append(y_expected[0])
        self.data['tests']['test_510']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.9999, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.3126, max_value=2.1684, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.828, -0.552, -0.258, 0.375, 0.756, 0.919, 1.283, 1.442, 1.486, 1.999]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_511(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_511']['n_samples'] += 1
        self.data['tests']['test_511']['samples'].append(x_test)
        self.data['tests']['test_511']['y_expected'].append(y_expected[0])
        self.data['tests']['test_511']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.4487, max_value=1.9999, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1687, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.508, -0.374, -0.252, -0.038, 0.053, 0.141, 0.145, 0.261, 0.529, 0.621]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_512(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_512']['n_samples'] += 1
        self.data['tests']['test_512']['samples'].append(x_test)
        self.data['tests']['test_512']['y_expected'].append(y_expected[0])
        self.data['tests']['test_512']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0001, max_value=2.1963, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4342, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_513(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_513']['n_samples'] += 1
        self.data['tests']['test_513']['samples'].append(x_test)
        self.data['tests']['test_513']['y_expected'].append(y_expected[0])
        self.data['tests']['test_513']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0001, max_value=2.1963, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=0.7379, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4339, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_514(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_514']['n_samples'] += 1
        self.data['tests']['test_514']['samples'].append(x_test)
        self.data['tests']['test_514']['y_expected'].append(y_expected[0])
        self.data['tests']['test_514']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0001, max_value=2.1963, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7382, max_value=0.7918, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4339, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_515(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_515']['n_samples'] += 1
        self.data['tests']['test_515']['samples'].append(x_test)
        self.data['tests']['test_515']['y_expected'].append(y_expected[0])
        self.data['tests']['test_515']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0001, max_value=2.1963, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7921, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4339, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_516(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_516']['n_samples'] += 1
        self.data['tests']['test_516']['samples'].append(x_test)
        self.data['tests']['test_516']['y_expected'].append(y_expected[0])
        self.data['tests']['test_516']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.4866, max_value=1.4768, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.1966, max_value=4.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8512, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6922, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.878, -0.636, -0.374, -0.276, -0.239, -0.166, -0.134, 0.76, 1.248, 1.698]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_517(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_517']['n_samples'] += 1
        self.data['tests']['test_517']['samples'].append(x_test)
        self.data['tests']['test_517']['y_expected'].append(y_expected[0])
        self.data['tests']['test_517']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4771, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.269, 0.466, 0.506, 0.973, 1.114, 1.212, 1.494, 1.735, 1.941, 2.707]),
           st.floats(min_value=-1.823, max_value=1.1233, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.761, -0.448, -0.438, -0.344, -0.339, -0.319, 0.056, 0.1, 0.227, 0.556]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_518(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_518']['n_samples'] += 1
        self.data['tests']['test_518']['samples'].append(x_test)
        self.data['tests']['test_518']['y_expected'].append(y_expected[0])
        self.data['tests']['test_518']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4771, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.542, 0.675, 0.852, 0.926, 0.961, 1.277, 1.279, 1.36, 1.646, 1.678]),
           st.floats(min_value=1.1236, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.284, max_value=-0.4161, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_519(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_519']['n_samples'] += 1
        self.data['tests']['test_519']['samples'].append(x_test)
        self.data['tests']['test_519']['y_expected'].append(y_expected[0])
        self.data['tests']['test_519']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4771, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.788, 0.307, 0.927, 1.019, 1.034, 2.371, 2.589, 2.959, 2.989, 3.042]),
           st.floats(min_value=1.1236, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.4158, max_value=0.5973, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_520(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_520']['n_samples'] += 1
        self.data['tests']['test_520']['samples'].append(x_test)
        self.data['tests']['test_520']['y_expected'].append(y_expected[0])
        self.data['tests']['test_520']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4771, max_value=4.107, exclude_min=True, allow_nan=False),
           st.sampled_from([0.184, 0.473, 0.58, 0.719, 0.742, 0.822, 0.901, 1.005, 2.168, 2.271]),
           st.floats(min_value=1.1236, max_value=3.199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5766, max_value=2.826, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5976, max_value=2.719, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_521(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_521']['n_samples'] += 1
        self.data['tests']['test_521']['samples'].append(x_test)
        self.data['tests']['test_521']['y_expected'].append(y_expected[0])
        self.data['tests']['test_521']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
