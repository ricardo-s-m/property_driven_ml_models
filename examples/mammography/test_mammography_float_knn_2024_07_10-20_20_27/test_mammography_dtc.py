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
    request.cls.data['n_test'] = 161
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_mammography_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestMammographyProperty:

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.46135, allow_nan=False),
           st.sampled_from([-0.32114211, -0.050652758, 0.94114155, 1.1665493, 1.4821203, 2.834567, 4.2320954, 4.6378294, 5.088645, 9.4164747]),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.floats(min_value=-0.461347, max_value=-0.357395, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.50146835, 0.80589687, 1.0763862, 1.301794, 1.4821203, 1.6173649, 1.9780174, 2.7444039, 3.6460351, 23.977818]),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.011683, allow_nan=False),
           st.floats(min_value=-0.357392, max_value=-0.352971, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=0.011686, max_value=1.265371, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.357392, max_value=-0.352971, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.floats(min_value=-0.357392, max_value=-0.352971, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.321141, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.523051, allow_nan=False),
           st.floats(min_value=-0.352968, max_value=-0.346336, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.523051, allow_nan=False),
           st.floats(min_value=-0.346333, max_value=-0.339701, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.523054, max_value=1.265371, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.352968, max_value=-0.339701, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.floats(min_value=-0.339698, max_value=-0.268923, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.193855, allow_nan=False),
           st.floats(min_value=-0.26892, max_value=-0.264499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.193858, max_value=1.265371, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.26892, max_value=-0.264499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.116532, allow_nan=False),
           st.floats(min_value=-0.352968, max_value=-0.264499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.116535, max_value=0.117294, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.352968, max_value=-0.264499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.117297, max_value=1.265371, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.352968, max_value=-0.264499, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.floats(min_value=-0.264496, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=0.332539, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.99385, allow_nan=False),
           st.floats(min_value=-0.264496, max_value=2.002597, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.332542, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.99385, allow_nan=False),
           st.floats(min_value=2.0026, max_value=2.102128, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.332542, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.99385, allow_nan=False),
           st.floats(min_value=2.102131, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.332542, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.993853, max_value=1.048477, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.264496, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.332542, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.04848, max_value=1.265371, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.264496, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.332542, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=-0.673899, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=1.265371, allow_nan=False),
           st.sampled_from([-0.29325125, -0.076494735, 1.4983077, 2.3299449, 2.7988467, 2.8342355, 3.0996517, 3.6349075, 3.6614492, 4.7983149]),
           st.sampled_from([-0.59163147, 0.17475504, 0.26491816, 0.89605999, 2.834567, 3.3755458, 3.6460351, 5.088645, 10.318106, 23.977818]),
           st.floats(min_value=-0.673896, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.265374, max_value=1.266049, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.4038413, -0.38614689, -0.23132081, -0.2092028, -0.18266119, -0.076494735, 0.069484139, 0.073907741, 0.28624065, 0.35701828]),
           st.sampled_from([-0.50146835, -0.36622367, -0.27606055, -0.18589744, -0.0055711987, 0.12967348, 0.2198366, 0.26491816, 0.44524439, 0.98622311]),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.266052, max_value=1.267743, exclude_min=True, allow_nan=False),
           st.sampled_from([0.15795618, 0.50299716, 0.50742076, 0.81264932, 1.8035362, 2.1087648, 2.6661387, 3.0554157, 3.1438877, 3.8870529]),
           st.sampled_from([0.12967348, 0.58048907, 0.76081531, 0.98622311, 2.4739146, 3.3755458, 5.1337266, 9.2361485, 13.113163, 14.645936]),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.267746, max_value=1.268336, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.41711211, -0.4038413, -0.37287609, -0.33306367, -0.30209845, -0.28440404, -0.049953122, 0.016400912, 0.05178973, 0.082754945]),
           st.sampled_from([-0.41130523, -0.36622367, -0.32114211, -0.27606055, 0.039510361, 0.26491816, 0.30999972, 0.76081531, 0.98622311, 1.9780174]),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.268339, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.26670963, -0.14284877, 0.31720586, 0.52953877, 1.4319536, 1.6885226, 2.6617151, 2.7192219, 4.1568926, 4.4665448]),
           st.floats(min_value=-0.59163147, max_value=0.06205, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.268339, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.291041, allow_nan=False),
           st.floats(min_value=0.062053, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.268339, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.291038, max_value=0.071694, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.062053, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.268339, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.071697, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.062053, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.008994, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.sampled_from([-0.4082649, -0.34191087, -0.33306367, -0.29767485, -0.22247361, -0.16054318, 0.21988661, 0.35701828, 0.41894872, 4.2055522]),
           st.floats(min_value=-0.59163147, max_value=-0.546551, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.299473, allow_nan=False),
           st.sampled_from([-0.44365372, -0.4038413, -0.35518168, -0.27555684, -0.2003556, -0.16939038, -0.010140702, 0.15795618, 1.1311487, 1.5823561]),
           st.floats(min_value=-0.546548, max_value=-0.388766, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-0.29947, max_value=0.358834, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.308735, allow_nan=False),
           st.floats(min_value=-0.546548, max_value=-0.388766, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.358837, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.308735, allow_nan=False),
           st.floats(min_value=-0.546548, max_value=-0.388766, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-0.29947, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.308732, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.546548, max_value=-0.388766, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=1.887853, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=0.962401, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.246668, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=1.887853, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.962404, max_value=0.972317, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.246668, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=1.887853, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.97232, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.246668, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=1.887853, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.246671, max_value=1.249715, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=1.887853, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.249718, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=1.887856, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.19182, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.445867, allow_nan=False),
           st.floats(min_value=1.887856, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.191823, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=0.850977, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.102487, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.030323, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.85098, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.102487, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.025753, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.85098, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=0.069356, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.025756, max_value=1.030323, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.85098, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.069359, max_value=1.102487, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.025756, max_value=1.030323, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.102487, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.030326, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.729952, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.10249, max_value=1.103331, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.418795, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.103334, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.418798, max_value=0.421844, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.103334, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.421847, max_value=0.471473, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.103334, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.471476, max_value=0.472659, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.103334, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.472662, max_value=0.729952, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.103334, max_value=2.500178, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.729955, max_value=0.738082, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.738085, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.445864, max_value=-0.286617, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.738085, max_value=1.69933, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.286614, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.500178, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.008997, max_value=1.325893, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.sampled_from([-0.45250093, -0.42595931, -0.3994177, -0.27113323, -0.22247361, -0.16939038, -0.12957796, -0.10745995, -0.023411508, 0.35701828]),
           st.sampled_from([-0.50146835, -0.45638679, -0.36622367, -0.230979, -0.18589744, -0.14081588, -0.050652758, 0.26491816, 0.44524439, 0.53540751]),
           st.floats(min_value=2.500181, max_value=2.52465, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.711352, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.sampled_from([-0.010140702, 0.45876114, 0.63570523, 0.64455243, 0.92766297, 0.971899, 1.1753847, 2.1706952, 4.1259274, 4.7894677]),
           st.sampled_from([-0.18589744, -0.0055711987, 0.44524439, 0.71573375, 3.8714429, 4.8632372, 5.4042159, 6.1706024, 20.686865, 29.477769]),
           st.floats(min_value=2.524653, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.711352, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.69933, allow_nan=False),
           st.sampled_from([-0.44807732, -0.39057049, -0.37729969, -0.195932, -0.010140702, 0.1889214, 0.28181705, 0.35701828, 0.89227416, 1.2063499]),
           st.sampled_from([-0.45638679, -0.18589744, -0.095734317, -0.050652758, -0.0055711987, 0.039510361, 0.17475504, 0.2198366, 0.26491816, 0.49032595]),
           st.floats(min_value=2.500181, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.711355, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.699333, max_value=1.701702, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.41711211, -0.33306367, -0.14727237, 0.15795618, 0.25527543, 0.60031641, 0.91439217, 1.0426766, 1.2063499, 1.9185499]),
           st.sampled_from([-0.36622367, -0.32114211, -0.27606055, -0.050652758, 0.12967348, 0.17475504, 0.30999972, 0.49032595, 0.76081531, 0.89605999]),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.701705, max_value=1.707461, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.2047792, 0.01197731, 0.05178973, 0.79495491, 0.88785055, 1.0294058, 1.4938841, 2.8475063, 2.8873188, 4.49751]),
           st.sampled_from([0.76081531, 1.4370387, 2.5189961, 2.834567, 2.8796486, 3.961606, 4.5025847, 5.4492975, 7.0271521, 18.207379]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.707464, max_value=1.712373, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.4082649, -0.32864006, -0.31094565, -0.27998044, -0.2136264, -0.2047792, -0.13842517, -0.058800326, -0.010140702, 0.25527543]),
           st.sampled_from([-0.45638679, -0.41130523, -0.27606055, -0.18589744, -0.095734317, 0.12967348, 0.30999972, 0.49032595, 0.71573375, 1.9780174]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.712376, max_value=3.74302, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.322006, allow_nan=False),
           st.sampled_from([-0.27606055, 1.2567125, 2.2034252, 2.2935883, 2.3837514, 2.5640777, 3.2853826, 3.8263613, 8.0189464, 23.977818]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.743023, max_value=3.750981, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.322006, allow_nan=False),
           st.sampled_from([-0.59163147, -0.50146835, -0.41130523, -0.32114211, -0.230979, -0.14081588, -0.095734317, 0.039510361, 0.08459192, 0.30999972]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.750984, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.322006, allow_nan=False),
           st.sampled_from([-0.27606055, 2.3837514, 3.555872, 3.6460351, 4.0066876, 4.0968507, 6.1706024, 7.0271521, 10.318106, 10.543514]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.712376, max_value=2.320373, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.322003, max_value=1.082488, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.320376, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.322003, max_value=0.821495, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.673378, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.320376, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.821498, max_value=1.082488, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.673378, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.320376, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.322003, max_value=1.082488, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.321144, allow_nan=False),
           st.floats(min_value=1.673381, max_value=2.072537, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.712376, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.322003, max_value=1.082488, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.321141, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.712376, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082491, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.41130523, 0.039510361, 0.08459192, 0.17475504, 1.9329359, 3.1050564, 4.1419323, 5.0435634, 6.1706024, 20.686865]),
           st.floats(min_value=-0.85955255, max_value=2.072537, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.701705, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.43923012, -0.41268851, -0.32864006, -0.30652205, -0.14284877, -0.041105917, 0.073907741, 0.52511517, 1.0205586, 1.2063499]),
           st.sampled_from([-0.54654991, -0.36622367, -0.32114211, 0.039510361, 0.17475504, 0.2198366, 0.35508128, 0.49032595, 0.71573375, 0.76081531]),
           st.floats(min_value=2.07254, max_value=2.205238, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.701705, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.089765542, 0.62243442, 0.99401701, 1.1001835, 1.2151971, 1.4319536, 1.8123834, 2.3830281, 3.0731101, 4.953141]),
           st.sampled_from([-0.36622367, 2.1132621, 2.2485068, 2.5640777, 2.7444039, 2.7894855, 4.5025847, 8.4246804, 11.760716, 14.645936]),
           st.floats(min_value=2.205241, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.643937, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.701705, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.43038291, -0.37287609, -0.25343882, -0.16939038, -0.10303635, -0.072071133, 0.61801082, 0.72860087, 0.7816841, 1.1311487]),
           st.sampled_from([-0.54654991, -0.50146835, -0.45638679, -0.32114211, -0.230979, 0.12967348, 0.35508128, 0.40016284, 0.44524439, 0.71573375]),
           st.floats(min_value=2.205241, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.64394, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.325893, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.10569738, 0.004570916, 0.49493453, 0.53558643, 0.60215393, 0.96412527, 1.1285961, 1.1465507, 1.3919866, 1.7014492]),
           st.sampled_from([-0.47019533, -0.45250093, -0.31536926, -0.27555684, -0.26228603, -0.23574442, -0.16054318, -0.010140702, 0.28624065, 0.35701828]),
           st.floats(min_value=-0.59163147, max_value=-0.298603, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.32894, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.45767513, -0.43887362, -0.35926364, -0.2881228, -0.18835625, 0.047932949, 0.066226306, 0.27422856, 0.31860689, 0.34553877]),
           st.sampled_from([-0.42153571, -0.067647531, 0.3879835, 0.66667044, 1.7902654, 2.4449585, 3.0819573, 3.7499212, 3.9622541, 4.0463026]),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=0.874215, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.328943, max_value=1.499578, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.10688306, -0.028458757, 0.26914707, 0.40126576, 0.40651663, 0.44310335, 1.7014492, 1.7614108, 2.2592272, 6.1960255]),
           st.sampled_from([-0.35960528, -0.35518168, -0.32864006, -0.32421646, -0.29325125, -0.28440404, -0.041105917, 0.11814376, 0.46318474, 0.61801082]),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=0.874218, max_value=0.879278, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.328943, max_value=1.499578, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.42041088, -0.39161578, -0.29557565, -0.18022587, -0.029983204, 0.14922395, 0.77272255, 0.937024, 1.2839203, 1.3553999]),
           st.sampled_from([-0.094189144, 0.50299716, 0.69763566, 0.96305179, 2.0114455, 2.7369163, 3.3916094, 4.1347746, 4.9487174, 5.0460367]),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=0.879281, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.328943, max_value=1.499578, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.51492656, -0.35773919, -0.20783529, 0.050812459, 0.2191791, 0.43700556, 0.91009211, 0.96717417, 1.3884295, 5.1861644]),
           st.sampled_from([-0.023411508, 0.67994125, 0.83034372, 1.4009884, 1.6796754, 2.3034033, 3.2146653, 3.5597063, 3.8649349, 4.7673497]),
           st.floats(min_value=-0.388763, max_value=-0.298603, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.328943, max_value=1.499578, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.625189, allow_nan=False),
           st.sampled_from([-0.44365372, -0.31094565, -0.18708479, -0.18266119, -0.13842517, -0.12957796, 0.22431022, 0.35701828, 0.52511517, 1.5823561]),
           st.floats(min_value=-0.59163147, max_value=-0.298603, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.499581, max_value=1.533096, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.625192, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.4038413, 0.042942525, 0.069484139, 0.073907741, 0.36586549, 0.51184436, 1.060371, 2.4007225, 2.5024654, 4.8735162]),
           st.floats(min_value=-0.59163147, max_value=-0.298603, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.499581, max_value=1.533096, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.48511516, -0.17243426, 0.26711447, 0.2698246, 0.29878908, 0.34791014, 0.45919473, 0.62468186, 0.7891527, 2.4557114]),
           st.sampled_from([0.68436485, 0.7861077, 0.93208658, 1.2063499, 1.4452244, 1.9804803, 2.8607772, 3.9622541, 4.130351, 4.5992528]),
           st.floats(min_value=-0.59163147, max_value=-0.298603, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.894779, allow_nan=False),
           st.floats(min_value=1.533099, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.58708369, -0.31353025, -0.0072858899, 0.026421315, 0.044714673, 0.29387698, 0.3343595, 0.62959397, 0.72326273, 1.5010692]),
           st.sampled_from([-0.067647531, 0.14910898, 0.68878845, 1.1355723, 1.2196207, 1.3567524, 1.3788704, 1.5779325, 3.2190889, 4.8204329]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.265779, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.658359, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.028458757, 0.11670242, 0.1605726, 0.23882752, 0.26914707, 0.47867377, 0.91415731, 1.2596985, 1.8916663, 2.5654716]),
           st.sampled_from([-0.38172329, -0.35960528, -0.31536926, -0.16496678, -0.16054318, -0.10303635, -0.041105917, 0.89227416, 1.0869127, 1.1665375]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.265782, max_value=1.268311, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.658359, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.47817046, 0.095698938, 0.16700916, 0.2073223, 0.23069714, 0.4058391, 0.43243222, 0.59165219, 1.7920691, 2.6809908]),
           st.sampled_from([-0.21805001, 0.3879835, 0.66224684, 0.7861077, 2.2901325, 2.4272641, 3.5597063, 3.8030044, 4.4665448, 5.0725783]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.268314, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.658359, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.25001164, -0.03726667, 0.15176469, 0.37365634, 0.40126576, 0.46004164, 0.89027431, 1.4033352, 2.3191888, 2.8701915]),
           st.sampled_from([-0.43038291, -0.42153571, -0.32864006, -0.2136264, -0.10745995, 0.069484139, 0.082754945, 0.41894872, 1.7637238, 4.2055522]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=3.702508, allow_nan=False),
           st.floats(min_value=0.658362, max_value=0.894779, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.335034, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.50815124, -0.5056105, -0.2325652, -0.15193892, 0.10078043, 0.31504985, 0.32267208, 0.38856204, 0.85741402, 1.3279598]),
           st.sampled_from([-0.041105917, 0.32162947, 0.44106673, 1.8079598, 2.1928132, 2.3255213, 2.3476393, 3.3562206, 3.7322268, 4.6744541]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.85955255, max_value=1.866206, allow_nan=False),
           st.floats(min_value=0.658362, max_value=0.894779, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.335037, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.23087137, -0.012536761, 0.11145155, 0.13787529, 0.40922676, 0.56065511, 0.72681977, 0.73088496, 0.81354384, 1.4033352]),
           st.sampled_from([-0.43923012, -0.36845248, -0.35075807, -0.32421646, -0.2003556, -0.19150839, -0.098612747, 0.15795618, 0.28181705, 0.91439217]),
           st.floats(min_value=-0.2986, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.866209, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.658362, max_value=0.894779, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.335037, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.078259, allow_nan=False),
           st.sampled_from([-0.14727237, 0.50742076, 0.5870456, 1.1046071, 1.1532667, 1.750453, 1.8433486, 2.7590343, 4.9133286, 5.0504603]),
           st.sampled_from([-0.50146835, -0.45638679, -0.27606055, 0.76081531, 1.3468756, 1.7976912, 2.2485068, 3.5107904, 5.9902762, 9.4164747]),
           st.floats(min_value=-0.85955255, max_value=0.63223, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.078259, allow_nan=False),
           st.sampled_from([-0.44807732, -0.41268851, -0.39057049, -0.37729969, -0.36845248, -0.35960528, -0.2003556, -0.12957796, 0.25527543, 1.4584952]),
           st.sampled_from([-0.59163147, -0.54654991, -0.27606055, -0.230979, -0.18589744, 0.039510361, 0.08459192, 0.17475504, 0.35508128, 0.98622311]),
           st.floats(min_value=0.632233, max_value=1.000798, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.078256, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.sampled_from([0.087178548, 0.12256737, 0.1844978, 0.21546301, 0.55165678, 0.68436485, 0.95862819, 2.5467014, 2.8253883, 4.7363845]),
           st.sampled_from([-0.36622367, -0.0055711987, 1.7976912, 2.1583436, 2.3837514, 3.3755458, 4.9083188, 5.4492975, 6.2607655, 20.686865]),
           st.floats(min_value=-0.85955255, max_value=1.000798, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.03726667, 0.047932949, 0.085705344, 0.12347774, 0.33063308, 0.5449025, 0.91415731, 3.7918041, 6.1960255, 6.2441303]),
           st.floats(min_value=-0.47019533, max_value=-0.339701, allow_nan=False),
           st.sampled_from([-0.59163147, -0.54654991, -0.45638679, -0.36622367, 0.17475504, 0.26491816, 0.35508128, 0.44524439, 0.49032595, 0.89605999]),
           st.floats(min_value=1.000801, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.349094, allow_nan=False),
           st.floats(min_value=-0.339698, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, 0.98622311, 1.2567125, 1.4370387, 2.4739146, 3.6460351, 6.7566627, 7.7935386, 11.760716, 13.113163]),
           st.floats(min_value=1.000801, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.349097, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.339698, max_value=-0.147274, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.59163147, -0.50146835, -0.45638679, -0.36622367, 0.039510361, 0.17475504, 0.2198366, 0.35508128, 0.53540751, 0.98622311]),
           st.floats(min_value=1.000801, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.349097, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.147271, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.095734317, 0.62557063, 3.150138, 4.2320954, 4.6378294, 4.8181556, 5.088645, 7.9738648, 10.678758, 11.760716]),
           st.floats(min_value=1.000801, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.894782, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.325896, max_value=1.578802, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.45293241, -0.32996039, -0.26796624, -0.22409605, -0.05047854, 0.047086034, 0.065040626, 0.27219596, 0.29235253, 0.96344774]),
           st.sampled_from([-0.4082649, -0.22689721, -0.195932, -0.15611958, -0.010140702, 1.361176, 2.2193548, 2.6705623, 3.1571585, 3.7808864]),
           st.sampled_from([1.4370387, 2.2485068, 3.0148933, 3.1952195, 3.5107904, 3.6009535, 3.8263613, 4.6378294, 5.7197868, 9.2361485]),
           st.floats(min_value=-0.85955255, max_value=1.217045, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.802765, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.25577066, -0.05742324, -0.028797523, 0.07740558, 0.099086597, 0.22155046, 0.24289271, 0.46393745, 0.51695431, 0.54710447]),
           st.sampled_from([0.087178548, 0.30835866, 1.5381201, 1.6885226, 1.9185499, 2.2901325, 2.617479, 3.5420119, 3.696838, 4.9000578]),
           st.sampled_from([0.30999972, 0.89605999, 0.94114155, 1.0313047, 1.6624465, 3.3755458, 3.8714429, 4.0066876, 4.8632372, 10.543514]),
           st.floats(min_value=-0.85955255, max_value=0.965144, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.802768, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.48697838, -0.17768513, -0.15769794, 0.016258339, 0.022356125, 0.56201017, 0.78288553, 0.93092622, 1.846441, 3.4164515]),
           st.floats(min_value=-0.47019533, max_value=-0.390572, allow_nan=False),
           st.sampled_from([-0.45638679, 0.08459192, 0.98622311, 2.8796486, 3.3304642, 3.555872, 3.6911167, 4.0968507, 4.2320954, 5.4042159]),
           st.floats(min_value=0.965147, max_value=1.217045, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.802768, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.032857867, 0.23764184, 0.39872502, 0.42294678, 0.87265849, 1.0974296, 1.2049878, 1.2659657, 1.297979, 2.8701915]),
           st.floats(min_value=-0.390569, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.45638679, -0.36622367, -0.32114211, -0.230979, -0.14081588, 0.12967348, 0.30999972, 0.40016284, 0.49032595, 0.76081531]),
           st.floats(min_value=0.965147, max_value=1.217045, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.548133, allow_nan=False),
           st.floats(min_value=1.802768, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.43328398, -0.26627241, -0.18378291, -0.18124217, -0.12399074, 0.097731533, 0.11128217, 0.13397948, 0.24441716, 0.35451607]),
           st.sampled_from([-0.4038413, 0.060636934, 0.10929656, 1.3744468, 1.42753, 2.9271312, 3.006756, 3.4800815, 3.6791436, 5.0416131]),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=1.217048, max_value=1.321265, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.9872, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.46123217, -0.061657814, 0.087229791, 0.10891081, 0.31064589, 0.50746887, 0.7156405, 1.1086089, 4.1664791, 5.0267751]),
           st.sampled_from([-0.43480652, -0.36402888, -0.26228603, -0.25786243, -0.14284877, -0.10303635, -0.023411508, 0.15353258, 0.61801082, 0.72860087]),
           st.floats(min_value=-0.59163147, max_value=-0.388766, allow_nan=False),
           st.floats(min_value=1.321268, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.9872, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.44039807, -0.42193533, -0.32945224, -0.28100872, -0.26254598, 0.041327014, 0.37789092, 0.74375807, 1.0611817, 1.3938498]),
           st.sampled_from([-0.08534194, 0.003130105, 0.29508785, 0.36144189, 0.49414995, 0.82592012, 1.4142592, 2.1353064, 2.7723051, 4.0816914]),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.217048, max_value=2.022959, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.9872, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.15024509, -0.042178776, 0.02150921, 0.085705344, 0.48714291, 0.49713651, 0.61350259, 0.69243504, 0.73359509, 1.2815489]),
           st.sampled_from([-0.33748727, -0.24459162, -0.089765542, 0.073907741, 0.28181705, 0.35701828, 0.41894872, 0.7816841, 0.91439217, 1.7637238]),
           st.floats(min_value=-0.388763, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.022962, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=0.9872, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.32183001, -0.043872605, 0.15413605, 0.17920473, 0.27947943, 0.31064589, 0.61350259, 1.0217155, 1.2361543, 1.2815489]),
           st.sampled_from([-0.16496678, -0.14727237, -0.10745995, -0.10303635, -0.04552952, 0.003130105, 0.15353258, 0.29508785, 0.35701828, 0.60031641]),
           st.sampled_from([-0.54654991, -0.45638679, -0.41130523, -0.32114211, -0.27606055, -0.18589744, 0.08459192, 0.12967348, 0.49032595, 1.9780174]),
           st.floats(min_value=1.217048, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.987203, max_value=1.548133, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.578805, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.412613, allow_nan=False),
           st.sampled_from([-0.072071133, 1.31694, 2.3210977, 2.3962989, 2.4139933, 3.4225746, 3.7101088, 4.4444268, 4.7275373, 4.9929534]),
           st.sampled_from([-0.41130523, 0.2198366, 1.4370387, 2.3837514, 2.8796486, 3.4657089, 3.5107904, 4.8181556, 7.9738648, 20.686865]),
           st.floats(min_value=-0.85955255, max_value=1.154808, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.173537, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.412616, max_value=0.481298, exclude_min=True, allow_nan=False),
           st.sampled_from([0.11372016, 0.22431022, 0.47645554, 0.60474001, 1.0249822, 1.1886555, 1.3479052, 1.3877176, 2.573243, 4.0153373]),
           st.sampled_from([1.6624465, 1.8878543, 2.1583436, 2.9247302, 2.9698117, 3.150138, 3.3755458, 4.1870138, 5.6747053, 9.2361485]),
           st.floats(min_value=-0.85955255, max_value=1.154808, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.037941, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_104(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_104']['n_samples'] += 1
        self.data['tests']['test_104']['samples'].append(x_test)
        self.data['tests']['test_104']['y_expected'].append(y_expected[0])
        self.data['tests']['test_104']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.412616, max_value=0.481298, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.32421646, -0.31536926, -0.25786243, -0.16054318, -0.15611958, -0.12957796, -0.058800326, 0.14910898, 0.15353258, 0.61801082]),
           st.sampled_from([-0.50146835, -0.14081588, 0.039510361, 0.08459192, 0.12967348, 0.17475504, 0.2198366, 0.30999972, 0.76081531, 0.89605999]),
           st.floats(min_value=-0.85955255, max_value=0.983499, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.037944, max_value=1.173537, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.412616, max_value=0.481298, exclude_min=True, allow_nan=False),
           st.sampled_from([0.87015614, 0.92766297, 1.1665375, 1.4009884, 1.9008555, 2.1087648, 2.8253883, 3.7278032, 4.108233, 4.8690926]),
           st.sampled_from([0.35508128, 0.67065219, 0.71573375, 1.6624465, 2.3837514, 2.6993224, 3.961606, 4.0066876, 5.4492975, 6.9369889]),
           st.floats(min_value=0.983502, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.037944, max_value=1.173537, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_106(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_106']['n_samples'] += 1
        self.data['tests']['test_106']['samples'].append(x_test)
        self.data['tests']['test_106']['y_expected'].append(y_expected[0])
        self.data['tests']['test_106']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.481301, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.164968, allow_nan=False),
           st.sampled_from([-0.54654991, -0.50146835, -0.41130523, -0.36622367, -0.32114211, -0.18589744, -0.095734317, 0.12967348, 0.35508128, 0.53540751]),
           st.floats(min_value=-0.85955255, max_value=1.154808, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=0.946528, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.481301, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.164968, allow_nan=False),
           st.sampled_from([-0.54654991, -0.27606055, 0.17475504, 1.0763862, 2.2935883, 2.4739146, 3.2853826, 3.3304642, 3.8714429, 7.1173152]),
           st.floats(min_value=-0.85955255, max_value=1.154808, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.946531, max_value=1.173537, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_108(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_108']['n_samples'] += 1
        self.data['tests']['test_108']['samples'].append(x_test)
        self.data['tests']['test_108']['y_expected'].append(y_expected[0])
        self.data['tests']['test_108']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.481301, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.164965, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.32114211, -0.095734317, 0.039510361, 0.35508128, 2.428833, 3.5107904, 6.7566627, 6.9369889, 10.318106, 23.977818]),
           st.floats(min_value=-0.85955255, max_value=1.154808, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.173537, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_109(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_109']['n_samples'] += 1
        self.data['tests']['test_109']['samples'].append(x_test)
        self.data['tests']['test_109']['y_expected'].append(y_expected[0])
        self.data['tests']['test_109']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.38026712, -0.28151687, -0.27321711, -0.25356869, -0.25289115, -0.24205065, -0.19292959, -0.1194174, 0.51017899, 0.78305491]),
           st.sampled_from([0.082754945, 0.71533007, 1.1444195, 1.706217, 2.1087648, 2.2547436, 2.7147983, 3.8560877, 4.3072951, 4.6346417]),
           st.sampled_from([0.44524439, 0.67065219, 1.3468756, 1.8427727, 1.8878543, 2.7444039, 3.555872, 4.7730741, 9.2361485, 10.273024]),
           st.floats(min_value=-0.85955255, max_value=0.750796, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.145165, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.410478, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=0.219835, allow_nan=False),
           st.floats(min_value=0.750799, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_111(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_111']['n_samples'] += 1
        self.data['tests']['test_111']['samples'].append(x_test)
        self.data['tests']['test_111']['y_expected'].append(y_expected[0])
        self.data['tests']['test_111']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.145165, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.410478, allow_nan=False),
           st.floats(min_value=0.219838, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.750799, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_112(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_112']['n_samples'] += 1
        self.data['tests']['test_112']['samples'].append(x_test)
        self.data['tests']['test_112']['y_expected'].append(y_expected[0])
        self.data['tests']['test_112']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.145162, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.410478, allow_nan=False),
           st.sampled_from([-0.095734317, -0.0055711987, 0.039510361, 0.12967348, 0.26491816, 0.30999972, 0.53540751, 0.71573375, 0.76081531, 0.89605999]),
           st.floats(min_value=0.750799, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_113(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_113']['n_samples'] += 1
        self.data['tests']['test_113']['samples'].append(x_test)
        self.data['tests']['test_113']['y_expected'].append(y_expected[0])
        self.data['tests']['test_113']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.63027, allow_nan=False),
           st.floats(min_value=-0.410475, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([1.5722834, 1.6173649, 2.0681805, 3.6911167, 3.8714429, 3.961606, 4.0066876, 4.9083188, 4.9984819, 13.473815]),
           st.floats(min_value=0.750799, max_value=1.056917, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.533096, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_114(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_114']['n_samples'] += 1
        self.data['tests']['test_114']['samples'].append(x_test)
        self.data['tests']['test_114']['y_expected'].append(y_expected[0])
        self.data['tests']['test_114']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.63027, allow_nan=False),
           st.floats(min_value=-0.410475, max_value=-0.359607, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, -0.41130523, -0.230979, -0.18589744, -0.050652758, 0.039510361, 0.17475504, 0.26491816, 0.89605999, 1.9780174]),
           st.floats(min_value=0.750799, max_value=1.056917, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.533099, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_115(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_115']['n_samples'] += 1
        self.data['tests']['test_115']['samples'].append(x_test)
        self.data['tests']['test_115']['y_expected'].append(y_expected[0])
        self.data['tests']['test_115']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.63027, allow_nan=False),
           st.floats(min_value=-0.359604, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([0.40016284, 2.2935883, 3.2853826, 3.4206273, 3.4657089, 4.5025847, 4.6378294, 5.1337266, 6.0804393, 10.318106]),
           st.floats(min_value=0.750799, max_value=1.056917, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.533099, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_116(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_116']['n_samples'] += 1
        self.data['tests']['test_116']['samples'].append(x_test)
        self.data['tests']['test_116']['y_expected'].append(y_expected[0])
        self.data['tests']['test_116']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.63027, allow_nan=False),
           st.floats(min_value=-0.410475, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.32114211, -0.27606055, -0.18589744, 0.039510361, 0.2198366, 0.44524439, 0.71573375, 0.76081531, 0.89605999, 1.9780174]),
           st.floats(min_value=1.05692, max_value=1.085609, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_117(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_117']['n_samples'] += 1
        self.data['tests']['test_117']['samples'].append(x_test)
        self.data['tests']['test_117']['y_expected'].append(y_expected[0])
        self.data['tests']['test_117']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.63027, allow_nan=False),
           st.floats(min_value=-0.410475, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.27606055, 0.40016284, 1.5272018, 2.023099, 2.0681805, 2.1132621, 2.2034252, 2.3837514, 4.9083188, 7.2525599]),
           st.floats(min_value=1.085612, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_118(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_118']['n_samples'] += 1
        self.data['tests']['test_118']['samples'].append(x_test)
        self.data['tests']['test_118']['y_expected'].append(y_expected[0])
        self.data['tests']['test_118']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.630273, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.410475, max_value=-0.149486, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.50146835, -0.45638679, -0.27606055, -0.18589744, 0.08459192, 0.17475504, 0.26491816, 0.40016284, 0.49032595, 0.71573375]),
           st.floats(min_value=0.750799, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_119(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_119']['n_samples'] += 1
        self.data['tests']['test_119']['samples'].append(x_test)
        self.data['tests']['test_119']['y_expected'].append(y_expected[0])
        self.data['tests']['test_119']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.630273, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.149483, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([0.85097843, 1.0763862, 1.2567125, 1.3468756, 2.6542408, 3.7361982, 4.0968507, 4.1870138, 4.9984819, 6.1706024]),
           st.floats(min_value=0.750799, max_value=1.154808, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.17354, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_120(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_120']['n_samples'] += 1
        self.data['tests']['test_120']['samples'].append(x_test)
        self.data['tests']['test_120']['y_expected'].append(y_expected[0])
        self.data['tests']['test_120']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=-0.390516, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.sampled_from([-0.54654991, -0.14081588, -0.050652758, 0.08459192, 0.17475504, 0.2198366, 0.40016284, 0.49032595, 0.71573375, 0.98622311]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_121(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_121']['n_samples'] += 1
        self.data['tests']['test_121']['samples'].append(x_test)
        self.data['tests']['test_121']['y_expected'].append(y_expected[0])
        self.data['tests']['test_121']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.390513, max_value=0.18005, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.sampled_from([0.35508128, 1.5272018, 1.6173649, 2.6091592, 3.4206273, 3.8714429, 4.1419323, 4.9083188, 6.9369889, 20.686865]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.272568, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_122(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_122']['n_samples'] += 1
        self.data['tests']['test_122']['samples'].append(x_test)
        self.data['tests']['test_122']['y_expected'].append(y_expected[0])
        self.data['tests']['test_122']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.390513, max_value=0.125932, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.sampled_from([0.17475504, 0.62557063, 1.0313047, 1.4370387, 1.7075281, 4.1870138, 5.2689712, 7.7935386, 9.4164747, 14.645936]),
           st.floats(min_value=1.154811, max_value=1.387721, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.272571, max_value=1.4234, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_123(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_123']['n_samples'] += 1
        self.data['tests']['test_123']['samples'].append(x_test)
        self.data['tests']['test_123']['y_expected'].append(y_expected[0])
        self.data['tests']['test_123']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.390513, max_value=0.125932, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.sampled_from([-0.50146835, -0.36622367, -0.32114211, -0.18589744, -0.095734317, 0.039510361, 0.08459192, 0.2198366, 0.35508128, 0.98622311]),
           st.floats(min_value=1.387724, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.272571, max_value=1.4234, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_124(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_124']['n_samples'] += 1
        self.data['tests']['test_124']['samples'].append(x_test)
        self.data['tests']['test_124']['y_expected'].append(y_expected[0])
        self.data['tests']['test_124']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.125935, max_value=0.18005, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.sampled_from([0.85097843, 2.2034252, 2.9698117, 3.8263613, 4.0968507, 4.9083188, 5.6747053, 7.8837017, 13.473815, 20.055723]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.272571, max_value=1.4234, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_125(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_125']['n_samples'] += 1
        self.data['tests']['test_125']['samples'].append(x_test)
        self.data['tests']['test_125']['y_expected'].append(y_expected[0])
        self.data['tests']['test_125']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.180053, max_value=1.007825, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.233534, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.343684, allow_nan=False),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_126(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_126']['n_samples'] += 1
        self.data['tests']['test_126']['samples'].append(x_test)
        self.data['tests']['test_126']['y_expected'].append(y_expected[0])
        self.data['tests']['test_126']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.180053, max_value=1.007825, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.233531, max_value=-0.162756, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.343684, allow_nan=False),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_127(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_127']['n_samples'] += 1
        self.data['tests']['test_127']['samples'].append(x_test)
        self.data['tests']['test_127']['y_expected'].append(y_expected[0])
        self.data['tests']['test_127']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.007828, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.floats(min_value=-0.59163147, max_value=-0.343684, allow_nan=False),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_128(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_128']['n_samples'] += 1
        self.data['tests']['test_128']['samples'].append(x_test)
        self.data['tests']['test_128']['y_expected'].append(y_expected[0])
        self.data['tests']['test_128']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.180053, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.162756, allow_nan=False),
           st.floats(min_value=-0.343681, max_value=29.477769, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_129(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_129']['n_samples'] += 1
        self.data['tests']['test_129']['samples'].append(x_test)
        self.data['tests']['test_129']['y_expected'].append(y_expected[0])
        self.data['tests']['test_129']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=3.750727, allow_nan=False),
           st.floats(min_value=-0.162753, max_value=3.904746, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.45638679, -0.230979, 0.35508128, 0.76081531, 0.94114155, 1.0313047, 1.7075281, 2.4739146, 5.0435634, 11.760716]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_130(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_130']['n_samples'] += 1
        self.data['tests']['test_130']['samples'].append(x_test)
        self.data['tests']['test_130']['y_expected'].append(y_expected[0])
        self.data['tests']['test_130']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=1.645805, allow_nan=False),
           st.floats(min_value=3.904749, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, -0.36622367, -0.32114211, -0.18589744, 0.17475504, 0.26491816, 0.40016284, 0.44524439, 0.53540751, 1.9780174]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_131(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_131']['n_samples'] += 1
        self.data['tests']['test_131']['samples'].append(x_test)
        self.data['tests']['test_131']['y_expected'].append(y_expected[0])
        self.data['tests']['test_131']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.645808, max_value=3.750727, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.904749, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, 0.12967348, 0.80589687, 3.150138, 3.4657089, 3.555872, 4.0968507, 8.4246804, 18.207379, 23.977818]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_132(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_132']['n_samples'] += 1
        self.data['tests']['test_132']['samples'].append(x_test)
        self.data['tests']['test_132']['y_expected'].append(y_expected[0])
        self.data['tests']['test_132']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.75073, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.162753, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, -0.36622367, -0.14081588, -0.050652758, -0.0055711987, 0.08459192, 0.12967348, 0.2198366, 0.40016284, 0.76081531]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.4234, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_133(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_133']['n_samples'] += 1
        self.data['tests']['test_133']['samples'].append(x_test)
        self.data['tests']['test_133']['y_expected'].append(y_expected[0])
        self.data['tests']['test_133']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.035064692, 0.017613402, 0.046577885, 0.37467264, 0.3780603, 0.42294678, 0.45123373, 0.50746887, 2.0158239, 3.7801167]),
           st.sampled_from([-0.34633447, -0.28882764, -0.27113323, -0.21805001, -0.2003556, -0.16054318, 0.25527543, 0.65782324, 1.1665375, 1.5823561]),
           st.sampled_from([-0.59163147, -0.45638679, -0.32114211, -0.095734317, -0.0055711987, 0.039510361, 0.12967348, 0.17475504, 0.2198366, 1.9780174]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.423403, max_value=1.67631, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_134(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_134']['n_samples'] += 1
        self.data['tests']['test_134']['samples'].append(x_test)
        self.data['tests']['test_134']['y_expected'].append(y_expected[0])
        self.data['tests']['test_134']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.44615709, -0.38229972, -0.27643538, -0.0037288481, 0.016935871, 0.23001961, 0.25271692, 0.88112763, 0.95853564, 1.5178381]),
           st.sampled_from([-0.32421646, -0.22689721, -0.12073076, 0.21546301, 1.4408008, 1.9981747, 2.0601051, 3.0377212, 3.050992, 3.9755249]),
           st.sampled_from([-0.050652758, 0.2198366, 2.023099, 2.2034252, 2.2485068, 2.2935883, 4.8181556, 4.9984819, 6.7566627, 18.207379]),
           st.floats(min_value=1.154811, max_value=3.702508, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.548136, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.676313, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_135(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_135']['n_samples'] += 1
        self.data['tests']['test_135']['samples'].append(x_test)
        self.data['tests']['test_135']['y_expected'].append(y_expected[0])
        self.data['tests']['test_135']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.662961, allow_nan=False),
           st.sampled_from([-0.43923012, -0.43038291, -0.36845248, -0.35518168, -0.35075807, -0.32421646, -0.25786243, -0.25343882, -0.24459162, -0.10745995]),
           st.sampled_from([-0.41130523, -0.36622367, -0.32114211, -0.095734317, -0.0055711987, 0.2198366, 0.26491816, 0.49032595, 0.53540751, 0.71573375]),
           st.floats(min_value=3.702511, max_value=3.771496, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.942693, allow_nan=False),
           st.sampled_from([0.529076, 0.90387002, 0.91301134, 1.1811567, 1.190298, 1.2847583, 1.2938996, 1.5833746, 1.6595523, 1.6900233]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_136(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_136']['n_samples'] += 1
        self.data['tests']['test_136']['samples'].append(x_test)
        self.data['tests']['test_136']['y_expected'].append(y_expected[0])
        self.data['tests']['test_136']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.662961, allow_nan=False),
           st.sampled_from([0.15353258, 0.1844978, 0.3968307, 0.47645554, 0.75514249, 0.87900335, 1.4894605, 1.6885226, 3.829546, 4.7408081]),
           st.sampled_from([0.58048907, 1.6173649, 1.7526096, 1.8878543, 2.1132621, 3.3304642, 3.6460351, 4.2320954, 5.3140528, 11.760716]),
           st.floats(min_value=3.771499, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=1.942693, allow_nan=False),
           st.sampled_from([0.85816343, 1.190298, 1.278664, 1.2908525, 1.3670301, 1.3944541, 1.4035954, 1.4645375, 1.6199399, 1.7631539]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_137(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_137']['n_samples'] += 1
        self.data['tests']['test_137']['samples'].append(x_test)
        self.data['tests']['test_137']['y_expected'].append(y_expected[0])
        self.data['tests']['test_137']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.78441482, max_value=0.662961, allow_nan=False),
           st.sampled_from([-0.43480652, -0.37729969, -0.26228603, -0.23574442, -0.2092028, -0.11630716, -0.041105917, 0.14910898, 0.35701828, 0.7905313]),
           st.sampled_from([-0.45638679, -0.27606055, -0.230979, -0.095734317, -0.050652758, -0.0055711987, 0.08459192, 0.40016284, 0.71573375, 0.89605999]),
           st.floats(min_value=3.702511, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.942696, max_value=3.03099, exclude_min=True, allow_nan=False),
           st.sampled_from([0.69057261, 0.88254028, 1.1567798, 1.1689682, 1.2817111, 1.4035954, 1.4279722, 1.4980557, 1.5011028, 1.565092]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_138(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_138']['n_samples'] += 1
        self.data['tests']['test_138']['samples'].append(x_test)
        self.data['tests']['test_138']['y_expected'].append(y_expected[0])
        self.data['tests']['test_138']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.662964, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.47019533, max_value=-0.355183, allow_nan=False),
           st.sampled_from([0.44524439, 0.49032595, 1.0763862, 3.1050564, 4.1870138, 7.0271521, 8.0189464, 14.645936, 20.055723, 20.686865]),
           st.floats(min_value=3.702511, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=3.03099, allow_nan=False),
           st.sampled_from([0.23045962, 0.42242729, 0.67838418, 0.76065604, 0.81245684, 0.89777581, 1.190298, 1.537668, 1.8149547, 1.8606613]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_139(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_139']['n_samples'] += 1
        self.data['tests']['test_139']['samples'].append(x_test)
        self.data['tests']['test_139']['y_expected'].append(y_expected[0])
        self.data['tests']['test_139']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.662964, max_value=10.227084, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.35518, max_value=2.413992, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.45638679, -0.36622367, -0.230979, -0.18589744, 0.08459192, 0.12967348, 0.2198366, 0.44524439, 0.53540751, 1.9780174]),
           st.floats(min_value=3.702511, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=3.03099, allow_nan=False),
           st.sampled_from([0.93434108, 0.93738818, 1.0013774, 1.1567798, 1.1811567, 1.2573343, 1.2969467, 1.3274177, 1.5803275, 1.8850381]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_140(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_140']['n_samples'] += 1
        self.data['tests']['test_140']['samples'].append(x_test)
        self.data['tests']['test_140']['y_expected'].append(y_expected[0])
        self.data['tests']['test_140']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.662964, max_value=10.227084, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.413995, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([0.49032595, 1.9329359, 2.1132621, 2.5189961, 2.6542408, 3.6911167, 4.8632372, 7.7935386, 10.318106, 23.977818]),
           st.floats(min_value=3.702511, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=3.03099, allow_nan=False),
           st.sampled_from([0.40414466, 0.54431153, 0.56868837, 0.72409077, 1.0562253, 1.2908525, 1.3182764, 1.3335119, 1.3396062, 1.4127367]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_141(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_141']['n_samples'] += 1
        self.data['tests']['test_141']['samples'].append(x_test)
        self.data['tests']['test_141']['y_expected'].append(y_expected[0])
        self.data['tests']['test_141']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.227087, max_value=31.508443, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.35518, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.54654991, 0.76081531, 1.7075281, 2.5640777, 2.7444039, 2.9698117, 3.8263613, 5.088645, 7.9738648, 10.318106]),
           st.floats(min_value=3.702511, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.37786573, max_value=3.03099, allow_nan=False),
           st.sampled_from([0.087245646, 0.86425764, 1.01966, 1.0653666, 1.3121822, 1.3304648, 1.3731243, 1.5955631, 1.6199399, 1.6412696]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_142(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_142']['n_samples'] += 1
        self.data['tests']['test_142']['samples'].append(x_test)
        self.data['tests']['test_142']['y_expected'].append(y_expected[0])
        self.data['tests']['test_142']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.5045942, -0.30946506, -0.27508032, 0.0057565965, 0.43971569, 0.69531455, 0.81981101, 0.89942099, 1.6570709, 5.883514]),
           st.sampled_from([-0.018987906, 0.16237979, 0.25085183, 0.47645554, 0.93651018, 2.1706952, 2.2547436, 3.3341026, 3.4756579, 4.3072951]),
           st.sampled_from([-0.54654991, -0.18589744, 0.17475504, 0.89605999, 0.98622311, 1.7976912, 3.2403011, 3.6460351, 3.7361982, 5.1337266]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.030993, max_value=5.013885, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.216197, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_143(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_143']['n_samples'] += 1
        self.data['tests']['test_143']['samples'].append(x_test)
        self.data['tests']['test_143']['y_expected'].append(y_expected[0])
        self.data['tests']['test_143']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.23087137, 0.15176469, 0.15413605, 0.23882752, 0.33063308, 0.36230769, 0.48714291, 0.61350259, 3.389689, 5.0267751]),
           st.floats(min_value=-0.47019533, max_value=-0.220263, allow_nan=False),
           st.sampled_from([-0.32114211, -0.27606055, -0.230979, -0.095734317, 0.08459192, 0.17475504, 0.26491816, 0.35508128, 0.44524439, 0.89605999]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.030993, max_value=3.75436, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2162, max_value=1.447777, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_144(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_144']['n_samples'] += 1
        self.data['tests']['test_144']['samples'].append(x_test)
        self.data['tests']['test_144']['y_expected'].append(y_expected[0])
        self.data['tests']['test_144']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.27474155, -0.16769153, 0.049457395, 0.11839625, 0.12415527, 0.17259879, 0.19072277, 0.89298444, 1.3648853, 1.3940192]),
           st.floats(min_value=-0.22026, max_value=5.0858491, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.59163147, -0.45638679, 2.5640777, 4.8181556, 4.9984819, 5.3140528, 5.4042159, 6.1706024, 7.8837017, 7.9738648]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.030993, max_value=3.75436, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2162, max_value=1.447777, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_145(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_145']['n_samples'] += 1
        self.data['tests']['test_145']['samples'].append(x_test)
        self.data['tests']['test_145']['y_expected'].append(y_expected[0])
        self.data['tests']['test_145']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.44547956, -0.35790858, -0.19902738, -0.18615427, 0.0054178307, 0.49069995, 0.52677852, 0.67702119, 0.69853282, 1.6475854]),
           st.sampled_from([-0.39057049, 0.069484139, 0.25969903, 0.50742076, 1.7371822, 1.7991126, 1.9495151, 2.0158691, 3.373915, 3.5641299]),
           st.sampled_from([-0.0055711987, 0.49032595, 1.9329359, 2.9698117, 3.150138, 3.6009535, 5.2689712, 7.8837017, 10.678758, 14.645936]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.754363, max_value=5.013885, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2162, max_value=1.447777, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_146(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_146']['n_samples'] += 1
        self.data['tests']['test_146']['samples'].append(x_test)
        self.data['tests']['test_146']['y_expected'].append(y_expected[0])
        self.data['tests']['test_146']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.15024509, 0.087229791, 0.19749808, 0.53457014, 0.55930005, 0.7156405, 1.1072539, 3.7472564, 3.7801167, 4.6939376]),
           st.sampled_from([-0.38614689, -0.37287609, -0.34633447, -0.29767485, -0.13842517, -0.12957796, 0.016400912, 0.073907741, 0.21988661, 0.49414995]),
           st.sampled_from([-0.50146835, -0.27606055, -0.14081588, -0.050652758, 0.30999972, 0.40016284, 0.49032595, 0.53540751, 0.71573375, 0.76081531]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.030993, max_value=5.013885, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.44778, max_value=1.56052, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_147(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_147']['n_samples'] += 1
        self.data['tests']['test_147']['samples'].append(x_test)
        self.data['tests']['test_147']['y_expected'].append(y_expected[0])
        self.data['tests']['test_147']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.28100872, -0.21647382, -0.20783529, -0.13872705, -0.091469211, -0.068602514, 0.072324092, 0.48765106, 0.62366556, 1.6533445]),
           st.sampled_from([-0.2136264, -0.17823759, -0.010140702, 0.35701828, 0.52069157, 1.0338294, 2.4493821, 2.5865138, 2.8652008, 4.4665448]),
           st.sampled_from([0.2198366, 1.9329359, 2.3386699, 2.6993224, 3.0148933, 3.555872, 4.7730741, 6.9369889, 10.678758, 11.760716]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=3.030993, max_value=5.013885, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.560523, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_148(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_148']['n_samples'] += 1
        self.data['tests']['test_148']['samples'].append(x_test)
        self.data['tests']['test_148']['y_expected'].append(y_expected[0])
        self.data['tests']['test_148']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.36671649, -0.22341852, -0.13381495, -0.11670727, -0.036250373, 0.22239738, 0.70208986, 0.73579707, 0.79541986, 0.84352462]),
           st.sampled_from([-0.19150839, 0.18007419, 0.57377479, 1.0515238, 1.3080928, 1.6619809, 3.6658728, 3.8605113, 4.1215038, 4.7363845]),
           st.sampled_from([0.30999972, 1.5722834, 1.7075281, 2.7444039, 3.1952195, 3.3304642, 3.555872, 6.9369889, 10.678758, 23.977818]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=5.013888, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.044035, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_149(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_149']['n_samples'] += 1
        self.data['tests']['test_149']['samples'].append(x_test)
        self.data['tests']['test_149']['y_expected'].append(y_expected[0])
        self.data['tests']['test_149']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.028797523, 0.02150921, 0.26203299, 0.30878268, 0.5449025, 0.7673023, 1.1072539, 1.2656269, 1.2815489, 6.1960255]),
           st.sampled_from([-0.42595931, -0.31094565, -0.26670963, -0.195932, -0.023411508, 0.14910898, 0.61801082, 0.91439217, 0.92323937, 1.0869127]),
           st.sampled_from([-0.41130523, -0.18589744, -0.14081588, -0.095734317, -0.050652758, 0.2198366, 0.49032595, 0.53540751, 0.71573375, 1.9780174]),
           st.floats(min_value=-0.85955255, max_value=1.190673, allow_nan=False),
           st.floats(min_value=5.013888, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.044038, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_150(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_150']['n_samples'] += 1
        self.data['tests']['test_150']['samples'].append(x_test)
        self.data['tests']['test_150']['y_expected'].append(y_expected[0])
        self.data['tests']['test_150']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.78441482, -0.48562331, -0.48172751, 0.014225744, 0.11754934, 0.13635084, 0.14583629, 0.20766106, 0.26237175, 0.94142796]),
           st.sampled_from([-0.25786243, -0.058800326, 0.003130105, 0.193345, 0.582622, 1.5912033, 1.9539387, 2.6263262, 3.5685535, 4.6346417]),
           st.sampled_from([-0.54654991, -0.50146835, -0.36622367, -0.14081588, 1.8427727, 3.4206273, 4.1419323, 6.2607655, 10.273024, 20.686865]),
           st.floats(min_value=1.190676, max_value=1.383502, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.030993, max_value=3.17978, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.563567, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_151(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_151']['n_samples'] += 1
        self.data['tests']['test_151']['samples'].append(x_test)
        self.data['tests']['test_151']['y_expected'].append(y_expected[0])
        self.data['tests']['test_151']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.22494297, -0.19021947, -0.061657814, 0.22442997, 0.35790373, 0.36230769, 0.37433387, 0.55930005, 0.61350259, 3.7801167]),
           st.sampled_from([-0.37729969, -0.23132081, -0.16054318, -0.098612747, -0.058800326, -0.010140702, 0.21988661, 0.28181705, 0.7905313, 0.89227416]),
           st.sampled_from([-0.54654991, -0.50146835, -0.45638679, -0.36622367, -0.27606055, 0.49032595, 0.53540751, 0.76081531, 0.89605999, 1.9780174]),
           st.floats(min_value=1.190676, max_value=1.383502, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.179783, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.563567, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_152(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_152']['n_samples'] += 1
        self.data['tests']['test_152']['samples'].append(x_test)
        self.data['tests']['test_152']['y_expected'].append(y_expected[0])
        self.data['tests']['test_152']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.51391026, -0.23764669, -0.13974335, -0.07639413, -0.025240482, -0.0093184852, 0.042173929, 0.53575582, 0.8894274, 1.1275798]),
           st.sampled_from([-0.29325125, 0.18007419, 0.48972635, 0.92323937, 1.6354393, 2.462653, 3.0023324, 4.0330317, 4.7275373, 4.9354466]),
           st.sampled_from([0.12967348, 1.6173649, 1.8427727, 2.2034252, 2.6091592, 3.961606, 4.1870138, 4.7730741, 4.9984819, 7.2525599]),
           st.floats(min_value=1.190676, max_value=1.383502, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.030993, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.56357, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_153(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_153']['n_samples'] += 1
        self.data['tests']['test_153']['samples'].append(x_test)
        self.data['tests']['test_153']['y_expected'].append(y_expected[0])
        self.data['tests']['test_153']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.45377932, 0.14041603, 0.21579144, 0.26203299, 0.27947943, 0.32131701, 0.80338086, 1.0035915, 1.7732676, 3.7801167]),
           st.sampled_from([-0.43923012, -0.42595931, -0.4082649, -0.4038413, -0.28440404, -0.27555684, -0.10303635, 0.082754945, 0.1889214, 0.72860087]),
           st.sampled_from([-0.45638679, -0.41130523, -0.18589744, -0.050652758, -0.0055711987, 0.12967348, 0.2198366, 0.30999972, 0.53540751, 0.89605999]),
           st.floats(min_value=1.383505, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.030993, max_value=3.112859, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0440369, 1.1567798, 1.2146748, 1.220769, 1.421878, 1.4310193, 1.4645375, 1.5011028, 1.6138457, 1.8515199]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_154(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_154']['n_samples'] += 1
        self.data['tests']['test_154']['samples'].append(x_test)
        self.data['tests']['test_154']['y_expected'].append(y_expected[0])
        self.data['tests']['test_154']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.54507673, -0.5035779, -0.27118451, -0.093671189, -0.066908685, 0.50882393, 0.58013415, 0.63467545, 0.90043729, 1.1623033]),
           st.sampled_from([0.56935119, 1.794689, 1.8787375, 2.1928132, 2.206084, 2.3034033, 3.440269, 4.7054193, 4.7983149, 4.9442938]),
           st.sampled_from([-0.59163147, -0.050652758, -0.0055711987, 1.7075281, 2.023099, 2.1583436, 3.1952195, 4.7730741, 10.994329, 20.686865]),
           st.floats(min_value=1.383505, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.112862, max_value=3.187698, exclude_min=True, allow_nan=False),
           st.sampled_from([0.30663727, 0.4772752, 0.75760894, 0.98004767, 1.0105187, 1.0318485, 1.2542872, 1.3335119, 1.4706317, 1.5589978]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_155(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_155']['n_samples'] += 1
        self.data['tests']['test_155']['samples'].append(x_test)
        self.data['tests']['test_155']['y_expected'].append(y_expected[0])
        self.data['tests']['test_155']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.028966906, 0.26914707, 0.27947943, 0.37551956, 0.40837985, 0.54202299, 0.60317023, 0.7156405, 3.389689, 5.079792]),
           st.sampled_from([-0.45250093, -0.35075807, -0.28882764, -0.14284877, -0.0057170995, 0.003130105, 0.41894872, 0.46318474, 0.7905313, 1.2019263]),
           st.sampled_from([-0.27606055, -0.230979, -0.050652758, -0.0055711987, 0.12967348, 0.17475504, 0.2198366, 0.30999972, 0.53540751, 0.71573375]),
           st.floats(min_value=1.383505, max_value=1.533925, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.187701, max_value=4.632478, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.94572324, 0.529076, 0.96176503, 1.0592724, 1.1202145, 1.2299103, 1.2878054, 1.4828201, 1.565092, 1.6351754]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_156(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_156']['n_samples'] += 1
        self.data['tests']['test_156']['samples'].append(x_test)
        self.data['tests']['test_156']['y_expected'].append(y_expected[0])
        self.data['tests']['test_156']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.34469671, -0.2335815, -0.15702041, -0.083677596, -0.013553059, 0.11331476, 0.5299968, 0.78322429, 1.6821396, 1.8552489]),
           st.sampled_from([-0.036682315, 0.17565059, 0.56050399, 0.63570523, 1.7681474, 1.8433486, 1.9981747, 2.7192219, 3.9091709, 3.9401361]),
           st.sampled_from([2.023099, 3.0599748, 3.1050564, 3.150138, 4.0968507, 4.5927479, 4.9984819, 5.2689712, 9.0558222, 9.8672903]),
           st.floats(min_value=1.383505, max_value=1.533925, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.632481, max_value=5.569105, exclude_min=True, allow_nan=False),
           st.sampled_from([0.24264805, 0.30968437, 0.35234386, 0.73018498, 0.92824687, 1.1293559, 1.3578888, 1.4005483, 1.4279722, 1.796672]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_157(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_157']['n_samples'] += 1
        self.data['tests']['test_157']['samples'].append(x_test)
        self.data['tests']['test_157']['y_expected'].append(y_expected[0])
        self.data['tests']['test_157']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.45377932, -0.03726667, 0.37433387, 0.40126576, 0.42294678, 0.72427903, 0.74985585, 1.2837509, 1.297979, 3.389689]),
           st.sampled_from([-0.44365372, -0.3994177, -0.29767485, -0.25343882, -0.14284877, -0.12957796, -0.0057170995, 0.25527543, 0.72860087, 1.0869127]),
           st.sampled_from([-0.50146835, -0.45638679, -0.27606055, -0.14081588, 0.08459192, 0.12967348, 0.35508128, 0.40016284, 0.44524439, 0.76081531]),
           st.floats(min_value=1.383505, max_value=1.533925, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.569108, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.sampled_from([0.15732908, 0.67838418, 1.0105187, 1.0927906, 1.1080261, 1.4553962, 1.5559507, 1.5864217, 1.5955631, 1.6107986]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_158(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_158']['n_samples'] += 1
        self.data['tests']['test_158']['samples'].append(x_test)
        self.data['tests']['test_158']['y_expected'].append(y_expected[0])
        self.data['tests']['test_158']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.34317226, 0.10891081, 0.41938974, 0.43581988, 0.7156405, 1.2049878, 1.3242334, 2.0076936, 2.0158239, 6.1960255]),
           st.sampled_from([-0.41268851, -0.37729969, -0.28882764, -0.21805001, -0.16939038, -0.049953122, 0.60031641, 0.65782324, 0.72860087, 1.4584952]),
           st.sampled_from([-0.41130523, -0.27606055, -0.18589744, -0.0055711987, 0.039510361, 0.26491816, 0.44524439, 0.49032595, 0.53540751, 0.98622311]),
           st.floats(min_value=1.533928, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.187701, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.94572324, max_value=1.626033, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_159(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_159']['n_samples'] += 1
        self.data['tests']['test_159']['samples'].append(x_test)
        self.data['tests']['test_159']['y_expected'].append(y_expected[0])
        self.data['tests']['test_159']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.48257442, -0.45377932, -0.3477456, -0.23087137, -0.043872605, 0.20291834, 0.24492531, 1.1072539, 1.2815489, 1.8916663]),
           st.sampled_from([-0.4082649, -0.35960528, -0.34633447, -0.32864006, -0.27998044, -0.14284877, -0.0057170995, 0.52511517, 0.65782324, 1.5823561]),
           st.sampled_from([-0.45638679, -0.36622367, -0.32114211, -0.095734317, -0.050652758, 0.039510361, 0.12967348, 0.2198366, 0.26491816, 0.53540751]),
           st.floats(min_value=1.533928, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.187701, max_value=5.364935, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.626036, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_160(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_160']['n_samples'] += 1
        self.data['tests']['test_160']['samples'].append(x_test)
        self.data['tests']['test_160']['y_expected'].append(y_expected[0])
        self.data['tests']['test_160']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([-0.41380495, -0.28558206, -0.2345978, -0.07182079, -0.060472133, 0.063685562, 0.16192767, 0.44632162, 0.50594442, 1.3938498]),
           st.sampled_from([-0.11188355, -0.041105917, 0.83476733, 0.90554496, 1.0780654, 1.449648, 1.5602381, 2.4184169, 4.2851771, 4.3957671]),
           st.sampled_from([-0.32114211, 0.039510361, 0.17475504, 0.71573375, 1.3468756, 2.3837514, 2.428833, 2.9247302, 4.5927479, 7.1173152]),
           st.floats(min_value=1.533928, max_value=9.591164, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.364938, max_value=23.617122, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.626036, max_value=1.9490273, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_161(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_161']['n_samples'] += 1
        self.data['tests']['test_161']['samples'].append(x_test)
        self.data['tests']['test_161']['y_expected'].append(y_expected[0])
        self.data['tests']['test_161']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
