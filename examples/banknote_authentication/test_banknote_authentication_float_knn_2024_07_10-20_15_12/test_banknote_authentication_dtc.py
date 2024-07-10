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
    request.cls.data['n_test'] = 27
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_banknote_authentication_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestDataBankNoteAuthenticationProperty:

    @given(st.floats(min_value=-7.0421, max_value=-0.403101, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=7.293148, allow_nan=False),
           st.floats(min_value=-5.2861, max_value=6.218648, allow_nan=False),
           st.sampled_from([-3.8194, -2.4642, -2.1138, -1.091, -0.13924, 0.42843, 0.7517, 1.0414, 1.106, 1.1965]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-7.0421, max_value=-0.403101, allow_nan=False),
           st.floats(min_value=7.293151, max_value=7.565298, exclude_min=True, allow_nan=False),
           st.floats(min_value=-5.2861, max_value=0.247763, allow_nan=False),
           st.sampled_from([-5.4719, -4.2629, -3.937, -1.8228, -1.4116, -0.85949, -0.77027, 0.74394, 1.0233, 1.1965]))
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

    @given(st.floats(min_value=-7.0421, max_value=-0.403101, allow_nan=False),
           st.floats(min_value=7.293151, max_value=7.565298, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.247766, max_value=6.218648, exclude_min=True, allow_nan=False),
           st.sampled_from([-7.8719, -3.3409, -3.1108, -3.0086, -1.4892, -0.59182, -0.5621, 0.49567, 0.60429, 1.437]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-7.0421, max_value=-0.403101, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=-4.674502, allow_nan=False),
           st.floats(min_value=6.218651, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-6.9642, -6.3862, -4.0327, -2.5017, -2.4642, -0.5026, -0.28406, 0.27972, 0.48533, 0.88231]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-7.0421, max_value=-2.117102, allow_nan=False),
           st.floats(min_value=-4.674499, max_value=-1.391602, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.218651, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-5.2159, -3.1457, -2.9452, -1.9909, -0.41984, -0.1457, 0.16981, 0.42067, 0.96765, 1.5043]))
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

    @given(st.floats(min_value=-7.0421, max_value=-2.117102, allow_nan=False),
           st.floats(min_value=-1.391599, max_value=7.565298, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.218651, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-5.5793, -3.9564, -2.1784, -1.5875, -0.9888, -0.38751, -0.037083, 0.42843, 0.49567, 2.0564]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-2.117099, max_value=-0.403101, exclude_min=True, allow_nan=False),
           st.floats(min_value=-4.674499, max_value=7.565298, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.218651, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.131, -0.54139, -0.34355, -0.2776, -0.24398, -0.037083, 0.18662, 0.47886, 0.71291, 0.84351]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.403098, max_value=0.320163, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=5.453548, allow_nan=False),
           st.floats(min_value=-5.2861, max_value=2.624649, allow_nan=False),
           st.sampled_from([-6.0823, -5.8198, -2.825, -0.94742, -0.66423, -0.40173, 0.057313, 0.11162, 0.9017, 1.106]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.403098, max_value=0.320163, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=5.453548, allow_nan=False),
           st.floats(min_value=2.624652, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=-8.5482, max_value=1.228229, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.403098, max_value=0.320163, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=5.453548, allow_nan=False),
           st.floats(min_value=2.624652, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.228232, max_value=2.4495, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-0.403098, max_value=0.320163, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.453551, max_value=7.565298, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.9351, -0.37816, 0.48382, 0.90967, 1.0708, 2.045, 2.2129, 2.7236, 3.1769, 6.7756]),
           st.sampled_from([-5.4435, -1.872, -1.6677, -1.1608, -0.5582, -0.50648, -0.39786, -0.066824, 0.94955, 1.225]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-7.0421, max_value=-4.726002, allow_nan=False),
           st.floats(min_value=7.565301, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.sampled_from([-3.6684, -2.1964, -1.7302, -1.0002, -0.53324, -0.31218, -0.25392, -0.035421, 0.1548, 11.9552]),
           st.sampled_from([-5.9763, -4.572, -1.9547, -1.2862, -0.94613, -0.66811, -0.3332, -0.1276, 0.54998, 1.0414]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=-4.725999, max_value=0.320163, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.565301, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.sampled_from([-4.0351, -3.9606, -3.3034, 0.61406, 0.84027, 2.1341, 2.8633, 3.8255, 6.0096, 8.1628]),
           st.sampled_from([-6.9978, -3.9202, -1.5875, -1.3754, -0.74182, -0.68234, 0.35084, 1.0129, 1.031, 1.7991]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=6.8248, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=7.191799, allow_nan=False),
           st.floats(min_value=-5.2861, max_value=-4.386051, allow_nan=False),
           st.sampled_from([-2.7965, -1.2862, -1.2823, -0.77027, -0.57889, -0.48579, -0.47027, -0.15605, 0.5461, 0.66119]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=6.8248, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.191802, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.floats(min_value=-5.2861, max_value=-4.386051, allow_nan=False),
           st.sampled_from([-3.1095, -2.2586, -1.9677, -1.3509, -1.1556, -0.60863, -0.40691, 0.35084, 1.1151, 1.2379]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=5.666699, allow_nan=False),
           st.floats(min_value=-4.386048, max_value=-2.272201, exclude_min=True, allow_nan=False),
           st.sampled_from([-5.2159, -4.6832, -1.2953, -1.1724, -1.0871, -0.82716, -0.50001, -0.45346, 0.29524, 0.65214]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.666702, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.floats(min_value=-4.386048, max_value=-2.272201, exclude_min=True, allow_nan=False),
           st.sampled_from([-7.0495, -4.6056, -0.39915, -0.39786, 0.29136, 0.38576, 0.49826, 0.8267, 0.99868, 1.1823]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=0.400864, exclude_min=True, allow_nan=False),
           st.sampled_from([-3.7454, 2.4769, 2.6963, 3.4184, 3.5488, 6.1416, 7.946, 9.0552, 10.4023, 11.4535]),
           st.floats(min_value=-2.272198, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=-8.5482, max_value=0.08188, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.400867, max_value=0.420018, exclude_min=True, allow_nan=False),
           st.sampled_from([-8.304, -5.8126, 0.57318, 0.9022, 1.9327, 3.2405, 3.3377, 3.4329, 7.3708, 7.9295]),
           st.floats(min_value=-2.272198, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=-8.5482, max_value=0.08188, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.420021, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.sampled_from([-3.8696, 0.57732, 3.5757, 5.6864, 5.8312, 7.9274, 8.2274, 9.647, 10.1105, 10.4643]),
           st.floats(min_value=-2.272198, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=-8.5482, max_value=0.08188, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=3.559173, allow_nan=False),
           st.floats(min_value=-2.272198, max_value=1.853049, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.081883, max_value=2.4495, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.559176, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.floats(min_value=-2.272198, max_value=1.853049, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.081883, max_value=2.4495, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.320166, max_value=1.592198, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76771, 0.23175, 1.3781, 2.2493, 2.4107, 3.0294, 5.2187, 6.3485, 6.7769, 9.9946]),
           st.floats(min_value=1.853052, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.081883, max_value=2.4495, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.592201, max_value=2.036548, exclude_min=True, allow_nan=False),
           st.floats(min_value=-13.7731, max_value=6.472699, allow_nan=False),
           st.floats(min_value=-4.386048, max_value=-2.648352, exclude_min=True, allow_nan=False),
           st.sampled_from([-6.3694, -3.456, -2.8948, -1.7323, -1.0276, -1.0043, -0.34872, -0.21036, 0.59007, 0.77369]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.592201, max_value=2.036548, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.472702, max_value=12.9516, exclude_min=True, allow_nan=False),
           st.floats(min_value=-4.386048, max_value=-2.648352, exclude_min=True, allow_nan=False),
           st.sampled_from([-4.8448, -3.3409, -2.7746, -1.3884, -1.1698, -0.59182, -0.43277, 0.29653, 0.44912, 0.62627]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.592201, max_value=2.036548, exclude_min=True, allow_nan=False),
           st.sampled_from([-4.4076, 0.46351, 0.71596, 0.9829, 1.4816, 1.552, 7.2321, 8.7261, 9.0862, 11.1472]),
           st.floats(min_value=-2.648349, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-4.4039, -3.9784, -3.7133, -2.4099, -0.96811, -0.79742, -0.18967, 1.1797, 1.1875, 1.5534]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.036551, max_value=6.8248, exclude_min=True, allow_nan=False),
           st.sampled_from([-6.4748, -5.1153, -5.0325, -0.63321, 0.27727, 1.3781, 5.9947, 8.96, 10.3567, 10.8223]),
           st.floats(min_value=-4.386048, max_value=17.9274, exclude_min=True, allow_nan=False),
           st.sampled_from([-5.2107, -4.4194, -3.7483, -3.7405, -1.3509, -0.70432, -0.017686, 0.22283, 0.95343, 1.2741]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
