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
    request.cls.data['n_test'] = 198
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_blood_transfusion_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestBloodTransfusionProperty:

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=1, max_value=1.4, allow_nan=False),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=2, max_value=2.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=1.6, max_value=11.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=2, max_value=2.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.sampled_from([1, 3, 4, 5, 8, 9, 19, 20, 34, 41]),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=2.6, max_value=2.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.sampled_from([1, 2, 3, 7, 8, 12, 14, 17, 22, 24]),
           st.floats(min_value=350.0, max_value=374.9, allow_nan=False),
           st.floats(min_value=3.6, max_value=6.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.8, max_value=0.9, allow_nan=False),
           st.sampled_from([1, 6, 8, 9, 12, 14, 15, 18, 23, 38]),
           st.floats(min_value=375.1, max_value=525.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=5.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.1, max_value=1.4, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 3, 4, 5, 7, 9, 11, 13, 14, 38]),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=3.1, max_value=3.4, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 4, 9, 12, 13, 15, 18, 22, 24, 44]),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=1.1, max_value=1.4, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 3, 7, 10, 11, 12, 18, 19, 22, 23]),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.6, max_value=7.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.4, exclude_min=True, allow_nan=False),
           st.sampled_from([3, 4, 8, 9, 10, 12, 13, 15, 24, 44]),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.6, max_value=7.3, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=1.1, max_value=1.8, exclude_min=True, allow_nan=False),
           st.sampled_from([3, 6, 7, 8, 9, 12, 17, 18, 19, 44]),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=10.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.1, max_value=1.8, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 8, 9, 11, 13, 19, 22, 26, 33, 41]),
           st.floats(min_value=625.1, max_value=675.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=4.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.1, max_value=1.8, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 3, 7, 9, 11, 13, 16, 17, 19, 38]),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=4.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.1, max_value=1.8, exclude_min=True, allow_nan=False),
           st.sampled_from([5, 8, 9, 14, 15, 18, 23, 24, 38, 44]),
           st.floats(min_value=625.1, max_value=725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.1, max_value=10.4, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=5.1, max_value=5.3, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 3, 4, 5, 8, 12, 13, 16, 17, 20]),
           st.floats(min_value=375.1, max_value=525.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.6, max_value=5.2, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=375.1, max_value=525.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.1, max_value=13.0, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=4.0, max_value=4.9, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=375.1, max_value=475.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.1, max_value=12.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.1, max_value=12.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.1, max_value=12.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=375.1, max_value=475.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.1, max_value=15.4, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=3.1, max_value=3.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=375.1, max_value=475.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.1, max_value=15.4, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.1, max_value=15.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.1, max_value=15.4, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=5.1, max_value=5.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=375.1, max_value=525.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.1, max_value=13.0, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.sampled_from([3, 5, 6, 9, 13, 14, 15, 18, 22, 23]),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=17.1, max_value=18.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.sampled_from([1, 4, 9, 11, 19, 26, 34, 41, 46, 50]),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=24.6, max_value=24.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=25.6, max_value=30.1, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=48.6, max_value=49.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=950.0, max_value=1124.9, allow_nan=False),
           st.floats(min_value=51.6, max_value=60.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=25.6, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=36.6, max_value=38.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=45.1, max_value=48.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=63.6, max_value=66.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=76.1, max_value=80.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.6, max_value=40.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=25.6, max_value=40.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.6, max_value=25.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.6, max_value=29.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=37.6, max_value=38.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=42.1, max_value=42.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=44.6, max_value=55.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.8, max_value=12.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=4.6, max_value=5.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.1, max_value=13.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=5.6, max_value=7.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.1, max_value=13.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.6, max_value=7.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.1, max_value=13.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=3600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1225.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.6, max_value=19.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=23.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.6, max_value=30.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=33.6, max_value=34.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=22.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.6, max_value=25.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.1, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.6, max_value=31.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=33.6, max_value=33.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=34.6, max_value=35.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=22.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.1, max_value=24.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.1, max_value=27.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=23.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.1, max_value=30.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.1, max_value=29.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=32.6, max_value=33.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=4.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=36.6, max_value=36.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=24.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=38.6, max_value=38.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1225.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=40.1, max_value=40.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=42.1, max_value=43.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=42.1, max_value=43.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=11.8, max_value=14.4, allow_nan=False),
           st.floats(min_value=1375.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=42.1, max_value=43.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.6, max_value=19.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=22.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=21.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.6, max_value=23.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.6, max_value=25.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.1, max_value=27.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.6, max_value=28.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=1725.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.6, max_value=34.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2125.1, max_value=2175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=22.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2125.1, max_value=2175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.1, max_value=29.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2125.1, max_value=2175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.1, max_value=28.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2125.1, max_value=2175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=33.1, max_value=34.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2125.1, max_value=2175.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.1, max_value=40.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=2375.1, max_value=4400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=26.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.4, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=48.6, max_value=48.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.8, max_value=0.9, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.6, max_value=20.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.1, max_value=2.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.6, max_value=20.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.2, max_value=1.4, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.1, max_value=33.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6, max_value=2.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.1, max_value=32.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6, max_value=2.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=43.6, max_value=44.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6, max_value=2.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=46.6, max_value=47.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.6, max_value=1.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.1, max_value=33.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1625.1, max_value=3800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.1, max_value=33.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=14.6, max_value=21.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=40.0, max_value=49.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=51.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=57.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.0, max_value=2.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.6, max_value=66.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.6, max_value=3.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=59.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=2625.1, max_value=4600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=52.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=2625.1, max_value=4600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.6, max_value=64.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.4, max_value=2.9, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=2625.1, max_value=4600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.6, max_value=74.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.1, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=2625.1, max_value=4600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=72.6, max_value=74.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=10.2, max_value=12.4, allow_nan=False),
           st.floats(min_value=2625.1, max_value=4600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=80.6, max_value=84.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_104(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_104']['n_samples'] += 1
        self.data['tests']['test_104']['samples'].append(x_test)
        self.data['tests']['test_104']['y_expected'].append(y_expected[0])
        self.data['tests']['test_104']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=12.6, max_value=15.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=51.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.4, allow_nan=False),
           st.floats(min_value=12.6, max_value=15.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1575.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_106(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_106']['n_samples'] += 1
        self.data['tests']['test_106']['samples'].append(x_test)
        self.data['tests']['test_106']['y_expected'].append(y_expected[0])
        self.data['tests']['test_106']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=2.8, max_value=3.4, allow_nan=False),
           st.floats(min_value=12.6, max_value=15.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3375.1, max_value=5200.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.6, max_value=15.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=1650.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_108(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_108']['n_samples'] += 1
        self.data['tests']['test_108']['samples'].append(x_test)
        self.data['tests']['test_108']['y_expected'].append(y_expected[0])
        self.data['tests']['test_108']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.6, max_value=14.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=3750.1, max_value=5500.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_109(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_109']['n_samples'] += 1
        self.data['tests']['test_109']['samples'].append(x_test)
        self.data['tests']['test_109']['y_expected'].append(y_expected[0])
        self.data['tests']['test_109']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.6, max_value=3.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=22.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=3750.1, max_value=5500.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.6, max_value=15.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=57.6, max_value=65.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_111(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_111']['n_samples'] += 1
        self.data['tests']['test_111']['samples'].append(x_test)
        self.data['tests']['test_111']['y_expected'].append(y_expected[0])
        self.data['tests']['test_111']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=25.1, max_value=30.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3075.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=59.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_112(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_112']['n_samples'] += 1
        self.data['tests']['test_112']['samples'].append(x_test)
        self.data['tests']['test_112']['y_expected'].append(y_expected[0])
        self.data['tests']['test_112']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=25.1, max_value=29.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10875.1, max_value=11200.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=59.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_113(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_113']['n_samples'] += 1
        self.data['tests']['test_113']['samples'].append(x_test)
        self.data['tests']['test_113']['y_expected'].append(y_expected[0])
        self.data['tests']['test_113']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.2, max_value=6.4, allow_nan=False),
           st.floats(min_value=45.1, max_value=46.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=10875.1, max_value=11200.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=49.6, max_value=59.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_114(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_114']['n_samples'] += 1
        self.data['tests']['test_114']['samples'].append(x_test)
        self.data['tests']['test_114']['y_expected'].append(y_expected[0])
        self.data['tests']['test_114']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1, max_value=1.4, allow_nan=False),
           st.sampled_from([750, 1000, 1500, 1750, 2750, 3000, 3750, 4750, 5750, 9500]),
           st.floats(min_value=8.4, max_value=9.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_115(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_115']['n_samples'] += 1
        self.data['tests']['test_115']['samples'].append(x_test)
        self.data['tests']['test_115']['y_expected'].append(y_expected[0])
        self.data['tests']['test_115']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1, max_value=1.4, allow_nan=False),
           st.sampled_from([250, 1000, 2250, 2750, 3000, 3500, 3750, 4000, 5750, 9500]),
           st.floats(min_value=10.1, max_value=10.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_116(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_116']['n_samples'] += 1
        self.data['tests']['test_116']['samples'].append(x_test)
        self.data['tests']['test_116']['y_expected'].append(y_expected[0])
        self.data['tests']['test_116']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6, max_value=1.7, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 500, 750, 1000, 1250, 2250, 2500, 2750, 4750, 9500]),
           st.floats(min_value=10.4, max_value=12.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_117(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_117']['n_samples'] += 1
        self.data['tests']['test_117']['samples'].append(x_test)
        self.data['tests']['test_117']['y_expected'].append(y_expected[0])
        self.data['tests']['test_117']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=350.0, max_value=374.9, allow_nan=False),
           st.floats(min_value=12.6, max_value=13.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_118(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_118']['n_samples'] += 1
        self.data['tests']['test_118']['samples'].append(x_test)
        self.data['tests']['test_118']['y_expected'].append(y_expected[0])
        self.data['tests']['test_118']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=375.1, max_value=2800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.6, max_value=13.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_119(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_119']['n_samples'] += 1
        self.data['tests']['test_119']['samples'].append(x_test)
        self.data['tests']['test_119']['y_expected'].append(y_expected[0])
        self.data['tests']['test_119']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.4, allow_nan=False),
           st.floats(min_value=375.1, max_value=2800.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=12.6, max_value=13.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_120(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_120']['n_samples'] += 1
        self.data['tests']['test_120']['samples'].append(x_test)
        self.data['tests']['test_120']['y_expected'].append(y_expected[0])
        self.data['tests']['test_120']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 1000, 1250, 2500, 3250, 4000, 4250, 4500, 5750, 6000]),
           st.floats(min_value=12.8, max_value=15.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_121(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_121']['n_samples'] += 1
        self.data['tests']['test_121']['samples'].append(x_test)
        self.data['tests']['test_121']['y_expected'].append(y_expected[0])
        self.data['tests']['test_121']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.1, max_value=12.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.sampled_from([1250, 1750, 2000, 2500, 3250, 3750, 4000, 5500, 6500, 12500]),
           st.floats(min_value=12.8, max_value=15.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_122(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_122']['n_samples'] += 1
        self.data['tests']['test_122']['samples'].append(x_test)
        self.data['tests']['test_122']['y_expected'].append(y_expected[0])
        self.data['tests']['test_122']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6, max_value=12.0, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 1000, 1250, 1500, 2000, 2250, 2750, 5750, 6000, 9500]),
           st.floats(min_value=12.8, max_value=15.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_123(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_123']['n_samples'] += 1
        self.data['tests']['test_123']['samples'].append(x_test)
        self.data['tests']['test_123']['y_expected'].append(y_expected[0])
        self.data['tests']['test_123']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=6.9, exclude_min=True, allow_nan=False),
           st.sampled_from([1, 2, 3, 8, 11, 18, 19, 22, 23, 44]),
           st.sampled_from([250, 500, 2500, 3250, 4000, 4250, 4500, 5500, 5750, 11000]),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_124(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_124']['n_samples'] += 1
        self.data['tests']['test_124']['samples'].append(x_test)
        self.data['tests']['test_124']['y_expected'].append(y_expected[0])
        self.data['tests']['test_124']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=8.6, max_value=8.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.4, max_value=6.4, allow_nan=False),
           st.sampled_from([1000, 1500, 1750, 2000, 3500, 3750, 4250, 5500, 6500, 11500]),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_125(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_125']['n_samples'] += 1
        self.data['tests']['test_125']['samples'].append(x_test)
        self.data['tests']['test_125']['y_expected'].append(y_expected[0])
        self.data['tests']['test_125']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=10.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.4, max_value=6.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_126(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_126']['n_samples'] += 1
        self.data['tests']['test_126']['samples'].append(x_test)
        self.data['tests']['test_126']['y_expected'].append(y_expected[0])
        self.data['tests']['test_126']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=9.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.4, max_value=6.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_127(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_127']['n_samples'] += 1
        self.data['tests']['test_127']['samples'].append(x_test)
        self.data['tests']['test_127']['y_expected'].append(y_expected[0])
        self.data['tests']['test_127']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.6, max_value=10.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.4, max_value=6.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_128(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_128']['n_samples'] += 1
        self.data['tests']['test_128']['samples'].append(x_test)
        self.data['tests']['test_128']['y_expected'].append(y_expected[0])
        self.data['tests']['test_128']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=8.6, max_value=9.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.6, max_value=15.2, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 750, 1000, 1500, 1750, 2250, 3500, 5500, 9500, 11000]),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_129(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_129']['n_samples'] += 1
        self.data['tests']['test_129']['samples'].append(x_test)
        self.data['tests']['test_129']['y_expected'].append(y_expected[0])
        self.data['tests']['test_129']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.1, max_value=12.3, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 4, 11, 13, 16, 22, 23, 24, 38, 44]),
           st.sampled_from([750, 1500, 1750, 3000, 3500, 3750, 4000, 5500, 5750, 9500]),
           st.floats(min_value=15.6, max_value=15.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_130(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_130']['n_samples'] += 1
        self.data['tests']['test_130']['samples'].append(x_test)
        self.data['tests']['test_130']['y_expected'].append(y_expected[0])
        self.data['tests']['test_130']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.9, exclude_min=True, allow_nan=False),
           st.sampled_from([1, 2, 3, 4, 8, 15, 17, 18, 24, 44]),
           st.sampled_from([250, 500, 1250, 1750, 2000, 3500, 3750, 4500, 4750, 5750]),
           st.floats(min_value=16.6, max_value=17.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_131(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_131']['n_samples'] += 1
        self.data['tests']['test_131']['samples'].append(x_test)
        self.data['tests']['test_131']['y_expected'].append(y_expected[0])
        self.data['tests']['test_131']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([4, 5, 7, 9, 10, 14, 15, 16, 18, 24]),
           st.sampled_from([250, 750, 1250, 2500, 3750, 4500, 4750, 5500, 5750, 6000]),
           st.floats(min_value=15.6, max_value=15.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_132(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_132']['n_samples'] += 1
        self.data['tests']['test_132']['samples'].append(x_test)
        self.data['tests']['test_132']['y_expected'].append(y_expected[0])
        self.data['tests']['test_132']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([3, 4, 6, 8, 13, 17, 21, 26, 41, 46]),
           st.floats(min_value=750.0, max_value=874.9, allow_nan=False),
           st.floats(min_value=17.6, max_value=18.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_133(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_133']['n_samples'] += 1
        self.data['tests']['test_133']['samples'].append(x_test)
        self.data['tests']['test_133']['y_expected'].append(y_expected[0])
        self.data['tests']['test_133']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([2, 3, 8, 14, 20, 21, 33, 41, 43, 50]),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=20.1, max_value=20.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_134(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_134']['n_samples'] += 1
        self.data['tests']['test_134']['samples'].append(x_test)
        self.data['tests']['test_134']['y_expected'].append(y_expected[0])
        self.data['tests']['test_134']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([5, 7, 8, 9, 11, 13, 17, 22, 24, 38]),
           st.floats(min_value=625.1, max_value=675.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=20.1, max_value=20.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_135(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_135']['n_samples'] += 1
        self.data['tests']['test_135']['samples'].append(x_test)
        self.data['tests']['test_135']['y_expected'].append(y_expected[0])
        self.data['tests']['test_135']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=13.6, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1, 3, 4, 10, 11, 15, 20, 21, 22, 43]),
           st.floats(min_value=875.1, max_value=3200.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=17.6, max_value=18.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_136(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_136']['n_samples'] += 1
        self.data['tests']['test_136']['samples'].append(x_test)
        self.data['tests']['test_136']['y_expected'].append(y_expected[0])
        self.data['tests']['test_136']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.sampled_from([750, 1000, 2250, 3000, 3250, 3500, 3750, 4000, 4250, 5500]),
           st.floats(min_value=23.6, max_value=24.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_137(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_137']['n_samples'] += 1
        self.data['tests']['test_137']['samples'].append(x_test)
        self.data['tests']['test_137']['y_expected'].append(y_expected[0])
        self.data['tests']['test_137']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=27.6, max_value=28.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_138(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_138']['n_samples'] += 1
        self.data['tests']['test_138']['samples'].append(x_test)
        self.data['tests']['test_138']['y_expected'].append(y_expected[0])
        self.data['tests']['test_138']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=32.1, max_value=32.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_139(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_139']['n_samples'] += 1
        self.data['tests']['test_139']['samples'].append(x_test)
        self.data['tests']['test_139']['y_expected'].append(y_expected[0])
        self.data['tests']['test_139']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=36.6, max_value=39.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_140(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_140']['n_samples'] += 1
        self.data['tests']['test_140']['samples'].append(x_test)
        self.data['tests']['test_140']['y_expected'].append(y_expected[0])
        self.data['tests']['test_140']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.6, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_141(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_141']['n_samples'] += 1
        self.data['tests']['test_141']['samples'].append(x_test)
        self.data['tests']['test_141']['y_expected'].append(y_expected[0])
        self.data['tests']['test_141']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=675.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.6, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_142(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_142']['n_samples'] += 1
        self.data['tests']['test_142']['samples'].append(x_test)
        self.data['tests']['test_142']['y_expected'].append(y_expected[0])
        self.data['tests']['test_142']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=875.1, max_value=925.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.6, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_143(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_143']['n_samples'] += 1
        self.data['tests']['test_143']['samples'].append(x_test)
        self.data['tests']['test_143']['y_expected'].append(y_expected[0])
        self.data['tests']['test_143']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=1125.1, max_value=3400.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=27.6, max_value=27.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_144(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_144']['n_samples'] += 1
        self.data['tests']['test_144']['samples'].append(x_test)
        self.data['tests']['test_144']['y_expected'].append(y_expected[0])
        self.data['tests']['test_144']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=28.6, max_value=30.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_145(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_145']['n_samples'] += 1
        self.data['tests']['test_145']['samples'].append(x_test)
        self.data['tests']['test_145']['y_expected'].append(y_expected[0])
        self.data['tests']['test_145']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=700.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=36.1, max_value=36.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_146(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_146']['n_samples'] += 1
        self.data['tests']['test_146']['samples'].append(x_test)
        self.data['tests']['test_146']['y_expected'].append(y_expected[0])
        self.data['tests']['test_146']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=1000.1, max_value=1100.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=36.1, max_value=36.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_147(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_147']['n_samples'] += 1
        self.data['tests']['test_147']['samples'].append(x_test)
        self.data['tests']['test_147']['y_expected'].append(y_expected[0])
        self.data['tests']['test_147']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=1500.1, max_value=3700.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=36.1, max_value=36.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_148(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_148']['n_samples'] += 1
        self.data['tests']['test_148']['samples'].append(x_test)
        self.data['tests']['test_148']['y_expected'].append(y_expected[0])
        self.data['tests']['test_148']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=37.6, max_value=37.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_149(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_149']['n_samples'] += 1
        self.data['tests']['test_149']['samples'].append(x_test)
        self.data['tests']['test_149']['y_expected'].append(y_expected[0])
        self.data['tests']['test_149']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.1, max_value=39.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_150(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_150']['n_samples'] += 1
        self.data['tests']['test_150']['samples'].append(x_test)
        self.data['tests']['test_150']['y_expected'].append(y_expected[0])
        self.data['tests']['test_150']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=36.1, max_value=36.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_151(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_151']['n_samples'] += 1
        self.data['tests']['test_151']['samples'].append(x_test)
        self.data['tests']['test_151']['y_expected'].append(y_expected[0])
        self.data['tests']['test_151']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=40.6, max_value=42.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_152(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_152']['n_samples'] += 1
        self.data['tests']['test_152']['samples'].append(x_test)
        self.data['tests']['test_152']['y_expected'].append(y_expected[0])
        self.data['tests']['test_152']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=50.6, max_value=50.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_153(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_153']['n_samples'] += 1
        self.data['tests']['test_153']['samples'].append(x_test)
        self.data['tests']['test_153']['y_expected'].append(y_expected[0])
        self.data['tests']['test_153']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=11.6, max_value=12.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=50.6, max_value=50.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_154(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_154']['n_samples'] += 1
        self.data['tests']['test_154']['samples'].append(x_test)
        self.data['tests']['test_154']['y_expected'].append(y_expected[0])
        self.data['tests']['test_154']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.sampled_from([500, 750, 1000, 1500, 2750, 3750, 4750, 5500, 9500, 11000]),
           st.floats(min_value=51.6, max_value=53.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_155(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_155']['n_samples'] += 1
        self.data['tests']['test_155']['samples'].append(x_test)
        self.data['tests']['test_155']['y_expected'].append(y_expected[0])
        self.data['tests']['test_155']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.sampled_from([250, 750, 1750, 3250, 3750, 4250, 4750, 5500, 8500, 10750]),
           st.floats(min_value=61.6, max_value=61.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_156(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_156']['n_samples'] += 1
        self.data['tests']['test_156']['samples'].append(x_test)
        self.data['tests']['test_156']['y_expected'].append(y_expected[0])
        self.data['tests']['test_156']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.2, max_value=7.4, allow_nan=False),
           st.sampled_from([250, 500, 750, 1000, 1250, 2500, 3000, 3500, 4500, 5750]),
           st.floats(min_value=63.1, max_value=70.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_157(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_157']['n_samples'] += 1
        self.data['tests']['test_157']['samples'].append(x_test)
        self.data['tests']['test_157']['y_expected'].append(y_expected[0])
        self.data['tests']['test_157']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 2500, 3000, 3250, 4000, 4250, 5000, 5250, 6500, 10750]),
           st.floats(min_value=23.6, max_value=38.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_158(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_158']['n_samples'] += 1
        self.data['tests']['test_158']['samples'].append(x_test)
        self.data['tests']['test_158']['y_expected'].append(y_expected[0])
        self.data['tests']['test_158']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=9.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([750, 1500, 1750, 3000, 4000, 4250, 4750, 5500, 9500, 11000]),
           st.floats(min_value=23.6, max_value=27.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_159(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_159']['n_samples'] += 1
        self.data['tests']['test_159']['samples'].append(x_test)
        self.data['tests']['test_159']['y_expected'].append(y_expected[0])
        self.data['tests']['test_159']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.6, max_value=10.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1500, 1750, 2500, 4250, 5000, 5500, 8250, 8500, 10750, 11500]),
           st.floats(min_value=23.6, max_value=27.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_160(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_160']['n_samples'] += 1
        self.data['tests']['test_160']['samples'].append(x_test)
        self.data['tests']['test_160']['y_expected'].append(y_expected[0])
        self.data['tests']['test_160']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=10.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1500, 1750, 2250, 3250, 3500, 4000, 4500, 5500, 9500, 11000]),
           st.floats(min_value=43.6, max_value=44.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_161(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_161']['n_samples'] += 1
        self.data['tests']['test_161']['samples'].append(x_test)
        self.data['tests']['test_161']['y_expected'].append(y_expected[0])
        self.data['tests']['test_161']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.1, max_value=12.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 1750, 2500, 3250, 3750, 4000, 4750, 5500, 5750, 11000]),
           st.floats(min_value=23.6, max_value=28.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_162(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_162']['n_samples'] += 1
        self.data['tests']['test_162']['samples'].append(x_test)
        self.data['tests']['test_162']['y_expected'].append(y_expected[0])
        self.data['tests']['test_162']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=10.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 2000, 2250, 2750, 3500, 3750, 4000, 4750, 5000, 11500]),
           st.floats(min_value=49.1, max_value=51.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_163(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_163']['n_samples'] += 1
        self.data['tests']['test_163']['samples'].append(x_test)
        self.data['tests']['test_163']['y_expected'].append(y_expected[0])
        self.data['tests']['test_163']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=12.6, max_value=12.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 4750, 11000]),
           st.floats(min_value=49.1, max_value=51.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_164(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_164']['n_samples'] += 1
        self.data['tests']['test_164']['samples'].append(x_test)
        self.data['tests']['test_164']['y_expected'].append(y_expected[0])
        self.data['tests']['test_164']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.6, max_value=10.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([750, 1000, 2000, 2250, 2500, 3000, 4250, 4500, 4750, 5750]),
           st.floats(min_value=62.1, max_value=69.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_165(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_165']['n_samples'] += 1
        self.data['tests']['test_165']['samples'].append(x_test)
        self.data['tests']['test_165']['y_expected'].append(y_expected[0])
        self.data['tests']['test_165']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=7.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=8.9, exclude_min=True, allow_nan=False),
           st.sampled_from([1750, 2250, 2500, 3000, 3500, 3750, 4500, 5500, 6000, 9500]),
           st.floats(min_value=23.6, max_value=33.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_166(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_166']['n_samples'] += 1
        self.data['tests']['test_166']['samples'].append(x_test)
        self.data['tests']['test_166']['y_expected'].append(y_expected[0])
        self.data['tests']['test_166']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=6.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=18.4, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 1000, 1750, 2000, 3000, 3500, 4500, 4750, 6000, 9500]),
           st.floats(min_value=23.6, max_value=33.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_167(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_167']['n_samples'] += 1
        self.data['tests']['test_167']['samples'].append(x_test)
        self.data['tests']['test_167']['y_expected'].append(y_expected[0])
        self.data['tests']['test_167']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=8.1, max_value=8.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=18.4, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 500, 750, 1000, 1750, 2500, 3750, 8250, 10250, 12500]),
           st.floats(min_value=23.6, max_value=33.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_168(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_168']['n_samples'] += 1
        self.data['tests']['test_168']['samples'].append(x_test)
        self.data['tests']['test_168']['y_expected'].append(y_expected[0])
        self.data['tests']['test_168']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.1, max_value=10.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=16.8, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 500, 1000, 1750, 2500, 2750, 3750, 6000, 9500, 11000]),
           st.floats(min_value=23.6, max_value=33.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_169(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_169']['n_samples'] += 1
        self.data['tests']['test_169']['samples'].append(x_test)
        self.data['tests']['test_169']['y_expected'].append(y_expected[0])
        self.data['tests']['test_169']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=16.8, exclude_min=True, allow_nan=False),
           st.sampled_from([1500, 1750, 2500, 2750, 3750, 4000, 4250, 8500, 10750, 11500]),
           st.floats(min_value=71.6, max_value=72.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_170(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_170']['n_samples'] += 1
        self.data['tests']['test_170']['samples'].append(x_test)
        self.data['tests']['test_170']['y_expected'].append(y_expected[0])
        self.data['tests']['test_170']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=16.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=2050.0, max_value=2499.9, allow_nan=False),
           st.floats(min_value=75.1, max_value=79.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_171(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_171']['n_samples'] += 1
        self.data['tests']['test_171']['samples'].append(x_test)
        self.data['tests']['test_171']['y_expected'].append(y_expected[0])
        self.data['tests']['test_171']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.6, max_value=10.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=2500.1, max_value=4500.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=75.1, max_value=79.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_172(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_172']['n_samples'] += 1
        self.data['tests']['test_172']['samples'].append(x_test)
        self.data['tests']['test_172']['y_expected'].append(y_expected[0])
        self.data['tests']['test_172']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=16.6, max_value=23.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=2500.1, max_value=2875.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=75.1, max_value=79.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_173(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_173']['n_samples'] += 1
        self.data['tests']['test_173']['samples'].append(x_test)
        self.data['tests']['test_173']['y_expected'].append(y_expected[0])
        self.data['tests']['test_173']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.6, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=16.6, max_value=23.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=4375.1, max_value=6000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=75.1, max_value=79.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_174(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_174']['n_samples'] += 1
        self.data['tests']['test_174']['samples'].append(x_test)
        self.data['tests']['test_174']['y_expected'].append(y_expected[0])
        self.data['tests']['test_174']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=15.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=21.6, max_value=26.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_175(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_175']['n_samples'] += 1
        self.data['tests']['test_175']['samples'].append(x_test)
        self.data['tests']['test_175']['y_expected'].append(y_expected[0])
        self.data['tests']['test_175']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=19.6, max_value=30.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1, max_value=1.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=17.6, max_value=21.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_176(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_176']['n_samples'] += 1
        self.data['tests']['test_176']['samples'].append(x_test)
        self.data['tests']['test_176']['y_expected'].append(y_expected[0])
        self.data['tests']['test_176']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=19.6, max_value=30.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.6, max_value=2.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=17.6, max_value=21.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_177(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_177']['n_samples'] += 1
        self.data['tests']['test_177']['samples'].append(x_test)
        self.data['tests']['test_177']['y_expected'].append(y_expected[0])
        self.data['tests']['test_177']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=19.6, max_value=30.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=550.0, max_value=624.9, allow_nan=False),
           st.floats(min_value=21.6, max_value=21.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_178(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_178']['n_samples'] += 1
        self.data['tests']['test_178']['samples'].append(x_test)
        self.data['tests']['test_178']['y_expected'].append(y_expected[0])
        self.data['tests']['test_178']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=19.6, max_value=30.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=350.0, max_value=374.9, allow_nan=False),
           st.floats(min_value=22.6, max_value=23.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_179(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_179']['n_samples'] += 1
        self.data['tests']['test_179']['samples'].append(x_test)
        self.data['tests']['test_179']['y_expected'].append(y_expected[0])
        self.data['tests']['test_179']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=19.6, max_value=30.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=375.1, max_value=425.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.6, max_value=23.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_180(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_180']['n_samples'] += 1
        self.data['tests']['test_180']['samples'].append(x_test)
        self.data['tests']['test_180']['y_expected'].append(y_expected[0])
        self.data['tests']['test_180']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=16.4, max_value=19.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_181(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_181']['n_samples'] += 1
        self.data['tests']['test_181']['samples'].append(x_test)
        self.data['tests']['test_181']['y_expected'].append(y_expected[0])
        self.data['tests']['test_181']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=20.1, max_value=20.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_182(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_182']['n_samples'] += 1
        self.data['tests']['test_182']['samples'].append(x_test)
        self.data['tests']['test_182']['y_expected'].append(y_expected[0])
        self.data['tests']['test_182']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=15.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.1, max_value=22.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_183(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_183']['n_samples'] += 1
        self.data['tests']['test_183']['samples'].append(x_test)
        self.data['tests']['test_183']['y_expected'].append(y_expected[0])
        self.data['tests']['test_183']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.6, max_value=29.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.floats(min_value=625.1, max_value=3000.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.6, max_value=26.4, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_184(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_184']['n_samples'] += 1
        self.data['tests']['test_184']['samples'].append(x_test)
        self.data['tests']['test_184']['y_expected'].append(y_expected[0])
        self.data['tests']['test_184']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=26.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.4, allow_nan=False),
           st.sampled_from([500, 750, 1750, 2000, 2500, 2750, 3250, 3500, 4750, 5500]),
           st.floats(min_value=26.6, max_value=40.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_185(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_185']['n_samples'] += 1
        self.data['tests']['test_185']['samples'].append(x_test)
        self.data['tests']['test_185']['y_expected'].append(y_expected[0])
        self.data['tests']['test_185']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.sampled_from([1000, 1250, 1500, 2750, 3000, 3500, 3750, 4250, 5500, 11000]),
           st.floats(min_value=27.6, max_value=33.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_186(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_186']['n_samples'] += 1
        self.data['tests']['test_186']['samples'].append(x_test)
        self.data['tests']['test_186']['y_expected'].append(y_expected[0])
        self.data['tests']['test_186']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=1150.0, max_value=1374.9, allow_nan=False),
           st.floats(min_value=34.1, max_value=36.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_187(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_187']['n_samples'] += 1
        self.data['tests']['test_187']['samples'].append(x_test)
        self.data['tests']['test_187']['y_expected'].append(y_expected[0])
        self.data['tests']['test_187']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=1375.1, max_value=3600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=34.1, max_value=34.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_188(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_188']['n_samples'] += 1
        self.data['tests']['test_188']['samples'].append(x_test)
        self.data['tests']['test_188']['y_expected'].append(y_expected[0])
        self.data['tests']['test_188']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=4.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=1375.1, max_value=3600.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=37.6, max_value=39.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_189(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_189']['n_samples'] += 1
        self.data['tests']['test_189']['samples'].append(x_test)
        self.data['tests']['test_189']['y_expected'].append(y_expected[0])
        self.data['tests']['test_189']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=6.6, max_value=6.7, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 2500, 2750, 3250, 3500, 4250, 5500, 5750, 6000, 11000]),
           st.floats(min_value=36.4, max_value=44.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_190(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_190']['n_samples'] += 1
        self.data['tests']['test_190']['samples'].append(x_test)
        self.data['tests']['test_190']['y_expected'].append(y_expected[0])
        self.data['tests']['test_190']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=5.1, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 1250, 1500, 1750, 2250, 3250, 3500, 4250, 4500, 4750]),
           st.floats(min_value=45.1, max_value=52.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_191(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_191']['n_samples'] += 1
        self.data['tests']['test_191']['samples'].append(x_test)
        self.data['tests']['test_191']['y_expected'].append(y_expected[0])
        self.data['tests']['test_191']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=5.1, exclude_min=True, allow_nan=False),
           st.sampled_from([2000, 2500, 2750, 3500, 4250, 4750, 5250, 5500, 6500, 11500]),
           st.floats(min_value=84.1, max_value=85.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_192(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_192']['n_samples'] += 1
        self.data['tests']['test_192']['samples'].append(x_test)
        self.data['tests']['test_192']['y_expected'].append(y_expected[0])
        self.data['tests']['test_192']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=5.1, exclude_min=True, allow_nan=False),
           st.sampled_from([750, 1000, 1500, 3250, 3750, 4250, 4500, 5500, 5750, 9500]),
           st.floats(min_value=90.1, max_value=91.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_193(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_193']['n_samples'] += 1
        self.data['tests']['test_193']['samples'].append(x_test)
        self.data['tests']['test_193']['y_expected'].append(y_expected[0])
        self.data['tests']['test_193']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=14.6, max_value=14.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.6, max_value=16.0, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 500, 750, 1000, 1250, 1500, 1750, 2500, 5500, 11000]),
           st.sampled_from([29, 41, 43, 46, 72, 75, 77, 82, 88, 95]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_194(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_194']['n_samples'] += 1
        self.data['tests']['test_194']['samples'].append(x_test)
        self.data['tests']['test_194']['y_expected'].append(y_expected[0])
        self.data['tests']['test_194']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=16.6, max_value=17.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=13.6, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 2250, 3000, 3250, 3500, 5500, 6500, 8500, 10250, 11500]),
           st.floats(min_value=66.0, max_value=81.9, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_195(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_195']['n_samples'] += 1
        self.data['tests']['test_195']['samples'].append(x_test)
        self.data['tests']['test_195']['y_expected'].append(y_expected[0])
        self.data['tests']['test_195']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=16.6, max_value=17.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=13.6, exclude_min=True, allow_nan=False),
           st.sampled_from([500, 750, 2000, 2500, 3250, 3500, 4000, 4250, 4500, 5750]),
           st.floats(min_value=82.1, max_value=85.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_196(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_196']['n_samples'] += 1
        self.data['tests']['test_196']['samples'].append(x_test)
        self.data['tests']['test_196']['y_expected'].append(y_expected[0])
        self.data['tests']['test_196']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=20.6, max_value=21.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=13.6, exclude_min=True, allow_nan=False),
           st.sampled_from([250, 750, 1000, 1500, 1750, 2500, 3500, 4500, 5500, 6000]),
           st.sampled_from([2, 9, 38, 49, 57, 60, 61, 65, 76, 98]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_197(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_197']['n_samples'] += 1
        self.data['tests']['test_197']['samples'].append(x_test)
        self.data['tests']['test_197']['y_expected'].append(y_expected[0])
        self.data['tests']['test_197']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=25.6, max_value=35.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=13.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1000, 2000, 2500, 2750, 3000, 3250, 3500, 4000, 5250, 10750]),
           st.sampled_from([3, 15, 40, 45, 50, 52, 53, 58, 69, 89]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_198(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_198']['n_samples'] += 1
        self.data['tests']['test_198']['samples'].append(x_test)
        self.data['tests']['test_198']['y_expected'].append(y_expected[0])
        self.data['tests']['test_198']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
