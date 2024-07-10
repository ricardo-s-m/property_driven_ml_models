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
    request.cls.data['n_test'] = 103
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_haberman_survival_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestHabermanSurvivalProperty:

    @given(st.floats(min_value=32.0, max_value=32.49, allow_nan=False),
           st.floats(min_value=59.2, max_value=59.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=32.51, max_value=32.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.2, max_value=59.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=33.6, max_value=34.49, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=34.51, max_value=35.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.0, max_value=65.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=36.8, max_value=38.49, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.51, max_value=38.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=38.4, max_value=40.49, allow_nan=False),
           st.floats(min_value=66.51, max_value=67.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=40.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=40.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=41.51, max_value=41.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8, max_value=0.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=41.51, max_value=41.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.01, max_value=1.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.51, max_value=43.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.2, max_value=59.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=46.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=59.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=46.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.51, max_value=61.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=40.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8, max_value=0.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.01, max_value=42.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8, max_value=0.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.01, max_value=44.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8, max_value=0.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.01, max_value=1.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=64.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.01, max_value=66.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.01, max_value=44.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.01, max_value=44.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.01, max_value=44.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.51, max_value=66.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=44.01, max_value=44.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.51, max_value=66.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.01, max_value=46.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.51, max_value=67.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.51, max_value=41.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=68.51, max_value=68.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=49.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=61.6, max_value=62.49, allow_nan=False),
           st.floats(min_value=1.2, max_value=1.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=48.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.2, max_value=59.49, allow_nan=False),
           st.floats(min_value=1.51, max_value=1.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.01, max_value=52.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.2, max_value=59.49, allow_nan=False),
           st.floats(min_value=1.51, max_value=1.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=49.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=1.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=47.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=63.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=48.51, max_value=48.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=62.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=48.51, max_value=48.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=50.51, max_value=52.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=63.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_36(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_36']['n_samples'] += 1
        self.data['tests']['test_36']['samples'].append(x_test)
        self.data['tests']['test_36']['y_expected'].append(y_expected[0])
        self.data['tests']['test_36']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=48.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=63.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_37(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_37']['n_samples'] += 1
        self.data['tests']['test_37']['samples'].append(x_test)
        self.data['tests']['test_37']['y_expected'].append(y_expected[0])
        self.data['tests']['test_37']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=51.51, max_value=52.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=63.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=58.01, max_value=58.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=63.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_39(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_39']['n_samples'] += 1
        self.data['tests']['test_39']['samples'].append(x_test)
        self.data['tests']['test_39']['y_expected'].append(y_expected[0])
        self.data['tests']['test_39']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=60.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=63.01, max_value=63.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_41(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_41']['n_samples'] += 1
        self.data['tests']['test_41']['samples'].append(x_test)
        self.data['tests']['test_41']['y_expected'].append(y_expected[0])
        self.data['tests']['test_41']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=65.51, max_value=66.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=61.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=68.01, max_value=68.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=71.01, max_value=72.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=61.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=71.01, max_value=71.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=71.01, max_value=71.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=63.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_49(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_49']['n_samples'] += 1
        self.data['tests']['test_49']['samples'].append(x_test)
        self.data['tests']['test_49']['y_expected'].append(y_expected[0])
        self.data['tests']['test_49']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=72.51, max_value=73.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=60.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.01, max_value=66.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=59.51, max_value=59.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=61.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=61.51, max_value=63.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=71.01, max_value=72.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=53.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=66.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4, max_value=0.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_57(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_57']['n_samples'] += 1
        self.data['tests']['test_57']['samples'].append(x_test)
        self.data['tests']['test_57']['y_expected'].append(y_expected[0])
        self.data['tests']['test_57']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=53.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=53.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.51, max_value=67.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.51, max_value=0.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=47.51, max_value=53.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.51, max_value=67.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.51, max_value=1.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.4, max_value=50.49, allow_nan=False),
           st.floats(min_value=66.4, max_value=68.49, allow_nan=False),
           st.floats(min_value=2.51, max_value=2.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.4, max_value=50.49, allow_nan=False),
           st.floats(min_value=68.51, max_value=68.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=2.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=50.51, max_value=51.1, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 59.0, 61.0, 62.0, 63.0, 64.0, 66.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=2.51, max_value=2.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_63(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_63']['n_samples'] += 1
        self.data['tests']['test_63']['samples'].append(x_test)
        self.data['tests']['test_63']['y_expected'].append(y_expected[0])
        self.data['tests']['test_63']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=50.51, max_value=51.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.8, max_value=58.99, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_64(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_64']['n_samples'] += 1
        self.data['tests']['test_64']['samples'].append(x_test)
        self.data['tests']['test_64']['y_expected'].append(y_expected[0])
        self.data['tests']['test_64']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=50.51, max_value=51.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.01, max_value=60.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_65(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_65']['n_samples'] += 1
        self.data['tests']['test_65']['samples'].append(x_test)
        self.data['tests']['test_65']['y_expected'].append(y_expected[0])
        self.data['tests']['test_65']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=50.51, max_value=51.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=65.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.6, max_value=64.99, allow_nan=False),
           st.floats(min_value=2.51, max_value=2.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.01, max_value=65.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=2.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_68(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_68']['n_samples'] += 1
        self.data['tests']['test_68']['samples'].append(x_test)
        self.data['tests']['test_68']['y_expected'].append(y_expected[0])
        self.data['tests']['test_68']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.51, max_value=67.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=2.9, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_69(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_69']['n_samples'] += 1
        self.data['tests']['test_69']['samples'].append(x_test)
        self.data['tests']['test_69']['y_expected'].append(y_expected[0])
        self.data['tests']['test_69']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.51, max_value=66.2, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 60.0, 61.0, 62.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=2.51, max_value=2.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=73.01, max_value=73.5, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 66.0, 67.0, 68.0]),
           st.floats(min_value=2.51, max_value=2.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_71(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_71']['n_samples'] += 1
        self.data['tests']['test_71']['samples'].append(x_test)
        self.data['tests']['test_71']['y_expected'].append(y_expected[0])
        self.data['tests']['test_71']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=75.51, max_value=75.9, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 59.0, 60.0, 62.0, 63.0, 64.0, 65.0, 66.0, 68.0, 69.0]),
           st.floats(min_value=2.51, max_value=2.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.51, max_value=67.1, exclude_min=True, allow_nan=False),
           st.sampled_from([59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=77.51, max_value=78.6, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 59.0, 60.0, 62.0, 63.0, 64.0, 65.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=3.6, max_value=4.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=33.6, max_value=34.49, allow_nan=False),
           st.floats(min_value=64.8, max_value=66.49, allow_nan=False),
           st.floats(min_value=4.51, max_value=5.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=33.6, max_value=34.49, allow_nan=False),
           st.floats(min_value=66.51, max_value=67.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=5.5, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_76(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_76']['n_samples'] += 1
        self.data['tests']['test_76']['samples'].append(x_test)
        self.data['tests']['test_76']['y_expected'].append(y_expected[0])
        self.data['tests']['test_76']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=33.6, max_value=34.49, allow_nan=False),
           st.sampled_from([58.0, 60.0, 61.0, 62.0, 63.0, 65.0, 66.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=9.51, max_value=11.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=34.51, max_value=36.1, exclude_min=True, allow_nan=False),
           st.sampled_from([58.0, 60.0, 61.0, 62.0, 63.0, 65.0, 66.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=4.51, max_value=7.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_78(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_78']['n_samples'] += 1
        self.data['tests']['test_78']['samples'].append(x_test)
        self.data['tests']['test_78']['y_expected'].append(y_expected[0])
        self.data['tests']['test_78']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.0, max_value=42.49, allow_nan=False),
           st.sampled_from([58.0, 59.0, 60.0, 62.0, 63.0, 64.0, 65.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=20.51, max_value=21.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_79(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_79']['n_samples'] += 1
        self.data['tests']['test_79']['samples'].append(x_test)
        self.data['tests']['test_79']['y_expected'].append(y_expected[0])
        self.data['tests']['test_79']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=40.0, max_value=42.49, allow_nan=False),
           st.sampled_from([58.0, 59.0, 60.0, 61.0, 62.0, 65.0, 66.0, 67.0, 68.0, 69.0]),
           st.floats(min_value=26.51, max_value=31.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.51, max_value=46.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.4, max_value=58.49, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.51, max_value=42.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=44.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=5.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=47.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=4.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.01, max_value=54.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=4.7, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=48.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=6.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=54.51, max_value=55.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=6.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=57.01, max_value=58.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=60.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=6.1, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_88(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_88']['n_samples'] += 1
        self.data['tests']['test_88']['samples'].append(x_test)
        self.data['tests']['test_88']['y_expected'].append(y_expected[0])
        self.data['tests']['test_88']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=46.51, max_value=49.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.01, max_value=66.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=5.3, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=44.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=48.01, max_value=49.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=58.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_91(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_91']['n_samples'] += 1
        self.data['tests']['test_91']['samples'].append(x_test)
        self.data['tests']['test_91']['y_expected'].append(y_expected[0])
        self.data['tests']['test_91']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=45.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=60.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=11.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=21.01, max_value=22.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=58.51, max_value=59.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=30.01, max_value=34.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_95(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_95']['n_samples'] += 1
        self.data['tests']['test_95']['samples'].append(x_test)
        self.data['tests']['test_95']['y_expected'].append(y_expected[0])
        self.data['tests']['test_95']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=53.51, max_value=55.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=62.51, max_value=62.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=47.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=64.51, max_value=64.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=47.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.51, max_value=65.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_98(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_98']['n_samples'] += 1
        self.data['tests']['test_98']['samples'].append(x_test)
        self.data['tests']['test_98']['y_expected'].append(y_expected[0])
        self.data['tests']['test_98']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=43.51, max_value=47.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=67.51, max_value=67.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=17.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=62.51, max_value=62.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.4, max_value=68.49, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=64.51, max_value=65.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.4, max_value=68.49, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=69.51, max_value=72.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=66.4, max_value=68.49, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=42.51, max_value=50.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=68.51, max_value=68.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=14.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3):
        x_test = [feature_1, feature_2, feature_3]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
