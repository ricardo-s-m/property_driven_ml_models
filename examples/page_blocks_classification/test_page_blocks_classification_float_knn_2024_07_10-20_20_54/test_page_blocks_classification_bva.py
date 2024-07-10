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
    request.cls.data['n_test'] = 165
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_page_blocks_classification_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestPageBlocksClassificationProperty:

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([13.0, 53.0, 72.0, 115.0, 126.0, 141.0, 291.0, 376.0, 419.0, 421.0]),
           st.sampled_from([94.0, 99.0, 294.0, 320.0, 848.0, 915.0, 2414.0, 5894.0, 6528.0, 11775.0]),
           st.sampled_from([0.357, 0.8, 2.071, 6.727, 12.0, 15.7, 18.556, 34.083, 38.778, 45.0]),
           st.floats(min_value=0.3703, max_value=0.4498, allow_nan=False),
           st.sampled_from([0.258, 0.346, 0.363, 0.558, 0.562, 0.615, 0.861, 0.95, 0.971, 0.973]),
           st.floats(min_value=1.284, max_value=1.354, allow_nan=False),
           st.floats(min_value=37.0, max_value=44.49, allow_nan=False),
           st.sampled_from([95.0, 205.0, 262.0, 569.0, 768.0, 811.0, 1114.0, 1483.0, 1857.0, 3262.0]),
           st.sampled_from([47.0, 72.0, 81.0, 82.0, 197.0, 234.0, 303.0, 307.0, 386.0, 403.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([24.0, 29.0, 45.0, 67.0, 84.0, 115.0, 123.0, 127.0, 176.0, 470.0]),
           st.sampled_from([51.0, 62.0, 79.0, 257.0, 398.0, 537.0, 858.0, 950.0, 1410.0, 12240.0]),
           st.sampled_from([12.5, 20.0, 28.875, 42.0, 57.0, 63.0, 180.0, 255.0, 283.0, 288.0]),
           st.floats(min_value=0.4501, max_value=0.56, exclude_min=True, allow_nan=False),
           st.sampled_from([0.353, 0.487, 0.536, 0.565, 0.629, 0.655, 0.684, 0.833, 0.875, 0.978]),
           st.floats(min_value=1.284, max_value=1.354, allow_nan=False),
           st.floats(min_value=37.0, max_value=44.49, allow_nan=False),
           st.sampled_from([22.0, 64.0, 81.0, 95.0, 124.0, 125.0, 159.0, 199.0, 790.0, 6035.0]),
           st.sampled_from([9.0, 10.0, 17.0, 30.0, 40.0, 45.0, 77.0, 120.0, 149.0, 272.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([16.0, 27.0, 32.0, 39.0, 63.0, 117.0, 132.0, 199.0, 213.0, 304.0]),
           st.sampled_from([23.0, 47.0, 48.0, 127.0, 152.0, 187.0, 379.0, 1665.0, 7500.0, 12240.0]),
           st.sampled_from([2.25, 3.0, 10.5, 28.0, 65.0, 69.0, 87.0, 166.67, 172.0, 179.33]),
           st.sampled_from([0.341, 0.356, 0.471, 0.533, 0.54, 0.616, 0.632, 0.778, 0.93, 0.975]),
           st.sampled_from([0.371, 0.512, 0.621, 0.667, 0.684, 0.69, 0.802, 0.816, 0.906, 0.92]),
           st.floats(min_value=1.284, max_value=1.354, allow_nan=False),
           st.floats(min_value=44.51, max_value=6639.0, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 46.0, 48.0, 114.0, 180.0, 186.0, 517.0, 571.0, 609.0, 735.0]),
           st.sampled_from([2.0, 4.0, 6.0, 8.0, 20.0, 24.0, 59.0, 176.0, 268.0, 272.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([36.0, 65.0, 66.0, 256.0, 280.0, 359.0, 416.0, 440.0, 455.0, 518.0]),
           st.sampled_from([592.0, 768.0, 819.0, 1099.0, 2420.0, 2988.0, 4184.0, 4480.0, 4650.0, 6784.0]),
           st.floats(min_value=2.1345, max_value=2.6663, allow_nan=False),
           st.sampled_from([0.134, 0.17, 0.179, 0.227, 0.264, 0.28, 0.301, 0.608, 0.894, 0.925]),
           st.sampled_from([0.396, 0.546, 0.57, 0.626, 0.695, 0.799, 0.867, 0.875, 0.974, 0.988]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 75.0, 78.0, 249.0, 292.0, 345.0, 543.0, 738.0, 1256.0, 2094.0]),
           st.sampled_from([277.0, 449.0, 517.0, 1164.0, 1219.0, 1501.0, 1672.0, 1702.0, 3176.0, 4481.0]),
           st.sampled_from([97.0, 100.0, 195.0, 377.0, 440.0, 447.0, 463.0, 473.0, 548.0, 559.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([28.0, 32.0, 51.0, 175.0, 209.0, 350.0, 356.0, 372.0, 414.0, 415.0]),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3059, max_value=0.3693, allow_nan=False),
           st.sampled_from([0.258, 0.285, 0.368, 0.484, 0.49, 0.495, 0.561, 0.618, 0.915, 0.947]),
           st.floats(min_value=1.357, max_value=2.985, exclude_min=True, allow_nan=False),
           st.sampled_from([83.0, 261.0, 461.0, 482.0, 589.0, 1035.0, 1235.0, 1260.0, 1317.0, 1579.0]),
           st.sampled_from([37.0, 110.0, 198.0, 207.0, 348.0, 500.0, 947.0, 1506.0, 2215.0, 3636.0]),
           st.sampled_from([3.0, 61.0, 169.0, 219.0, 274.0, 404.0, 417.0, 429.0, 575.0, 586.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([18.0, 43.0, 52.0, 62.0, 86.0, 94.0, 148.0, 176.0, 194.0, 278.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3059, max_value=0.3693, allow_nan=False),
           st.sampled_from([0.197, 0.34, 0.364, 0.616, 0.629, 0.686, 0.688, 0.875, 0.92, 1.0]),
           st.floats(min_value=1.357, max_value=2.985, exclude_min=True, allow_nan=False),
           st.sampled_from([21.0, 24.0, 27.0, 47.0, 54.0, 66.0, 178.0, 188.0, 263.0, 392.0]),
           st.sampled_from([16.0, 27.0, 37.0, 199.0, 206.0, 221.0, 587.0, 638.0, 641.0, 1264.0]),
           st.sampled_from([14.0, 17.0, 21.0, 37.0, 59.0, 176.0, 207.0, 268.0, 272.0, 579.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([5.0, 38.0, 68.0, 155.0, 163.0, 205.0, 240.0, 350.0, 371.0, 430.0]),
           st.sampled_from([246.0, 343.0, 464.0, 825.0, 1100.0, 1638.0, 2800.0, 3624.0, 4450.0, 4990.0]),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3696, max_value=0.4206, exclude_min=True, allow_nan=False),
           st.sampled_from([0.435, 0.461, 0.605, 0.669, 0.721, 0.783, 0.804, 0.841, 0.843, 0.926]),
           st.floats(min_value=1.357, max_value=2.985, exclude_min=True, allow_nan=False),
           st.sampled_from([14.0, 112.0, 313.0, 593.0, 732.0, 736.0, 1619.0, 1661.0, 2248.0, 4341.0]),
           st.sampled_from([226.0, 243.0, 282.0, 326.0, 569.0, 590.0, 616.0, 1577.0, 1865.0, 3433.0]),
           st.sampled_from([53.0, 95.0, 209.0, 216.0, 258.0, 273.0, 364.0, 463.0, 596.0, 643.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([1.0, 15.0, 37.0, 45.0, 78.0, 81.0, 108.0, 277.0, 475.0, 541.0]),
           st.sampled_from([9.0, 16.0, 26.0, 36.0, 46.0, 126.0, 159.0, 282.0, 1500.0, 3336.0]),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6251, max_value=0.7, exclude_min=True, allow_nan=False),
           st.sampled_from([0.254, 0.494, 0.555, 0.578, 0.681, 0.688, 0.689, 0.761, 0.882, 0.971]),
           st.floats(min_value=1.357, max_value=2.185, exclude_min=True, allow_nan=False),
           st.sampled_from([17.0, 33.0, 60.0, 109.0, 110.0, 125.0, 204.0, 365.0, 411.0, 1092.0]),
           st.sampled_from([12.0, 46.0, 48.0, 54.0, 71.0, 128.0, 288.0, 480.0, 614.0, 1264.0]),
           st.sampled_from([3.0, 7.0, 11.0, 13.0, 15.0, 16.0, 25.0, 28.0, 30.0, 272.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([13.0, 99.0, 115.0, 117.0, 158.0, 181.0, 207.0, 359.0, 498.0, 521.0]),
           st.sampled_from([88.0, 231.0, 1184.0, 1188.0, 1507.0, 1521.0, 1664.0, 2120.0, 3707.0, 4420.0]),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6251, max_value=0.7, exclude_min=True, allow_nan=False),
           st.sampled_from([0.285, 0.319, 0.338, 0.417, 0.566, 0.569, 0.572, 0.662, 0.774, 0.836]),
           st.floats(min_value=5.501, max_value=6.3, exclude_min=True, allow_nan=False),
           st.sampled_from([124.0, 328.0, 402.0, 454.0, 540.0, 623.0, 663.0, 845.0, 976.0, 1459.0]),
           st.sampled_from([10.0, 97.0, 399.0, 544.0, 1047.0, 1369.0, 2689.0, 2965.0, 3095.0, 9689.0]),
           st.sampled_from([72.0, 128.0, 202.0, 314.0, 323.0, 328.0, 349.0, 388.0, 473.0, 646.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([36.0, 48.0, 54.0, 94.0, 106.0, 127.0, 136.0, 138.0, 147.0, 277.0]),
           st.sampled_from([7.0, 62.0, 85.0, 138.0, 161.0, 277.0, 288.0, 314.0, 352.0, 4697.0]),
           st.floats(min_value=2.6666, max_value=3.6332, exclude_min=True, allow_nan=False),
           st.sampled_from([0.426, 0.439, 0.452, 0.518, 0.52, 0.811, 0.867, 0.92, 0.935, 0.984]),
           st.sampled_from([0.095, 0.197, 0.375, 0.406, 0.504, 0.554, 0.621, 0.85, 0.882, 0.971]),
           st.floats(min_value=9.501, max_value=998.6, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 16.0, 30.0, 57.0, 66.0, 392.0, 556.0, 576.0, 603.0, 2657.0]),
           st.sampled_from([25.0, 58.0, 75.0, 129.0, 146.0, 549.0, 662.0, 737.0, 760.0, 6035.0]),
           st.sampled_from([1.0, 7.0, 10.0, 15.0, 25.0, 26.0, 33.0, 120.0, 149.0, 272.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([26.0, 63.0, 91.0, 128.0, 239.0, 243.0, 273.0, 297.0, 393.0, 505.0]),
           st.sampled_from([468.0, 1440.0, 2466.0, 2618.0, 2862.0, 3556.0, 4225.0, 4872.0, 6112.0, 8016.0]),
           st.floats(min_value=7.5001, max_value=10.4667, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2007, max_value=0.2378, allow_nan=False),
           st.sampled_from([0.359, 0.526, 0.596, 0.609, 0.779, 0.819, 0.832, 0.881, 0.936, 0.99]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.sampled_from([85.0, 246.0, 250.0, 285.0, 288.0, 299.0, 350.0, 458.0, 816.0, 2736.0]),
           st.sampled_from([172.0, 290.0, 316.0, 320.0, 489.0, 710.0, 875.0, 1505.0, 2771.0, 2924.0]),
           st.sampled_from([64.0, 66.0, 212.0, 220.0, 349.0, 441.0, 452.0, 454.0, 460.0, 547.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([4.0, 11.0, 196.0, 201.0, 251.0, 306.0, 314.0, 393.0, 453.0, 464.0]),
           st.sampled_from([852.0, 855.0, 1304.0, 1665.0, 2004.0, 2384.0, 3178.0, 4464.0, 4576.0, 6640.0]),
           st.floats(min_value=22.3336, max_value=125.2668, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1439, max_value=0.1668, allow_nan=False),
           st.sampled_from([0.529, 0.558, 0.586, 0.718, 0.744, 0.792, 0.81, 0.814, 0.962, 0.983]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.sampled_from([33.0, 91.0, 411.0, 567.0, 607.0, 645.0, 684.0, 705.0, 731.0, 1490.0]),
           st.sampled_from([109.0, 284.0, 380.0, 444.0, 558.0, 732.0, 863.0, 1022.0, 1104.0, 1121.0]),
           st.sampled_from([20.0, 84.0, 260.0, 290.0, 343.0, 357.0, 404.0, 461.0, 555.0, 1218.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([9.0, 11.0, 14.0, 17.0, 21.0, 57.0, 64.0, 73.0, 114.0, 446.0]),
           st.sampled_from([37.0, 40.0, 47.0, 64.0, 76.0, 187.0, 196.0, 283.0, 950.0, 1608.0]),
           st.floats(min_value=22.3336, max_value=125.2668, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1671, max_value=0.1812, exclude_min=True, allow_nan=False),
           st.sampled_from([0.095, 0.173, 0.204, 0.34, 0.353, 0.556, 0.738, 0.786, 0.906, 0.978]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 26.0, 34.0, 49.0, 57.0, 122.0, 603.0, 628.0, 746.0, 907.0]),
           st.sampled_from([33.0, 37.0, 75.0, 85.0, 214.0, 325.0, 357.0, 735.0, 756.0, 4955.0]),
           st.sampled_from([1.0, 3.0, 5.0, 22.0, 23.0, 24.0, 36.0, 40.0, 149.0, 176.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([43.0, 49.0, 117.0, 147.0, 157.0, 159.0, 196.0, 470.0, 538.0, 544.0]),
           st.floats(min_value=8.2, max_value=8.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.273, 0.401, 0.421, 0.531, 0.536, 0.586, 0.6, 0.686, 0.821, 0.978]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.floats(min_value=9.8, max_value=10.49, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([45.0, 126.0, 138.0, 143.0, 167.0, 227.0, 318.0, 416.0, 525.0, 534.0]),
           st.floats(min_value=8.51, max_value=28805.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.224, 0.558, 0.602, 0.65, 0.664, 0.718, 0.863, 0.882, 0.928, 0.986]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.floats(min_value=9.8, max_value=10.49, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([94.0, 109.0, 133.0, 144.0, 176.0, 183.0, 257.0, 320.0, 398.0, 499.0]),
           st.sampled_from([32.0, 287.0, 721.0, 972.0, 1170.0, 2196.0, 3180.0, 3888.0, 4797.0, 12996.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.288, 0.552, 0.568, 0.578, 0.593, 0.617, 0.664, 0.708, 0.825, 0.962]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.floats(min_value=10.51, max_value=9235.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.2, max_value=2.49, allow_nan=False))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([158.0, 178.0, 190.0, 220.0, 249.0, 279.0, 295.0, 310.0, 364.0, 423.0]),
           st.floats(min_value=11.8, max_value=12.99, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.433, 0.462, 0.507, 0.56, 0.682, 0.866, 0.879, 0.913, 0.934, 0.956]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.sampled_from([251.0, 388.0, 402.0, 455.0, 700.0, 1479.0, 2732.0, 3357.0, 3435.0, 3863.0]),
           st.floats(min_value=2.51, max_value=644.4, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([29.0, 54.0, 65.0, 81.0, 85.0, 94.0, 193.0, 231.0, 325.0, 333.0]),
           st.floats(min_value=13.01, max_value=28809.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.304, 0.401, 0.494, 0.51, 0.578, 0.675, 0.681, 0.69, 0.816, 1.0]),
           st.floats(min_value=1.357, max_value=992.085, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.sampled_from([9.0, 31.0, 33.0, 39.0, 202.0, 257.0, 357.0, 596.0, 598.0, 703.0]),
           st.floats(min_value=2.51, max_value=644.4, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.sampled_from([17.0, 20.0, 33.0, 51.0, 75.0, 112.0, 132.0, 161.0, 184.0, 187.0]),
           st.sampled_from([20.0, 28.0, 46.0, 202.0, 283.0, 472.0, 537.0, 892.0, 1848.0, 4697.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.377, 0.406, 0.551, 0.57, 0.667, 0.764, 0.882, 0.897, 0.906, 0.927]),
           st.floats(min_value=1.357, max_value=1.585, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=8.7, exclude_min=True, allow_nan=False),
           st.sampled_from([87.0, 89.0, 131.0, 199.0, 213.0, 537.0, 627.0, 657.0, 685.0, 1264.0]),
           st.sampled_from([4.0, 6.0, 7.0, 11.0, 14.0, 17.0, 77.0, 268.0, 272.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.sampled_from([2.0, 39.0, 81.0, 89.0, 212.0, 234.0, 246.0, 294.0, 319.0, 386.0]),
           st.sampled_from([100.0, 238.0, 912.0, 924.0, 1098.0, 1210.0, 1236.0, 1256.0, 2752.0, 5872.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.438, 0.465, 0.537, 0.573, 0.64, 0.643, 0.724, 0.771, 0.813, 0.902]),
           st.floats(min_value=1.357, max_value=1.585, exclude_min=True, allow_nan=False),
           st.floats(min_value=13.51, max_value=13.7, exclude_min=True, allow_nan=False),
           st.sampled_from([261.0, 347.0, 631.0, 679.0, 1054.0, 1147.0, 1659.0, 1802.0, 3554.0, 5944.0]),
           st.sampled_from([41.0, 83.0, 284.0, 318.0, 432.0, 476.0, 485.0, 612.0, 624.0, 1218.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.sampled_from([9.0, 10.0, 27.0, 65.0, 75.0, 79.0, 104.0, 288.0, 325.0, 342.0]),
           st.sampled_from([17.0, 52.0, 78.0, 172.0, 193.0, 234.0, 830.0, 912.0, 1076.0, 1848.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.481, 0.529, 0.533, 0.565, 0.621, 0.689, 0.695, 0.786, 0.864, 0.998]),
           st.floats(min_value=1.357, max_value=1.585, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.51, max_value=6615.0, exclude_min=True, allow_nan=False),
           st.sampled_from([33.0, 48.0, 86.0, 101.0, 125.0, 128.0, 288.0, 349.0, 4890.0, 5790.0]),
           st.sampled_from([3.0, 6.0, 8.0, 17.0, 21.0, 26.0, 30.0, 36.0, 45.0, 268.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=25.8, max_value=31.99, allow_nan=False),
           st.sampled_from([9.0, 174.0, 700.0, 1728.0, 2304.0, 2613.0, 2840.0, 3330.0, 3976.0, 4537.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.401, 0.449, 0.638, 0.718, 0.748, 0.779, 0.804, 0.821, 0.847, 0.967]),
           st.floats(min_value=2.501, max_value=2.572, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=6609.4, exclude_min=True, allow_nan=False),
           st.sampled_from([35.0, 332.0, 510.0, 688.0, 1570.0, 1575.0, 1892.0, 2044.0, 2979.0, 6963.0]),
           st.sampled_from([96.0, 117.0, 121.0, 158.0, 231.0, 305.0, 441.0, 537.0, 621.0, 651.0]))
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

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.floats(min_value=32.01, max_value=136.2, exclude_min=True, allow_nan=False),
           st.sampled_from([45.0, 54.0, 73.0, 127.0, 147.0, 193.0, 199.0, 278.0, 1614.0, 5814.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.34, 0.353, 0.377, 0.387, 0.51, 0.577, 0.596, 0.703, 0.85, 0.882]),
           st.floats(min_value=2.501, max_value=2.572, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=6609.4, exclude_min=True, allow_nan=False),
           st.sampled_from([81.0, 89.0, 104.0, 129.0, 206.0, 213.0, 517.0, 596.0, 673.0, 4000.0]),
           st.sampled_from([10.0, 22.0, 23.0, 28.0, 29.0, 33.0, 40.0, 45.0, 207.0, 268.0]))
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

    @given(st.floats(min_value=1.51, max_value=1.9, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 52.0, 67.0, 70.0, 104.0, 126.0, 128.0, 142.0, 176.0, 286.0]),
           st.sampled_from([16.0, 40.0, 46.0, 96.0, 127.0, 196.0, 206.0, 413.0, 912.0, 1665.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.378, 0.512, 0.533, 0.614, 0.675, 0.677, 0.743, 0.747, 0.752, 0.92]),
           st.floats(min_value=1.357, max_value=1.657, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=7.9, exclude_min=True, allow_nan=False),
           st.sampled_from([26.0, 43.0, 46.0, 102.0, 127.0, 150.0, 242.0, 413.0, 790.0, 11341.0]),
           st.sampled_from([6.0, 21.0, 22.0, 25.0, 26.0, 33.0, 40.0, 77.0, 120.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=1.9, exclude_min=True, allow_nan=False),
           st.sampled_from([19.0, 36.0, 47.0, 117.0, 121.0, 148.0, 334.0, 342.0, 383.0, 424.0]),
           st.sampled_from([91.0, 144.0, 177.0, 340.0, 344.0, 595.0, 1310.0, 1491.0, 2520.0, 2622.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.3904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.323, 0.393, 0.48, 0.502, 0.531, 0.588, 0.693, 0.697, 0.735, 0.843]),
           st.floats(min_value=1.357, max_value=1.657, exclude_min=True, allow_nan=False),
           st.floats(min_value=9.51, max_value=6611.0, exclude_min=True, allow_nan=False),
           st.sampled_from([168.0, 356.0, 366.0, 513.0, 874.0, 1277.0, 1738.0, 1898.0, 1975.0, 10849.0]),
           st.sampled_from([10.0, 147.0, 162.0, 238.0, 262.0, 263.0, 300.0, 533.0, 537.0, 546.0]))
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

    @given(st.floats(min_value=1.4, max_value=1.49, allow_nan=False),
           st.sampled_from([38.0, 129.0, 155.0, 178.0, 220.0, 230.0, 239.0, 315.0, 347.0, 389.0]),
           st.sampled_from([368.0, 756.0, 945.0, 1337.0, 1712.0, 1925.0, 1960.0, 5208.0, 5350.0, 11304.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.2486, exclude_min=True, allow_nan=False),
           st.sampled_from([0.502, 0.601, 0.661, 0.672, 0.74, 0.851, 0.868, 0.873, 0.888, 0.903]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=6609.4, exclude_min=True, allow_nan=False),
           st.sampled_from([399.0, 460.0, 486.0, 791.0, 1882.0, 1981.0, 2090.0, 4219.0, 6963.0, 7519.0]),
           st.sampled_from([35.0, 41.0, 93.0, 100.0, 135.0, 141.0, 257.0, 323.0, 389.0, 492.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=1.51, max_value=1.9, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 37.0, 48.0, 57.0, 94.0, 106.0, 145.0, 206.0, 235.0, 255.0]),
           st.sampled_from([26.0, 37.0, 44.0, 76.0, 104.0, 187.0, 199.0, 1070.0, 1614.0, 1623.0]),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2381, max_value=0.2486, exclude_min=True, allow_nan=False),
           st.sampled_from([0.427, 0.596, 0.603, 0.605, 0.609, 0.614, 0.681, 0.722, 0.764, 0.897]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=6609.4, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 30.0, 38.0, 115.0, 124.0, 146.0, 184.0, 212.0, 802.0, 11341.0]),
           st.sampled_from([2.0, 7.0, 15.0, 18.0, 21.0, 24.0, 25.0, 28.0, 45.0, 268.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=19.4, max_value=23.99, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.3848, exclude_min=True, allow_nan=False),
           st.sampled_from([0.273, 0.413, 0.577, 0.586, 0.621, 0.655, 0.69, 0.695, 0.738, 0.996]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=8.1, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 17.0, 28.0, 36.0, 41.0, 42.0, 71.0, 85.0, 184.0, 5790.0]),
           st.sampled_from([7.0, 10.0, 24.0, 25.0, 27.0, 30.0, 33.0, 40.0, 59.0, 268.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=19.4, max_value=23.99, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7596, max_value=0.7765, exclude_min=True, allow_nan=False),
           st.sampled_from([0.286, 0.397, 0.429, 0.533, 0.668, 0.68, 0.734, 0.742, 0.76, 0.773]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([63.0, 94.0, 126.0, 195.0, 218.0, 288.0, 383.0, 392.0, 425.0, 525.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=19.4, max_value=23.99, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7596, max_value=0.7765, exclude_min=True, allow_nan=False),
           st.sampled_from([0.546, 0.571, 0.574, 0.649, 0.664, 0.669, 0.752, 0.799, 0.84, 0.998]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=8.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=9235.8, exclude_min=True, allow_nan=False),
           st.sampled_from([3.0, 134.0, 149.0, 152.0, 213.0, 222.0, 239.0, 289.0, 345.0, 1651.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=19.4, max_value=23.99, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8447, max_value=0.8757, exclude_min=True, allow_nan=False),
           st.sampled_from([0.144, 0.202, 0.554, 0.578, 0.596, 0.675, 0.722, 0.738, 0.75, 0.816]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 19.0, 39.0, 52.0, 71.0, 213.0, 489.0, 609.0, 641.0, 11341.0]),
           st.sampled_from([5.0, 7.0, 8.0, 24.0, 28.0, 29.0, 36.0, 37.0, 59.0, 268.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=7.8, max_value=9.49, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8447, max_value=0.8757, exclude_min=True, allow_nan=False),
           st.sampled_from([0.317, 0.344, 0.521, 0.607, 0.617, 0.716, 0.795, 0.832, 0.835, 0.897]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=8.9, exclude_min=True, allow_nan=False),
           st.sampled_from([157.0, 172.0, 228.0, 715.0, 1013.0, 1068.0, 1238.0, 1887.0, 2687.0, 3022.0]),
           st.sampled_from([68.0, 130.0, 152.0, 287.0, 308.0, 341.0, 407.0, 521.0, 643.0, 773.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=9.51, max_value=12.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8447, max_value=0.8757, exclude_min=True, allow_nan=False),
           st.sampled_from([0.353, 0.364, 0.387, 0.413, 0.586, 0.677, 0.689, 0.738, 0.882, 0.906]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=8.9, exclude_min=True, allow_nan=False),
           st.sampled_from([65.0, 69.0, 102.0, 125.0, 128.0, 136.0, 146.0, 413.0, 657.0, 703.0]),
           st.sampled_from([5.0, 7.0, 10.0, 17.0, 21.0, 24.0, 27.0, 28.0, 149.0, 207.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=19.4, max_value=23.99, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.197, 0.34, 0.375, 0.405, 0.529, 0.536, 0.556, 0.578, 0.677, 0.689]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.51, max_value=6611.8, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 38.0, 101.0, 108.0, 128.0, 193.0, 213.0, 283.0, 288.0, 704.0]),
           st.sampled_from([4.0, 7.0, 14.0, 19.0, 20.0, 23.0, 36.0, 65.0, 149.0, 176.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=24.01, max_value=129.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.3922, exclude_min=True, allow_nan=False),
           st.sampled_from([0.465, 0.49, 0.496, 0.515, 0.541, 0.565, 0.589, 0.674, 0.83, 0.958]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=11.1, exclude_min=True, allow_nan=False),
           st.sampled_from([60.0, 476.0, 1024.0, 1597.0, 1621.0, 1932.0, 1995.0, 2129.0, 2508.0, 4017.0]),
           st.sampled_from([7.0, 30.0, 147.0, 185.0, 186.0, 276.0, 285.0, 323.0, 359.0, 373.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=24.01, max_value=129.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7966, max_value=0.8372, exclude_min=True, allow_nan=False),
           st.sampled_from([0.395, 0.48, 0.502, 0.562, 0.643, 0.681, 0.758, 0.844, 0.849, 0.897]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=11.1, exclude_min=True, allow_nan=False),
           st.sampled_from([330.0, 691.0, 723.0, 762.0, 839.0, 1102.0, 1654.0, 1807.0, 3150.0, 4316.0]),
           st.sampled_from([51.0, 54.0, 84.0, 87.0, 160.0, 162.0, 284.0, 465.0, 499.0, 586.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.floats(min_value=24.01, max_value=129.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=23.4, max_value=27.49, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.378, 0.57, 0.603, 0.653, 0.689, 0.738, 0.821, 0.85, 0.906, 0.998]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=25.51, max_value=6623.8, exclude_min=True, allow_nan=False),
           st.sampled_from([17.0, 35.0, 62.0, 87.0, 214.0, 283.0, 306.0, 596.0, 667.0, 688.0]),
           st.sampled_from([1.0, 4.0, 9.0, 14.0, 17.0, 19.0, 20.0, 26.0, 65.0, 268.0]))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([2.0, 34.0, 37.0, 48.0, 65.0, 106.0, 127.0, 157.0, 204.0, 236.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=13.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.144, 0.197, 0.405, 0.519, 0.536, 0.558, 0.596, 0.605, 0.764, 0.92]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=141.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=155.4, max_value=192.49, allow_nan=False),
           st.sampled_from([2.0, 3.0, 5.0, 18.0, 21.0, 26.0, 30.0, 65.0, 149.0, 176.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_38(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_38']['n_samples'] += 1
        self.data['tests']['test_38']['samples'].append(x_test)
        self.data['tests']['test_38']['y_expected'].append(y_expected[0])
        self.data['tests']['test_38']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([9.0, 33.0, 47.0, 87.0, 106.0, 136.0, 145.0, 427.0, 475.0, 544.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.5001, max_value=39.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.066, 0.405, 0.529, 0.544, 0.551, 0.614, 0.681, 0.689, 0.752, 0.802]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=141.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=155.4, max_value=192.49, allow_nan=False),
           st.floats(min_value=1.8, max_value=1.99, allow_nan=False))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([23.0, 47.0, 63.0, 100.0, 108.0, 110.0, 131.0, 213.0, 231.0, 536.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.5001, max_value=39.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.421, 0.534, 0.621, 0.738, 0.747, 0.786, 0.85, 0.971, 0.978, 0.998]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=13.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=155.4, max_value=192.49, allow_nan=False),
           st.floats(min_value=2.01, max_value=644.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_40(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_40']['n_samples'] += 1
        self.data['tests']['test_40']['samples'].append(x_test)
        self.data['tests']['test_40']['y_expected'].append(y_expected[0])
        self.data['tests']['test_40']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([53.0, 75.0, 76.0, 89.0, 102.0, 121.0, 132.0, 323.0, 350.0, 518.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=39.5001, max_value=39.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.455, 0.482, 0.484, 0.517, 0.535, 0.627, 0.666, 0.752, 0.904, 0.928]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=35.51, max_value=163.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=155.4, max_value=192.49, allow_nan=False),
           st.floats(min_value=2.01, max_value=644.0, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([35.0, 78.0, 157.0, 163.0, 229.0, 230.0, 246.0, 340.0, 342.0, 498.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=14.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.41, 0.412, 0.501, 0.585, 0.614, 0.654, 0.682, 0.702, 0.74, 0.757]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=141.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=192.51, max_value=9380.6, exclude_min=True, allow_nan=False),
           st.sampled_from([43.0, 59.0, 79.0, 93.0, 299.0, 393.0, 410.0, 514.0, 542.0, 646.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_42(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_42']['n_samples'] += 1
        self.data['tests']['test_42']['samples'].append(x_test)
        self.data['tests']['test_42']['y_expected'].append(y_expected[0])
        self.data['tests']['test_42']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([19.0, 26.0, 42.0, 52.0, 104.0, 172.0, 186.0, 206.0, 286.0, 342.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=40.5001, max_value=139.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.173, 0.375, 0.494, 0.536, 0.594, 0.684, 0.816, 0.92, 0.971, 0.996]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=141.0, exclude_min=True, allow_nan=False),
           st.sampled_from([47.0, 114.0, 379.0, 587.0, 596.0, 626.0, 652.0, 685.0, 688.0, 737.0]),
           st.sampled_from([4.0, 5.0, 7.0, 22.0, 25.0, 27.0, 45.0, 65.0, 207.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_43(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_43']['n_samples'] += 1
        self.data['tests']['test_43']['samples'].append(x_test)
        self.data['tests']['test_43']['y_expected'].append(y_expected[0])
        self.data['tests']['test_43']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([42.0, 77.0, 130.0, 146.0, 172.0, 199.0, 323.0, 358.0, 377.0, 412.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.245, 0.439, 0.589, 0.604, 0.614, 0.681, 0.689, 0.872, 0.899, 0.917]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=675.01, max_value=676.8, exclude_min=True, allow_nan=False),
           st.sampled_from([198.0, 268.0, 734.0, 1412.0, 2130.0, 2437.0, 2681.0, 2875.0, 3363.0, 5649.0]),
           st.sampled_from([19.0, 185.0, 220.0, 323.0, 361.0, 386.0, 387.0, 463.0, 586.0, 650.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_44(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_44']['n_samples'] += 1
        self.data['tests']['test_44']['samples'].append(x_test)
        self.data['tests']['test_44']['y_expected'].append(y_expected[0])
        self.data['tests']['test_44']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([23.0, 29.0, 39.0, 43.0, 54.0, 64.0, 94.0, 231.0, 255.0, 550.0]),
           st.floats(min_value=27.51, max_value=28820.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.5001, max_value=113.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2912, max_value=0.4329, exclude_min=True, allow_nan=False),
           st.sampled_from([0.353, 0.377, 0.387, 0.406, 0.554, 0.578, 0.594, 0.611, 0.833, 0.947]),
           st.floats(min_value=2.862, max_value=993.289, exclude_min=True, allow_nan=False),
           st.floats(min_value=684.01, max_value=7150.6, exclude_min=True, allow_nan=False),
           st.sampled_from([17.0, 48.0, 50.0, 55.0, 128.0, 242.0, 349.0, 413.0, 626.0, 652.0]),
           st.sampled_from([15.0, 18.0, 19.0, 20.0, 29.0, 40.0, 120.0, 207.0, 272.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_45(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_45']['n_samples'] += 1
        self.data['tests']['test_45']['samples'].append(x_test)
        self.data['tests']['test_45']['y_expected'].append(y_expected[0])
        self.data['tests']['test_45']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([54.0, 85.0, 123.0, 127.0, 161.0, 180.0, 184.0, 206.0, 235.0, 286.0]),
           st.floats(min_value=31.4, max_value=37.49, allow_nan=False),
           st.floats(min_value=0.0237, max_value=0.0278, allow_nan=False),
           st.sampled_from([0.2, 0.253, 0.403, 0.51, 0.621, 0.692, 0.846, 0.858, 0.869, 0.909]),
           st.sampled_from([0.173, 0.512, 0.555, 0.556, 0.6, 0.614, 0.686, 0.802, 0.927, 0.996]),
           st.sampled_from([3.75, 6.31, 7.5, 12.5, 25.0, 29.32, 30.27, 30.67, 73.0, 123.0]),
           st.sampled_from([13.0, 20.0, 30.0, 63.0, 107.0, 114.0, 122.0, 127.0, 142.0, 177.0]),
           st.sampled_from([13.0, 27.0, 32.0, 67.0, 71.0, 114.0, 278.0, 1630.0, 2676.0, 5522.0]),
           st.sampled_from([1.0, 3.0, 9.0, 10.0, 19.0, 22.0, 28.0, 29.0, 207.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_46(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_46']['n_samples'] += 1
        self.data['tests']['test_46']['samples'].append(x_test)
        self.data['tests']['test_46']['y_expected'].append(y_expected[0])
        self.data['tests']['test_46']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.floats(min_value=37.51, max_value=28828.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0237, max_value=0.0278, allow_nan=False),
           st.sampled_from([0.516, 0.536, 0.598, 0.611, 0.643, 0.7, 0.706, 0.8, 0.816, 0.993]),
           st.sampled_from([0.09, 0.233, 0.486, 0.533, 0.536, 0.636, 0.7, 0.706, 0.841, 0.875]),
           st.sampled_from([1.67, 2.08, 10.0, 14.0, 23.0, 27.0, 31.0, 47.0, 59.0, 135.0]),
           st.sampled_from([9.0, 27.0, 31.0, 39.0, 41.0, 42.0, 47.0, 53.0, 84.0, 133.0]),
           st.sampled_from([8.0, 9.0, 11.0, 22.0, 23.0, 32.0, 34.0, 37.0, 84.0, 210.0]),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 32.0, 38.0, 50.0, 54.0, 64.0, 108.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_47(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_47']['n_samples'] += 1
        self.data['tests']['test_47']['samples'].append(x_test)
        self.data['tests']['test_47']['y_expected'].append(y_expected[0])
        self.data['tests']['test_47']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([8.0, 18.0, 22.0, 25.0, 26.0, 34.0, 53.0, 117.0, 174.0, 372.0]),
           st.floats(min_value=0.0281, max_value=0.041, exclude_min=True, allow_nan=False),
           st.sampled_from([0.282, 0.419, 0.611, 0.625, 0.636, 0.706, 0.816, 0.9, 0.993, 1.0]),
           st.sampled_from([0.444, 0.536, 0.667, 0.7, 0.722, 0.75, 0.8, 0.909, 0.952, 1.0]),
           st.sampled_from([2.08, 2.57, 10.0, 15.64, 24.0, 31.0, 42.0, 54.0, 97.0, 135.0]),
           st.sampled_from([10.0, 11.0, 24.0, 26.0, 36.0, 37.0, 76.0, 111.0, 117.0, 1689.0]),
           st.sampled_from([9.0, 17.0, 24.0, 32.0, 42.0, 81.0, 111.0, 112.0, 117.0, 136.0]),
           st.sampled_from([1.0, 10.0, 14.0, 16.0, 32.0, 38.0, 50.0, 54.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_48(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_48']['n_samples'] += 1
        self.data['tests']['test_48']['samples'].append(x_test)
        self.data['tests']['test_48']['y_expected'].append(y_expected[0])
        self.data['tests']['test_48']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 21.0, 24.0, 36.0, 69.0, 117.0, 128.0, 161.0, 379.0, 510.0]),
           st.sampled_from([15.0, 26.0, 47.0, 126.0, 172.0, 180.0, 288.0, 460.0, 639.0, 852.0]),
           st.floats(min_value=0.0931, max_value=0.0939, exclude_min=True, allow_nan=False),
           st.sampled_from([0.31, 0.36, 0.571, 0.588, 0.719, 0.813, 0.822, 0.826, 0.909, 0.945]),
           st.sampled_from([0.197, 0.424, 0.505, 0.6, 0.622, 0.75, 0.764, 0.826, 0.927, 0.978]),
           st.sampled_from([2.33, 10.27, 25.71, 30.25, 38.0, 51.0, 55.5, 56.0, 60.83, 67.82]),
           st.sampled_from([18.0, 29.0, 41.0, 51.0, 139.0, 142.0, 365.0, 411.0, 506.0, 534.0]),
           st.sampled_from([22.0, 47.0, 89.0, 106.0, 186.0, 366.0, 379.0, 598.0, 1213.0, 5522.0]),
           st.sampled_from([8.0, 9.0, 12.0, 13.0, 15.0, 24.0, 33.0, 36.0, 268.0, 272.0]))
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

    @given(st.floats(min_value=3.51, max_value=4.3, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([10.0, 14.0, 22.0, 37.0, 42.0, 47.0, 108.0, 117.0, 188.0, 210.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.063, 0.191, 0.216, 0.333, 0.38, 0.625, 0.657, 0.7, 0.875, 0.9]),
           st.sampled_from([0.533, 0.636, 0.667, 0.706, 0.714, 0.8, 0.816, 0.875, 0.909, 1.0]),
           st.sampled_from([2.08, 2.56, 15.0, 15.64, 17.0, 18.0, 23.0, 46.5, 135.0, 141.0]),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.sampled_from([17.0, 26.0, 54.0, 60.0, 72.0, 81.0, 111.0, 143.0, 210.0, 2058.0]),
           st.sampled_from([10.0, 14.0, 15.0, 16.0, 32.0, 50.0, 54.0, 64.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_50(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_50']['n_samples'] += 1
        self.data['tests']['test_50']['samples'].append(x_test)
        self.data['tests']['test_50']['y_expected'].append(y_expected[0])
        self.data['tests']['test_50']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=166.8, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([9.0, 10.0, 12.0, 26.0, 34.0, 39.0, 42.0, 108.0, 294.0, 7830.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.063, 0.339, 0.419, 0.536, 0.625, 0.636, 0.643, 0.7, 0.875, 1.0]),
           st.sampled_from([0.233, 0.533, 0.7, 0.722, 0.75, 0.8, 0.841, 0.875, 0.952, 1.0]),
           st.sampled_from([2.47, 7.0, 9.0, 14.0, 15.64, 18.0, 25.0, 27.0, 32.0, 59.0]),
           st.floats(min_value=7.4, max_value=7.49, allow_nan=False),
           st.sampled_from([17.0, 18.0, 22.0, 27.0, 47.0, 53.0, 54.0, 112.0, 117.0, 197.0]),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 16.0, 32.0, 38.0, 50.0, 54.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_51(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_51']['n_samples'] += 1
        self.data['tests']['test_51']['samples'].append(x_test)
        self.data['tests']['test_51']['y_expected'].append(y_expected[0])
        self.data['tests']['test_51']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=4.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([8.0, 9.0, 14.0, 23.0, 31.0, 34.0, 38.0, 108.0, 188.0, 2150.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.198, 0.216, 0.333, 0.38, 0.419, 0.516, 0.529, 0.625, 0.7, 0.706]),
           st.sampled_from([0.09, 0.221, 0.345, 0.432, 0.714, 0.75, 0.8, 0.875, 0.952, 1.0]),
           st.sampled_from([2.89, 9.0, 24.0, 26.0, 39.0, 46.5, 54.0, 59.0, 84.0, 141.0]),
           st.floats(min_value=7.51, max_value=26.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.2, max_value=8.49, allow_nan=False),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 16.0, 32.0, 38.0, 64.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_52(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_52']['n_samples'] += 1
        self.data['tests']['test_52']['samples'].append(x_test)
        self.data['tests']['test_52']['y_expected'].append(y_expected[0])
        self.data['tests']['test_52']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=4.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([7.0, 13.0, 26.0, 27.0, 31.0, 37.0, 42.0, 117.0, 126.0, 198.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.191, 0.198, 0.282, 0.339, 0.598, 0.611, 0.636, 0.657, 0.816, 0.9]),
           st.sampled_from([0.09, 0.233, 0.345, 0.533, 0.565, 0.75, 0.8, 0.816, 0.841, 0.875]),
           st.sampled_from([2.57, 2.6, 9.0, 12.0, 13.0, 27.0, 32.0, 37.0, 39.0, 84.0]),
           st.floats(min_value=7.51, max_value=26.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.51, max_value=9233.4, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 16.0, 32.0, 38.0, 54.0, 64.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_53(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_53']['n_samples'] += 1
        self.data['tests']['test_53']['samples'].append(x_test)
        self.data['tests']['test_53']['y_expected'].append(y_expected[0])
        self.data['tests']['test_53']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=9.51, max_value=168.4, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([16.0, 20.0, 22.0, 25.0, 26.0, 28.0, 42.0, 54.0, 117.0, 136.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.191, 0.198, 0.282, 0.333, 0.38, 0.529, 0.611, 0.706, 0.8, 0.993]),
           st.sampled_from([0.432, 0.444, 0.516, 0.706, 0.714, 0.722, 0.816, 0.875, 0.9, 0.952]),
           st.sampled_from([2.4, 7.0, 13.0, 15.64, 37.0, 42.0, 46.5, 84.0, 117.0, 141.0]),
           st.floats(min_value=7.51, max_value=26.1, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 12.0, 22.0, 56.0, 72.0, 81.0, 97.0, 112.0, 210.0, 475.0]),
           st.sampled_from([1.0, 14.0, 15.0, 16.0, 32.0, 38.0, 50.0, 54.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_54(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_54']['n_samples'] += 1
        self.data['tests']['test_54']['samples'].append(x_test)
        self.data['tests']['test_54']['y_expected'].append(y_expected[0])
        self.data['tests']['test_54']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.2, max_value=17.49, allow_nan=False),
           st.sampled_from([189.0, 304.0, 760.0, 913.0, 1220.0, 1232.0, 1472.0, 2992.0, 3024.0, 5317.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.119, 0.195, 0.235, 0.244, 0.317, 0.401, 0.431, 0.499, 0.601, 0.661]),
           st.sampled_from([0.435, 0.536, 0.621, 0.702, 0.715, 0.775, 0.78, 0.79, 0.922, 0.926]),
           st.sampled_from([1.34, 1.63, 3.96, 4.32, 5.23, 5.46, 5.93, 7.24, 7.42, 7.63]),
           st.floats(min_value=100.51, max_value=6683.8, exclude_min=True, allow_nan=False),
           st.sampled_from([35.0, 391.0, 551.0, 753.0, 1113.0, 2171.0, 3085.0, 3276.0, 3318.0, 5488.0]),
           st.sampled_from([89.0, 98.0, 180.0, 198.0, 309.0, 344.0, 429.0, 439.0, 469.0, 664.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_55(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_55']['n_samples'] += 1
        self.data['tests']['test_55']['samples'].append(x_test)
        self.data['tests']['test_55']['y_expected'].append(y_expected[0])
        self.data['tests']['test_55']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=17.51, max_value=124.6, exclude_min=True, allow_nan=False),
           st.sampled_from([14.0, 17.0, 22.0, 23.0, 32.0, 42.0, 132.0, 294.0, 2150.0, 7830.0]),
           st.floats(min_value=0.0977, max_value=0.1153, exclude_min=True, allow_nan=False),
           st.sampled_from([0.191, 0.282, 0.339, 0.48, 0.598, 0.625, 0.706, 0.9, 0.993, 1.0]),
           st.sampled_from([0.221, 0.345, 0.533, 0.706, 0.714, 0.722, 0.75, 0.8, 0.816, 0.909]),
           st.sampled_from([2.4, 9.0, 15.0, 18.0, 24.0, 34.0, 37.0, 59.0, 76.0, 97.0]),
           st.floats(min_value=100.51, max_value=6683.8, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 22.0, 34.0, 39.0, 60.0, 76.0, 117.0, 127.0, 180.0, 701.0]),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 16.0, 32.0, 50.0, 64.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_56(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_56']['n_samples'] += 1
        self.data['tests']['test_56']['samples'].append(x_test)
        self.data['tests']['test_56']['y_expected'].append(y_expected[0])
        self.data['tests']['test_56']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([33.0, 42.0, 46.0, 57.0, 106.0, 252.0, 286.0, 541.0, 550.0, 553.0]),
           st.sampled_from([30.0, 41.0, 61.0, 177.0, 202.0, 224.0, 830.0, 5814.0, 6156.0, 12240.0]),
           st.floats(min_value=0.1861, max_value=0.196, exclude_min=True, allow_nan=False),
           st.sampled_from([0.102, 0.231, 0.373, 0.537, 0.579, 0.616, 0.907, 0.939, 0.984, 0.994]),
           st.sampled_from([0.202, 0.34, 0.472, 0.558, 0.605, 0.609, 0.653, 0.688, 0.694, 0.786]),
           st.sampled_from([1.07, 9.75, 9.8, 10.75, 17.88, 18.0, 26.57, 32.11, 88.62, 117.8]),
           st.sampled_from([15.0, 25.0, 41.0, 82.0, 114.0, 125.0, 315.0, 532.0, 557.0, 4409.0]),
           st.floats(min_value=11.4, max_value=12.49, allow_nan=False),
           st.sampled_from([6.0, 8.0, 9.0, 12.0, 18.0, 23.0, 25.0, 29.0, 40.0, 268.0]))
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

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 3.0, 18.0, 30.0, 61.0, 69.0, 71.0, 198.0, 286.0, 402.0]),
           st.sampled_from([60.0, 84.0, 91.0, 168.0, 740.0, 3871.0, 4900.0, 13442.0, 45760.0, 140752.0]),
           st.floats(min_value=0.1861, max_value=0.196, exclude_min=True, allow_nan=False),
           st.sampled_from([0.059, 0.063, 0.117, 0.125, 0.143, 0.18, 0.273, 0.31, 0.367, 0.438]),
           st.sampled_from([0.105, 0.183, 0.248, 0.296, 0.364, 0.494, 0.504, 0.514, 0.769, 1.0]),
           st.floats(min_value=6.084, max_value=7.354, allow_nan=False),
           st.sampled_from([11.0, 13.0, 26.0, 60.0, 288.0, 331.0, 519.0, 728.0, 6238.0, 7353.0]),
           st.floats(min_value=12.51, max_value=9236.6, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 16.0, 28.0, 29.0, 42.0, 43.0, 153.0, 166.0, 193.0, 1003.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_58(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_58']['n_samples'] += 1
        self.data['tests']['test_58']['samples'].append(x_test)
        self.data['tests']['test_58']['y_expected'].append(y_expected[0])
        self.data['tests']['test_58']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=163.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([15.0, 17.0, 22.0, 26.0, 32.0, 37.0, 47.0, 53.0, 108.0, 372.0]),
           st.floats(min_value=0.1861, max_value=0.196, exclude_min=True, allow_nan=False),
           st.sampled_from([0.339, 0.38, 0.536, 0.598, 0.636, 0.7, 0.706, 0.816, 0.875, 0.993]),
           st.sampled_from([0.432, 0.516, 0.533, 0.565, 0.667, 0.7, 0.75, 0.841, 0.875, 1.0]),
           st.floats(min_value=7.357, max_value=996.885, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 8.0, 10.0, 17.0, 23.0, 27.0, 31.0, 37.0, 47.0, 76.0]),
           st.floats(min_value=12.51, max_value=9236.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 10.0, 14.0, 15.0, 16.0, 32.0, 38.0, 50.0, 54.0, 64.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_59(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_59']['n_samples'] += 1
        self.data['tests']['test_59']['samples'].append(x_test)
        self.data['tests']['test_59']['y_expected'].append(y_expected[0])
        self.data['tests']['test_59']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([61.0, 98.0, 109.0, 125.0, 252.0, 325.0, 393.0, 421.0, 424.0, 435.0]),
           st.sampled_from([784.0, 960.0, 1210.0, 1474.0, 1712.0, 1917.0, 2090.0, 2880.0, 4300.0, 4746.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0888, max_value=0.0979, allow_nan=False),
           st.sampled_from([0.408, 0.543, 0.569, 0.608, 0.629, 0.653, 0.66, 0.733, 0.739, 0.821]),
           st.floats(min_value=7.8, max_value=9.499, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=35.4, max_value=42.49, allow_nan=False),
           st.sampled_from([17.0, 47.0, 150.0, 227.0, 269.0, 309.0, 327.0, 395.0, 450.0, 456.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_60(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_60']['n_samples'] += 1
        self.data['tests']['test_60']['samples'].append(x_test)
        self.data['tests']['test_60']['y_expected'].append(y_expected[0])
        self.data['tests']['test_60']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 17.0, 18.0, 25.0, 27.0, 34.0, 43.0, 60.0, 172.0, 198.0]),
           st.sampled_from([128.0, 740.0, 918.0, 1428.0, 2312.0, 3555.0, 27058.0, 39006.0, 140752.0, 142290.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0982, max_value=0.1105, exclude_min=True, allow_nan=False),
           st.sampled_from([0.165, 0.295, 0.337, 0.371, 0.374, 0.417, 0.497, 0.633, 0.655, 0.75]),
           st.floats(min_value=7.8, max_value=9.499, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=35.4, max_value=42.49, allow_nan=False),
           st.sampled_from([4.0, 6.0, 8.0, 9.0, 16.0, 29.0, 51.0, 193.0, 256.0, 1003.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_61(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_61']['n_samples'] += 1
        self.data['tests']['test_61']['samples'].append(x_test)
        self.data['tests']['test_61']['y_expected'].append(y_expected[0])
        self.data['tests']['test_61']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([3.0, 5.0, 18.0, 20.0, 29.0, 30.0, 37.0, 49.0, 60.0, 442.0]),
           st.sampled_from([8.0, 15.0, 238.0, 924.0, 1392.0, 1548.0, 1830.0, 2312.0, 3255.0, 3871.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1601, max_value=0.1762, exclude_min=True, allow_nan=False),
           st.sampled_from([0.112, 0.296, 0.338, 0.441, 0.457, 0.531, 0.627, 0.655, 0.8, 0.859]),
           st.floats(min_value=4.4, max_value=5.249, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=22.6, max_value=26.49, allow_nan=False),
           st.sampled_from([12.0, 16.0, 18.0, 51.0, 150.0, 153.0, 161.0, 399.0, 1003.0, 3212.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_62(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_62']['n_samples'] += 1
        self.data['tests']['test_62']['samples'].append(x_test)
        self.data['tests']['test_62']['y_expected'].append(y_expected[0])
        self.data['tests']['test_62']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([41.0, 116.0, 125.0, 193.0, 206.0, 210.0, 241.0, 301.0, 393.0, 498.0]),
           st.sampled_from([98.0, 177.0, 265.0, 320.0, 1323.0, 1863.0, 1899.0, 2190.0, 2992.0, 4752.0]),
           st.floats(min_value=0.2362, max_value=0.4146, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1601, max_value=0.1762, exclude_min=True, allow_nan=False),
           st.sampled_from([0.428, 0.502, 0.551, 0.575, 0.73, 0.79, 0.836, 0.869, 0.92, 0.94]),
           st.floats(min_value=5.251, max_value=6.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=22.6, max_value=26.49, allow_nan=False),
           st.sampled_from([34.0, 48.0, 77.0, 184.0, 217.0, 236.0, 283.0, 293.0, 329.0, 563.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 23.0, 40.0, 45.0, 59.0, 70.0, 95.0, 131.0, 280.0, 325.0]),
           st.sampled_from([27.0, 43.0, 100.0, 127.0, 288.0, 460.0, 777.0, 1500.0, 2772.0, 4225.0]),
           st.floats(min_value=1.1286, max_value=108.3028, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1601, max_value=0.1762, exclude_min=True, allow_nan=False),
           st.sampled_from([0.34, 0.487, 0.558, 0.594, 0.677, 0.684, 0.689, 0.906, 0.998, 1.0]),
           st.floats(min_value=5.251, max_value=6.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=22.6, max_value=26.49, allow_nan=False),
           st.sampled_from([2.0, 15.0, 16.0, 21.0, 23.0, 26.0, 29.0, 33.0, 65.0, 579.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([44.0, 96.0, 194.0, 196.0, 225.0, 295.0, 314.0, 363.0, 378.0, 465.0]),
           st.sampled_from([246.0, 272.0, 544.0, 999.0, 1111.0, 1276.0, 1435.0, 1719.0, 1911.0, 2001.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1601, max_value=0.1762, exclude_min=True, allow_nan=False),
           st.sampled_from([0.395, 0.408, 0.429, 0.497, 0.621, 0.717, 0.739, 0.827, 0.972, 0.973]),
           st.floats(min_value=7.8, max_value=9.499, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=26.51, max_value=29.7, exclude_min=True, allow_nan=False),
           st.sampled_from([27.0, 92.0, 96.0, 106.0, 192.0, 377.0, 441.0, 473.0, 513.0, 658.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 87.0, 106.0, 128.0, 129.0, 142.0, 279.0, 333.0, 341.0, 535.0]),
           st.sampled_from([43.0, 51.0, 73.0, 79.0, 206.0, 235.0, 472.0, 1665.0, 2096.0, 4697.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2032, max_value=0.2409, allow_nan=False),
           st.sampled_from([0.095, 0.382, 0.51, 0.558, 0.57, 0.577, 0.686, 0.689, 0.69, 0.761]),
           st.floats(min_value=9.501, max_value=13.627, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=35.4, max_value=42.49, allow_nan=False),
           st.sampled_from([7.0, 9.0, 13.0, 17.0, 24.0, 30.0, 33.0, 65.0, 268.0, 272.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_66(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_66']['n_samples'] += 1
        self.data['tests']['test_66']['samples'].append(x_test)
        self.data['tests']['test_66']['y_expected'].append(y_expected[0])
        self.data['tests']['test_66']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([4.0, 157.0, 170.0, 172.0, 173.0, 196.0, 205.0, 229.0, 335.0, 505.0]),
           st.sampled_from([475.0, 648.0, 768.0, 1326.0, 1416.0, 1470.0, 2490.0, 3038.0, 6062.0, 6640.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2032, max_value=0.2409, allow_nan=False),
           st.sampled_from([0.294, 0.548, 0.552, 0.567, 0.681, 0.725, 0.767, 0.807, 0.818, 0.927]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=42.51, max_value=9260.6, exclude_min=True, allow_nan=False),
           st.sampled_from([28.0, 71.0, 117.0, 185.0, 205.0, 215.0, 305.0, 389.0, 485.0, 537.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_67(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_67']['n_samples'] += 1
        self.data['tests']['test_67']['samples'].append(x_test)
        self.data['tests']['test_67']['y_expected'].append(y_expected[0])
        self.data['tests']['test_67']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([72.0, 124.0, 130.0, 157.0, 212.0, 263.0, 343.0, 372.0, 498.0, 520.0]),
           st.floats(min_value=10.2, max_value=10.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3429, exclude_min=True, allow_nan=False),
           st.sampled_from([0.439, 0.484, 0.503, 0.524, 0.607, 0.654, 0.806, 0.844, 0.916, 0.938]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=8.6, max_value=8.99, allow_nan=False),
           st.sampled_from([38.0, 88.0, 112.0, 171.0, 255.0, 378.0, 384.0, 458.0, 645.0, 885.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([23.0, 171.0, 207.0, 300.0, 331.0, 350.0, 352.0, 376.0, 433.0, 498.0]),
           st.floats(min_value=10.2, max_value=10.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7501, max_value=0.7775, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6824, max_value=0.8374, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=8.6, max_value=8.99, allow_nan=False),
           st.sampled_from([11.0, 101.0, 109.0, 227.0, 228.0, 266.0, 472.0, 505.0, 654.0, 676.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([78.0, 178.0, 193.0, 202.0, 243.0, 264.0, 316.0, 377.0, 389.0, 465.0]),
           st.floats(min_value=10.2, max_value=10.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7501, max_value=0.7775, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8377, max_value=0.8701, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=8.6, max_value=8.99, allow_nan=False),
           st.sampled_from([26.0, 88.0, 108.0, 119.0, 146.0, 309.0, 326.0, 370.0, 454.0, 578.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_70(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_70']['n_samples'] += 1
        self.data['tests']['test_70']['samples'].append(x_test)
        self.data['tests']['test_70']['y_expected'].append(y_expected[0])
        self.data['tests']['test_70']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([50.0, 106.0, 177.0, 180.0, 228.0, 239.0, 250.0, 254.0, 274.0, 490.0]),
           st.floats(min_value=10.2, max_value=10.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3704, exclude_min=True, allow_nan=False),
           st.sampled_from([0.285, 0.465, 0.567, 0.591, 0.632, 0.67, 0.683, 0.761, 0.794, 0.976]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=9.01, max_value=9233.8, exclude_min=True, allow_nan=False),
           st.sampled_from([164.0, 179.0, 189.0, 238.0, 250.0, 349.0, 438.0, 467.0, 542.0, 669.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.floats(min_value=10.2, max_value=10.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.8876, max_value=0.91, exclude_min=True, allow_nan=False),
           st.sampled_from([0.345, 0.432, 0.444, 0.486, 0.516, 0.565, 0.636, 0.7, 0.8, 0.909]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([14.0, 18.0, 23.0, 27.0, 34.0, 47.0, 53.0, 56.0, 76.0, 117.0]),
           st.sampled_from([1.0, 10.0, 15.0, 16.0, 38.0, 50.0, 54.0, 64.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_72(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_72']['n_samples'] += 1
        self.data['tests']['test_72']['samples'].append(x_test)
        self.data['tests']['test_72']['y_expected'].append(y_expected[0])
        self.data['tests']['test_72']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([44.0, 61.0, 69.0, 154.0, 162.0, 317.0, 332.0, 372.0, 400.0, 500.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.175, 0.245, 0.259, 0.633, 0.684, 0.713, 0.731, 0.774, 0.874, 0.915]),
           st.floats(min_value=2.668, max_value=3.084, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([142.0, 171.0, 178.0, 191.0, 302.0, 409.0, 463.0, 590.0, 599.0, 1134.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_73(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_73']['n_samples'] += 1
        self.data['tests']['test_73']['samples'].append(x_test)
        self.data['tests']['test_73']['y_expected'].append(y_expected[0])
        self.data['tests']['test_73']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 5.0, 20.0, 24.0, 49.0, 79.0, 146.0, 198.0, 286.0, 442.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6191, max_value=0.7583, allow_nan=False),
           st.floats(min_value=3.087, max_value=3.419, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([16.0, 51.0, 59.0, 69.0, 98.0, 193.0, 227.0, 399.0, 401.0, 511.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_74(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_74']['n_samples'] += 1
        self.data['tests']['test_74']['samples'].append(x_test)
        self.data['tests']['test_74']['y_expected'].append(y_expected[0])
        self.data['tests']['test_74']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 43.0, 86.0, 113.0, 151.0, 231.0, 258.0, 301.0, 440.0, 498.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7586, max_value=0.8068, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.087, max_value=3.419, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([34.0, 116.0, 173.0, 231.0, 287.0, 302.0, 530.0, 614.0, 657.0, 1134.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_75(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_75']['n_samples'] += 1
        self.data['tests']['test_75']['samples'].append(x_test)
        self.data['tests']['test_75']['y_expected'].append(y_expected[0])
        self.data['tests']['test_75']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([22.0, 59.0, 138.0, 189.0, 285.0, 288.0, 305.0, 436.0, 468.0, 523.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3172, exclude_min=True, allow_nan=False),
           st.sampled_from([0.328, 0.397, 0.492, 0.609, 0.787, 0.813, 0.901, 0.911, 0.936, 0.973]),
           st.floats(min_value=4.751, max_value=9.827, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([90.0, 121.0, 143.0, 170.0, 215.0, 243.0, 370.0, 373.0, 470.0, 486.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6217, max_value=0.6283, exclude_min=True, allow_nan=False),
           st.sampled_from([0.432, 0.444, 0.533, 0.636, 0.667, 0.706, 0.714, 0.8, 0.816, 0.875]),
           st.floats(min_value=4.751, max_value=9.827, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([1.0, 10.0, 15.0, 16.0, 38.0, 50.0, 54.0, 64.0, 108.0, 205.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_77(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_77']['n_samples'] += 1
        self.data['tests']['test_77']['samples'].append(x_test)
        self.data['tests']['test_77']['y_expected'].append(y_expected[0])
        self.data['tests']['test_77']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([73.0, 138.0, 194.0, 203.0, 215.0, 236.0, 258.0, 307.0, 340.0, 342.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6552, max_value=0.7241, exclude_min=True, allow_nan=False),
           st.sampled_from([0.379, 0.447, 0.615, 0.658, 0.709, 0.778, 0.78, 0.825, 0.904, 0.907]),
           st.floats(min_value=4.751, max_value=9.827, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=12.2, max_value=13.49, allow_nan=False),
           st.sampled_from([3.0, 65.0, 96.0, 217.0, 224.0, 382.0, 425.0, 471.0, 532.0, 799.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([67.0, 138.0, 146.0, 167.0, 174.0, 261.0, 292.0, 301.0, 307.0, 533.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.328, 0.475, 0.668, 0.703, 0.721, 0.761, 0.779, 0.85, 0.949, 0.96]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.floats(min_value=13.51, max_value=16.7, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 96.0, 136.0, 163.0, 208.0, 286.0, 316.0, 385.0, 470.0, 527.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([27.0, 43.0, 85.0, 129.0, 158.0, 192.0, 205.0, 339.0, 353.0, 367.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.395, 0.532, 0.538, 0.575, 0.812, 0.857, 0.906, 0.941, 0.944, 0.972]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=9.4, max_value=9.99, allow_nan=False),
           st.floats(min_value=29.51, max_value=9250.2, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 10.0, 124.0, 136.0, 232.0, 364.0, 462.0, 503.0, 569.0, 676.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_80(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_80']['n_samples'] += 1
        self.data['tests']['test_80']['samples'].append(x_test)
        self.data['tests']['test_80']['y_expected'].append(y_expected[0])
        self.data['tests']['test_80']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 8.0, 11.0, 20.0, 22.0, 30.0, 67.0, 70.0, 72.0, 350.0]),
           st.floats(min_value=11.01, max_value=15.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.296, 0.374, 0.39, 0.398, 0.417, 0.494, 0.541, 0.627, 0.643, 0.778]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.01, max_value=10.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=29.51, max_value=9250.2, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 5.0, 17.0, 18.0, 45.0, 149.0, 193.0, 839.0, 888.0, 1003.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_81(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_81']['n_samples'] += 1
        self.data['tests']['test_81']['samples'].append(x_test)
        self.data['tests']['test_81']['y_expected'].append(y_expected[0])
        self.data['tests']['test_81']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=4.2, exclude_min=True, allow_nan=False),
           st.sampled_from([52.0, 64.0, 72.0, 163.0, 181.0, 198.0, 222.0, 316.0, 415.0, 433.0]),
           st.floats(min_value=35.51, max_value=28827.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.435, 0.495, 0.561, 0.581, 0.704, 0.852, 0.872, 0.936, 0.954, 0.99]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([61.0, 396.0, 431.0, 666.0, 1322.0, 1357.0, 1881.0, 2297.0, 2740.0, 4379.0]),
           st.floats(min_value=3.4, max_value=3.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_82(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_82']['n_samples'] += 1
        self.data['tests']['test_82']['samples'].append(x_test)
        self.data['tests']['test_82']['y_expected'].append(y_expected[0])
        self.data['tests']['test_82']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.01, max_value=11.1, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 9.0, 17.0, 25.0, 34.0, 47.0, 71.0, 277.0, 357.0, 465.0]),
           st.floats(min_value=35.51, max_value=28827.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.07, 0.123, 0.179, 0.257, 0.456, 0.473, 0.526, 0.534, 0.627, 0.769]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([41.0, 156.0, 173.0, 226.0, 316.0, 717.0, 951.0, 1958.0, 7279.0, 12422.0]),
           st.floats(min_value=3.4, max_value=3.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_83(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_83']['n_samples'] += 1
        self.data['tests']['test_83']['samples'].append(x_test)
        self.data['tests']['test_83']['y_expected'].append(y_expected[0])
        self.data['tests']['test_83']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 9.0, 21.0, 31.0, 35.0, 38.0, 111.0, 135.0, 315.0, 494.0]),
           st.floats(min_value=35.51, max_value=28827.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.4339, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.344, 0.45, 0.466, 0.485, 0.636, 0.807, 0.888, 0.92, 0.938, 0.947]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([168.0, 178.0, 320.0, 326.0, 427.0, 603.0, 686.0, 1335.0, 1483.0, 1953.0]),
           st.floats(min_value=4.01, max_value=645.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_84(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_84']['n_samples'] += 1
        self.data['tests']['test_84']['samples'].append(x_test)
        self.data['tests']['test_84']['y_expected'].append(y_expected[0])
        self.data['tests']['test_84']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.6, max_value=5.49, allow_nan=False),
           st.floats(min_value=11.01, max_value=28807.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2252, max_value=108.3801, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.sampled_from([0.07, 0.365, 0.417, 0.437, 0.449, 0.483, 0.494, 0.509, 0.533, 0.538]),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([19.0, 27.0, 30.0, 54.0, 163.0, 250.0, 373.0, 1358.0, 12964.0, 23092.0]),
           st.sampled_from([1.0, 7.0, 13.0, 16.0, 20.0, 36.0, 150.0, 401.0, 603.0, 1003.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_85(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_85']['n_samples'] += 1
        self.data['tests']['test_85']['samples'].append(x_test)
        self.data['tests']['test_85']['y_expected'].append(y_expected[0])
        self.data['tests']['test_85']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=5.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=28807.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2252, max_value=108.3801, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7551, max_value=0.9283, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([266.0, 349.0, 556.0, 775.0, 844.0, 1356.0, 1386.0, 3146.0, 3582.0, 4453.0]),
           st.sampled_from([31.0, 66.0, 79.0, 121.0, 272.0, 279.0, 284.0, 355.0, 471.0, 703.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_86(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_86']['n_samples'] += 1
        self.data['tests']['test_86']['samples'].append(x_test)
        self.data['tests']['test_86']['y_expected'].append(y_expected[0])
        self.data['tests']['test_86']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=116.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=28807.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2252, max_value=108.3801, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5372, max_value=0.6559, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([19.0, 38.0, 64.0, 226.0, 637.0, 697.0, 1215.0, 2086.0, 7279.0, 12964.0]),
           st.sampled_from([1.0, 5.0, 12.0, 29.0, 32.0, 105.0, 106.0, 153.0, 256.0, 526.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_87(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_87']['n_samples'] += 1
        self.data['tests']['test_87']['samples'].append(x_test)
        self.data['tests']['test_87']['y_expected'].append(y_expected[0])
        self.data['tests']['test_87']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=7.51, max_value=116.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=28807.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2252, max_value=108.3801, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6562, max_value=0.7106, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([79.0, 315.0, 374.0, 822.0, 875.0, 1576.0, 2248.0, 3043.0, 3400.0, 7740.0]),
           st.sampled_from([80.0, 123.0, 230.0, 255.0, 283.0, 334.0, 338.0, 397.0, 455.0, 459.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.51, max_value=115.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.01, max_value=28807.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2252, max_value=108.3801, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2412, max_value=0.3929, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.9286, max_value=0.9428, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=10.6, max_value=11.49, allow_nan=False),
           st.sampled_from([17.0, 18.0, 21.0, 41.0, 48.0, 143.0, 509.0, 823.0, 4287.0, 17081.0]),
           st.sampled_from([1.0, 15.0, 36.0, 43.0, 54.0, 88.0, 136.0, 147.0, 213.0, 674.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_89(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_89']['n_samples'] += 1
        self.data['tests']['test_89']['samples'].append(x_test)
        self.data['tests']['test_89']['y_expected'].append(y_expected[0])
        self.data['tests']['test_89']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([14.0, 20.0, 34.0, 47.0, 48.0, 67.0, 70.0, 105.0, 106.0, 350.0]),
           st.floats(min_value=2036.6, max_value=2543.99, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.sampled_from([0.052, 0.078, 0.139, 0.16, 0.172, 0.194, 0.208, 0.25, 0.275, 0.875]),
           st.floats(min_value=0.1323, max_value=0.1498, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 28.0, 95.0, 134.0, 283.0, 373.0, 378.0, 1580.0, 1958.0, 7780.0]),
           st.sampled_from([5.0, 7.0, 15.0, 69.0, 98.0, 136.0, 153.0, 164.0, 213.0, 1003.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_90(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_90']['n_samples'] += 1
        self.data['tests']['test_90']['samples'].append(x_test)
        self.data['tests']['test_90']['y_expected'].append(y_expected[0])
        self.data['tests']['test_90']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([28.0, 30.0, 67.0, 124.0, 142.0, 300.0, 535.0, 544.0, 550.0, 553.0]),
           st.floats(min_value=2544.01, max_value=30833.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.sampled_from([0.103, 0.326, 0.348, 0.356, 0.411, 0.519, 0.523, 0.759, 0.858, 0.941]),
           st.floats(min_value=0.1323, max_value=0.1498, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([71.0, 79.0, 95.0, 98.0, 122.0, 366.0, 480.0, 598.0, 657.0, 1630.0]),
           st.sampled_from([5.0, 7.0, 9.0, 10.0, 11.0, 13.0, 15.0, 16.0, 59.0, 65.0]))
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

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 15.0, 24.0, 28.0, 70.0, 167.0, 286.0, 350.0, 442.0, 465.0]),
           st.floats(min_value=91.4, max_value=112.49, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1355, max_value=0.1563, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.1864, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([27.0, 31.0, 70.0, 92.0, 95.0, 316.0, 685.0, 1358.0, 2086.0, 12626.0]),
           st.sampled_from([5.0, 6.0, 15.0, 32.0, 43.0, 44.0, 51.0, 64.0, 136.0, 149.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_92(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_92']['n_samples'] += 1
        self.data['tests']['test_92']['samples'].append(x_test)
        self.data['tests']['test_92']['y_expected'].append(y_expected[0])
        self.data['tests']['test_92']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 90.0, 134.0, 155.0, 210.0, 236.0, 293.0, 296.0, 427.0, 520.0]),
           st.floats(min_value=112.51, max_value=28888.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1355, max_value=0.1563, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.1864, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 133.0, 200.0, 215.0, 255.0, 293.0, 533.0, 1123.0, 6053.0, 8088.0]),
           st.sampled_from([23.0, 67.0, 77.0, 87.0, 134.0, 203.0, 228.0, 267.0, 397.0, 429.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_93(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_93']['n_samples'] += 1
        self.data['tests']['test_93']['samples'].append(x_test)
        self.data['tests']['test_93']['y_expected'].append(y_expected[0])
        self.data['tests']['test_93']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 9.0, 21.0, 27.0, 72.0, 198.0, 402.0, 461.0, 463.0, 465.0]),
           st.sampled_from([28.0, 30.0, 210.0, 528.0, 1800.0, 2730.0, 5865.0, 19278.0, 98368.0, 140752.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1355, max_value=0.1563, allow_nan=False),
           st.floats(min_value=0.3322, max_value=0.3702, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([37.0, 41.0, 42.0, 47.0, 110.0, 134.0, 165.0, 316.0, 4258.0, 17482.0]),
           st.floats(min_value=27.8, max_value=34.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_94(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_94']['n_samples'] += 1
        self.data['tests']['test_94']['samples'].append(x_test)
        self.data['tests']['test_94']['y_expected'].append(y_expected[0])
        self.data['tests']['test_94']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([18.0, 126.0, 153.0, 213.0, 263.0, 293.0, 332.0, 333.0, 335.0, 352.0]),
           st.sampled_from([195.0, 296.0, 480.0, 760.0, 1012.0, 1190.0, 1936.0, 2384.0, 3038.0, 4165.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1355, max_value=0.1563, allow_nan=False),
           st.floats(min_value=0.3322, max_value=0.3702, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([46.0, 811.0, 874.0, 1200.0, 1530.0, 2048.0, 3029.0, 4413.0, 6072.0, 7519.0]),
           st.floats(min_value=34.51, max_value=670.0, exclude_min=True, allow_nan=False))
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

    @given(st.floats(min_value=3.51, max_value=7.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=8.2, max_value=9.99, allow_nan=False),
           st.sampled_from([65.0, 102.0, 169.0, 266.0, 329.0, 1020.0, 2312.0, 4900.0, 13442.0, 143993.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1566, max_value=0.3252, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.2233, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.719, max_value=1.898, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([15.0, 19.0, 48.0, 98.0, 110.0, 173.0, 960.0, 8387.0, 17482.0, 18149.0]),
           st.sampled_from([7.0, 20.0, 28.0, 29.0, 32.0, 35.0, 59.0, 150.0, 153.0, 3212.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_96(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_96']['n_samples'] += 1
        self.data['tests']['test_96']['samples'].append(x_test)
        self.data['tests']['test_96']['y_expected'].append(y_expected[0])
        self.data['tests']['test_96']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=10.01, max_value=118.6, exclude_min=True, allow_nan=False),
           st.sampled_from([89.0, 627.0, 639.0, 940.0, 1450.0, 1665.0, 1710.0, 5090.0, 5940.0, 5964.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1566, max_value=0.3252, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.2233, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.719, max_value=1.898, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([84.0, 299.0, 367.0, 395.0, 683.0, 1238.0, 1244.0, 1510.0, 1629.0, 1632.0]),
           st.sampled_from([1.0, 21.0, 58.0, 93.0, 207.0, 225.0, 239.0, 351.0, 569.0, 731.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_97(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_97']['n_samples'] += 1
        self.data['tests']['test_97']['samples'].append(x_test)
        self.data['tests']['test_97']['y_expected'].append(y_expected[0])
        self.data['tests']['test_97']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.4, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 31.0, 67.0, 76.0, 91.0, 167.0, 176.0, 408.0, 504.0, 506.0]),
           st.sampled_from([243.0, 380.0, 494.0, 1755.0, 1832.0, 2079.0, 4172.0, 5740.0, 5744.0, 8533.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1566, max_value=0.3252, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.2233, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.901, max_value=7.547, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([286.0, 330.0, 667.0, 718.0, 1049.0, 1224.0, 2103.0, 5649.0, 8289.0, 11482.0]),
           st.sampled_from([131.0, 179.0, 180.0, 182.0, 229.0, 308.0, 321.0, 331.0, 381.0, 403.0]))
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

    @given(st.floats(min_value=23.01, max_value=23.9, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 6.0, 13.0, 16.0, 22.0, 48.0, 49.0, 71.0, 72.0, 464.0]),
           st.sampled_from([54.0, 90.0, 329.0, 1147.0, 1800.0, 2010.0, 3969.0, 13442.0, 26145.0, 67626.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1566, max_value=0.3252, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1501, max_value=0.2233, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 27.0, 38.0, 98.0, 165.0, 373.0, 5938.0, 7780.0, 12626.0, 16139.0]),
           st.sampled_from([6.0, 8.0, 12.0, 26.0, 32.0, 34.0, 45.0, 161.0, 256.0, 399.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_99(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_99']['n_samples'] += 1
        self.data['tests']['test_99']['samples'].append(x_test)
        self.data['tests']['test_99']['y_expected'].append(y_expected[0])
        self.data['tests']['test_99']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([21.0, 24.0, 25.0, 28.0, 34.0, 167.0, 198.0, 277.0, 357.0, 463.0]),
           st.sampled_from([40.0, 63.0, 168.0, 238.0, 918.0, 1020.0, 2640.0, 3555.0, 19278.0, 27058.0]),
           st.floats(min_value=0.2362, max_value=0.6199, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1566, max_value=0.3252, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5166, max_value=0.5177, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 16.0, 23.0, 28.0, 70.0, 80.0, 98.0, 165.0, 509.0, 619.0]),
           st.sampled_from([3.0, 9.0, 17.0, 20.0, 44.0, 136.0, 199.0, 603.0, 901.0, 2273.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_100(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_100']['n_samples'] += 1
        self.data['tests']['test_100']['samples'].append(x_test)
        self.data['tests']['test_100']['y_expected'].append(y_expected[0])
        self.data['tests']['test_100']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=3.7, exclude_min=True, allow_nan=False),
           st.sampled_from([17.0, 29.0, 44.0, 46.0, 52.0, 58.0, 107.0, 138.0, 159.0, 379.0]),
           st.sampled_from([19.0, 32.0, 71.0, 84.0, 87.0, 278.0, 413.0, 640.0, 777.0, 1611.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.115, 0.263, 0.374, 0.54, 0.637, 0.765, 0.813, 0.869, 0.901, 0.944]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.712, max_value=5.639, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 28.0, 51.0, 61.0, 88.0, 152.0, 574.0, 735.0, 756.0, 4890.0]),
           st.sampled_from([6.0, 8.0, 17.0, 19.0, 20.0, 26.0, 28.0, 33.0, 77.0, 268.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_101(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_101']['n_samples'] += 1
        self.data['tests']['test_101']['samples'].append(x_test)
        self.data['tests']['test_101']['y_expected'].append(y_expected[0])
        self.data['tests']['test_101']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=9.1, exclude_min=True, allow_nan=False),
           st.sampled_from([4.0, 13.0, 20.0, 25.0, 70.0, 105.0, 277.0, 286.0, 357.0, 463.0]),
           st.sampled_from([32.0, 60.0, 102.0, 329.0, 480.0, 1752.0, 2640.0, 2660.0, 5865.0, 11200.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.078, 0.113, 0.121, 0.133, 0.159, 0.16, 0.189, 0.194, 0.275, 0.295]),
           st.floats(min_value=0.1501, max_value=0.1965, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.855, max_value=2.068, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=46.6, max_value=56.49, allow_nan=False),
           st.floats(min_value=11.0, max_value=13.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_102(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_102']['n_samples'] += 1
        self.data['tests']['test_102']['samples'].append(x_test)
        self.data['tests']['test_102']['y_expected'].append(y_expected[0])
        self.data['tests']['test_102']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=9.1, exclude_min=True, allow_nan=False),
           st.sampled_from([121.0, 155.0, 200.0, 217.0, 250.0, 275.0, 282.0, 305.0, 385.0, 520.0]),
           st.sampled_from([109.0, 260.0, 420.0, 516.0, 2067.0, 2706.0, 3888.0, 5194.0, 5200.0, 19832.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.095, 0.272, 0.289, 0.328, 0.34, 0.38, 0.421, 0.455, 0.516, 0.526]),
           st.floats(min_value=0.3827, max_value=0.4106, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.855, max_value=2.068, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=46.6, max_value=56.49, allow_nan=False),
           st.floats(min_value=11.0, max_value=13.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_103(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_103']['n_samples'] += 1
        self.data['tests']['test_103']['samples'].append(x_test)
        self.data['tests']['test_103']['y_expected'].append(y_expected[0])
        self.data['tests']['test_103']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=9.1, exclude_min=True, allow_nan=False),
           st.sampled_from([35.0, 76.0, 133.0, 165.0, 319.0, 340.0, 414.0, 508.0, 523.0, 536.0]),
           st.sampled_from([38.0, 294.0, 1170.0, 1496.0, 1976.0, 2097.0, 2800.0, 4020.0, 4746.0, 4806.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.078, 0.12, 0.159, 0.251, 0.282, 0.29, 0.415, 0.511, 0.554, 0.621]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.071, max_value=2.784, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=46.6, max_value=56.49, allow_nan=False),
           st.floats(min_value=11.0, max_value=13.49, allow_nan=False))
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

    @given(st.floats(min_value=4.51, max_value=9.1, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 21.0, 25.0, 30.0]),
           st.sampled_from([8.0, 15.0, 17.0, 24.0, 34.0, 42.0, 76.0, 84.0, 136.0, 188.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.063, 0.198, 0.333, 0.339, 0.529, 0.598, 0.643, 0.816, 0.9, 1.0]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.712, max_value=5.639, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=46.6, max_value=56.49, allow_nan=False),
           st.floats(min_value=13.51, max_value=653.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_105(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [4]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_105']['n_samples'] += 1
        self.data['tests']['test_105']['samples'].append(x_test)
        self.data['tests']['test_105']['y_expected'].append(y_expected[0])
        self.data['tests']['test_105']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=4.51, max_value=9.1, exclude_min=True, allow_nan=False),
           st.sampled_from([11.0, 138.0, 157.0, 161.0, 174.0, 192.0, 250.0, 423.0, 497.0, 519.0]),
           st.sampled_from([340.0, 470.0, 885.0, 1122.0, 1199.0, 1326.0, 1472.0, 2349.0, 2620.0, 3230.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.13, 0.134, 0.155, 0.22, 0.311, 0.352, 0.395, 0.516, 0.563, 0.675]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.712, max_value=5.639, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=56.51, max_value=9271.8, exclude_min=True, allow_nan=False),
           st.sampled_from([30.0, 68.0, 139.0, 163.0, 178.0, 325.0, 433.0, 645.0, 664.0, 703.0]))
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

    @given(st.floats(min_value=3.51, max_value=4.3, exclude_min=True, allow_nan=False),
           st.sampled_from([4.0, 5.0, 9.0, 17.0, 20.0, 25.0, 27.0, 30.0, 315.0, 464.0]),
           st.sampled_from([45.0, 72.0, 153.0, 266.0, 528.0, 2730.0, 3969.0, 5865.0, 12390.0, 27058.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.087, 0.09, 0.098, 0.116, 0.121, 0.16, 0.181, 0.189, 0.344, 0.438]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.642, max_value=10.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=78.6, max_value=96.49, allow_nan=False),
           st.sampled_from([3.0, 6.0, 20.0, 34.0, 45.0, 59.0, 120.0, 136.0, 674.0, 838.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_107(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_107']['n_samples'] += 1
        self.data['tests']['test_107']['samples'].append(x_test)
        self.data['tests']['test_107']['y_expected'].append(y_expected[0])
        self.data['tests']['test_107']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=4.3, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 58.0, 61.0, 63.0, 64.0, 79.0, 138.0, 214.0, 286.0, 510.0]),
           st.sampled_from([9.0, 41.0, 52.0, 67.0, 75.0, 196.0, 339.0, 398.0, 858.0, 12240.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.373, 0.424, 0.464, 0.475, 0.507, 0.568, 0.579, 0.916, 0.955, 0.986]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.642, max_value=10.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=96.51, max_value=9303.8, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 7.0, 14.0, 17.0, 20.0, 26.0, 30.0, 37.0, 59.0, 120.0]))
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

    @given(st.floats(min_value=7.51, max_value=11.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.0, max_value=63.49, allow_nan=False),
           st.sampled_from([208.0, 371.0, 1208.0, 1304.0, 1834.0, 3090.0, 3766.0, 4944.0, 6016.0, 7275.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.097, 0.159, 0.169, 0.22, 0.326, 0.355, 0.592, 0.67, 0.686, 0.925]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.642, max_value=10.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([59.0, 293.0, 710.0, 941.0, 1828.0, 2248.0, 2318.0, 2612.0, 3150.0, 7729.0]),
           st.floats(min_value=9.0, max_value=10.99, allow_nan=False))
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

    @given(st.floats(min_value=7.51, max_value=11.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=51.0, max_value=63.49, allow_nan=False),
           st.sampled_from([20.0, 105.0, 169.0, 1708.0, 1728.0, 1800.0, 2660.0, 25935.0, 26062.0, 142290.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.052, 0.063, 0.08, 0.09, 0.128, 0.142, 0.159, 0.178, 0.256, 0.286]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.642, max_value=10.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([34.0, 38.0, 42.0, 80.0, 92.0, 173.0, 225.0, 283.0, 2588.0, 17081.0]),
           st.floats(min_value=11.01, max_value=651.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_110(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_110']['n_samples'] += 1
        self.data['tests']['test_110']['samples'].append(x_test)
        self.data['tests']['test_110']['y_expected'].append(y_expected[0])
        self.data['tests']['test_110']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=7.51, max_value=11.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=63.51, max_value=161.4, exclude_min=True, allow_nan=False),
           st.sampled_from([301.0, 679.0, 864.0, 869.0, 1067.0, 1370.0, 1550.0, 1690.0, 2401.0, 4872.0]),
           st.floats(min_value=2.1551, max_value=109.124, exclude_min=True, allow_nan=False),
           st.sampled_from([0.126, 0.2, 0.23, 0.248, 0.361, 0.376, 0.394, 0.436, 0.584, 0.662]),
           st.floats(min_value=0.1501, max_value=0.2245, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.642, max_value=10.54, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([152.0, 252.0, 408.0, 872.0, 1162.0, 1283.0, 1434.0, 1725.0, 1981.0, 2612.0]),
           st.sampled_from([56.0, 141.0, 147.0, 155.0, 206.0, 312.0, 362.0, 444.0, 450.0, 530.0]))
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

    @given(st.floats(min_value=3.51, max_value=4.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([18.0, 186.0, 265.0, 644.0, 665.0, 1410.0, 1431.0, 1533.0, 2736.0, 3177.0]),
           st.floats(min_value=0.2362, max_value=0.2674, exclude_min=True, allow_nan=False),
           st.sampled_from([0.056, 0.192, 0.31, 0.33, 0.417, 0.429, 0.438, 0.52, 0.573, 0.632]),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 84.0, 418.0, 847.0, 876.0, 940.0, 1024.0, 1190.0, 1455.0, 3273.0]),
           st.sampled_from([5.0, 15.0, 19.0, 196.0, 232.0, 255.0, 326.0, 338.0, 365.0, 773.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_112(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_112']['n_samples'] += 1
        self.data['tests']['test_112']['samples'].append(x_test)
        self.data['tests']['test_112']['y_expected'].append(y_expected[0])
        self.data['tests']['test_112']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=10.01, max_value=13.5, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([28.0, 104.0, 169.0, 266.0, 336.0, 2592.0, 2666.0, 6192.0, 67626.0, 98368.0]),
           st.floats(min_value=0.2362, max_value=0.2674, exclude_min=True, allow_nan=False),
           st.sampled_from([0.084, 0.095, 0.098, 0.114, 0.142, 0.156, 0.181, 0.201, 0.233, 0.256]),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 34.0, 38.0, 48.0, 50.0, 110.0, 143.0, 373.0, 924.0, 17452.0]),
           st.sampled_from([2.0, 4.0, 13.0, 17.0, 59.0, 149.0, 161.0, 227.0, 691.0, 901.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_113(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_113']['n_samples'] += 1
        self.data['tests']['test_113']['samples'].append(x_test)
        self.data['tests']['test_113']['y_expected'].append(y_expected[0])
        self.data['tests']['test_113']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=14.6, max_value=17.99, allow_nan=False),
           st.sampled_from([12.0, 19.0, 35.0, 37.0, 257.0, 408.0, 413.0, 1400.0, 1500.0, 7500.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0935, max_value=0.1038, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([38.0, 46.0, 77.0, 129.0, 366.0, 537.0, 574.0, 627.0, 662.0, 1213.0]),
           st.sampled_from([1.0, 7.0, 9.0, 16.0, 18.0, 20.0, 27.0, 37.0, 120.0, 268.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_114(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_114']['n_samples'] += 1
        self.data['tests']['test_114']['samples'].append(x_test)
        self.data['tests']['test_114']['y_expected'].append(y_expected[0])
        self.data['tests']['test_114']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.01, max_value=123.3, exclude_min=True, allow_nan=False),
           st.sampled_from([357.0, 364.0, 392.0, 410.0, 754.0, 1208.0, 2380.0, 3322.0, 3766.0, 7275.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.0935, max_value=0.1038, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([336.0, 419.0, 424.0, 565.0, 649.0, 808.0, 959.0, 1386.0, 2877.0, 4369.0]),
           st.sampled_from([3.0, 14.0, 41.0, 92.0, 114.0, 213.0, 241.0, 349.0, 575.0, 617.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_115(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_115']['n_samples'] += 1
        self.data['tests']['test_115']['samples'].append(x_test)
        self.data['tests']['test_115']['y_expected'].append(y_expected[0])
        self.data['tests']['test_115']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([36.0, 40.0, 60.0, 780.0, 1800.0, 2544.0, 19278.0, 23972.0, 26145.0, 140752.0]),
           st.floats(min_value=0.3926, max_value=0.4898, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.2832, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.064, max_value=1.079, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([18.0, 41.0, 50.0, 64.0, 70.0, 80.0, 685.0, 1296.0, 2588.0, 12964.0]),
           st.sampled_from([1.0, 5.0, 42.0, 511.0, 526.0, 691.0, 838.0, 1003.0, 1025.0, 3212.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_116(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_116']['n_samples'] += 1
        self.data['tests']['test_116']['samples'].append(x_test)
        self.data['tests']['test_116']['y_expected'].append(y_expected[0])
        self.data['tests']['test_116']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([319.0, 483.0, 612.0, 679.0, 1112.0, 1508.0, 1592.0, 1704.0, 1818.0, 2565.0]),
           st.floats(min_value=0.8792, max_value=108.1033, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.2832, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.064, max_value=1.079, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([240.0, 589.0, 883.0, 929.0, 930.0, 997.0, 1108.0, 1195.0, 2731.0, 7010.0]),
           st.sampled_from([61.0, 107.0, 116.0, 136.0, 164.0, 171.0, 186.0, 455.0, 485.0, 578.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_117(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_117']['n_samples'] += 1
        self.data['tests']['test_117']['samples'].append(x_test)
        self.data['tests']['test_117']['y_expected'].append(y_expected[0])
        self.data['tests']['test_117']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=4.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([14.0, 36.0, 41.0, 145.0, 199.0, 200.0, 348.0, 472.0, 1650.0, 1665.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.2832, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.5448, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.4, max_value=22.49, allow_nan=False),
           st.sampled_from([1.0, 3.0, 5.0, 15.0, 17.0, 21.0, 29.0, 37.0, 59.0, 176.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_118(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_118']['n_samples'] += 1
        self.data['tests']['test_118']['samples'].append(x_test)
        self.data['tests']['test_118']['y_expected'].append(y_expected[0])
        self.data['tests']['test_118']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.01, max_value=10.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([40.0, 128.0, 210.0, 464.0, 893.0, 924.0, 2496.0, 9999.0, 11200.0, 23972.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.2832, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.5448, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.4, max_value=22.49, allow_nan=False),
           st.sampled_from([1.0, 6.0, 18.0, 20.0, 32.0, 45.0, 59.0, 603.0, 838.0, 2273.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_119(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_119']['n_samples'] += 1
        self.data['tests']['test_119']['samples'].append(x_test)
        self.data['tests']['test_119']['y_expected'].append(y_expected[0])
        self.data['tests']['test_119']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([160.0, 249.0, 592.0, 970.0, 1183.0, 1470.0, 1520.0, 3230.0, 3288.0, 3633.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.2832, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6337, max_value=0.7069, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.4, max_value=22.49, allow_nan=False),
           st.sampled_from([81.0, 91.0, 229.0, 241.0, 403.0, 432.0, 501.0, 507.0, 533.0, 621.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_120(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_120']['n_samples'] += 1
        self.data['tests']['test_120']['samples'].append(x_test)
        self.data['tests']['test_120']['y_expected'].append(y_expected[0])
        self.data['tests']['test_120']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([55.0, 112.0, 351.0, 414.0, 1431.0, 2170.0, 2280.0, 2688.0, 3321.0, 4824.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.1107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=1.378, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 56.0, 94.0, 212.0, 229.0, 318.0, 361.0, 388.0, 527.0, 650.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_121(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_121']['n_samples'] += 1
        self.data['tests']['test_121']['samples'].append(x_test)
        self.data['tests']['test_121']['y_expected'].append(y_expected[0])
        self.data['tests']['test_121']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.sampled_from([484.0, 740.0, 850.0, 893.0, 924.0, 1147.0, 1769.0, 2201.0, 3392.0, 22991.0]),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1041, max_value=0.1107, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.566, max_value=8.079, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([28.0, 44.0, 45.0, 51.0, 54.0, 256.0, 526.0, 1025.0, 1227.0, 3212.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_122(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_122']['n_samples'] += 1
        self.data['tests']['test_122']['samples'].append(x_test)
        self.data['tests']['test_122']['y_expected'].append(y_expected[0])
        self.data['tests']['test_122']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=36.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_123(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_123']['n_samples'] += 1
        self.data['tests']['test_123']['samples'].append(x_test)
        self.data['tests']['test_123']['y_expected'].append(y_expected[0])
        self.data['tests']['test_123']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=90.01, max_value=9298.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8, max_value=4.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_124(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_124']['n_samples'] += 1
        self.data['tests']['test_124']['samples'].append(x_test)
        self.data['tests']['test_124']['y_expected'].append(y_expected[0])
        self.data['tests']['test_124']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.5346, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_125(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_125']['n_samples'] += 1
        self.data['tests']['test_125']['samples'].append(x_test)
        self.data['tests']['test_125']['y_expected'].append(y_expected[0])
        self.data['tests']['test_125']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5827, max_value=0.5828, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=23.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=29.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_126(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_126']['n_samples'] += 1
        self.data['tests']['test_126']['samples'].append(x_test)
        self.data['tests']['test_126']['y_expected'].append(y_expected[0])
        self.data['tests']['test_126']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5827, max_value=0.5828, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=23.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.51, max_value=9274.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_127(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_127']['n_samples'] += 1
        self.data['tests']['test_127']['samples'].append(x_test)
        self.data['tests']['test_127']['y_expected'].append(y_expected[0])
        self.data['tests']['test_127']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5827, max_value=0.5828, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=69.51, max_value=6659.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_128(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_128']['n_samples'] += 1
        self.data['tests']['test_128']['samples'].append(x_test)
        self.data['tests']['test_128']['y_expected'].append(y_expected[0])
        self.data['tests']['test_128']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=103.0, max_value=126.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5836, max_value=0.6668, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=35.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_129(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_129']['n_samples'] += 1
        self.data['tests']['test_129']['samples'].append(x_test)
        self.data['tests']['test_129']['y_expected'].append(y_expected[0])
        self.data['tests']['test_129']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=127.01, max_value=1843.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5836, max_value=0.6043, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=35.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_130(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_130']['n_samples'] += 1
        self.data['tests']['test_130']['samples'].append(x_test)
        self.data['tests']['test_130']['y_expected'].append(y_expected[0])
        self.data['tests']['test_130']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=127.01, max_value=1843.4, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.6876, max_value=0.75, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=35.9, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_131(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_131']['n_samples'] += 1
        self.data['tests']['test_131']['samples'].append(x_test)
        self.data['tests']['test_131']['y_expected'].append(y_expected[0])
        self.data['tests']['test_131']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5836, max_value=0.6668, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=89.51, max_value=9298.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.51, max_value=646.0, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_132(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_132']['n_samples'] += 1
        self.data['tests']['test_132']['samples'].append(x_test)
        self.data['tests']['test_132']['y_expected'].append(y_expected[0])
        self.data['tests']['test_132']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=24.51, max_value=25.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.1503, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([47.0, 109.0, 110.0, 144.0, 169.0, 193.0, 201.0, 262.0, 444.0, 609.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_133(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_133']['n_samples'] += 1
        self.data['tests']['test_133']['samples'].append(x_test)
        self.data['tests']['test_133']['y_expected'].append(y_expected[0])
        self.data['tests']['test_133']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=24.51, max_value=25.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2016, max_value=0.2045, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([3.0, 7.0, 8.0, 13.0, 14.0, 22.0, 26.0, 27.0, 30.0, 120.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_134(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_134']['n_samples'] += 1
        self.data['tests']['test_134']['samples'].append(x_test)
        self.data['tests']['test_134']['y_expected'].append(y_expected[0])
        self.data['tests']['test_134']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=24.51, max_value=25.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=6968.6, max_value=8708.99, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2166, max_value=0.3732, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([5.0, 50.0, 138.0, 477.0, 515.0, 530.0, 578.0, 621.0, 640.0, 773.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_135(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_135']['n_samples'] += 1
        self.data['tests']['test_135']['samples'].append(x_test)
        self.data['tests']['test_135']['y_expected'].append(y_expected[0])
        self.data['tests']['test_135']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=7.1, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=8709.01, max_value=35765.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 9.0, 12.0, 18.0, 21.0, 25.0, 36.0, 59.0, 65.0, 120.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_136(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_136']['n_samples'] += 1
        self.data['tests']['test_136']['samples'].append(x_test)
        self.data['tests']['test_136']['y_expected'].append(y_expected[0])
        self.data['tests']['test_136']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=21.51, max_value=22.7, exclude_min=True, allow_nan=False),
           st.floats(min_value=435.8, max_value=544.49, allow_nan=False),
           st.floats(min_value=8709.01, max_value=35765.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3926, max_value=107.714, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1376, max_value=0.31, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.082, max_value=6.892, exclude_min=True, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.floats(min_value=22.51, max_value=9244.6, exclude_min=True, allow_nan=False),
           st.sampled_from([137.0, 142.0, 255.0, 286.0, 308.0, 364.0, 425.0, 449.0, 458.0, 520.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_137(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_137']['n_samples'] += 1
        self.data['tests']['test_137']['samples'].append(x_test)
        self.data['tests']['test_137']['y_expected'].append(y_expected[0])
        self.data['tests']['test_137']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.floats(min_value=544.51, max_value=546.2, exclude_min=True, allow_nan=False),
           st.sampled_from([50.0, 62.0, 64.0, 87.0, 95.0, 232.0, 277.0, 278.0, 852.0, 892.0]),
           st.floats(min_value=0.2362, max_value=107.5889, exclude_min=True, allow_nan=False),
           st.sampled_from([0.31, 0.4, 0.469, 0.507, 0.55, 0.568, 0.897, 0.923, 0.941, 0.984]),
           st.floats(min_value=0.5227, max_value=0.6181, exclude_min=True, allow_nan=False),
           st.floats(min_value=24.308, max_value=30.134, allow_nan=False),
           st.floats(min_value=11.51, max_value=6612.6, exclude_min=True, allow_nan=False),
           st.sampled_from([36.0, 108.0, 152.0, 160.0, 186.0, 193.0, 257.0, 349.0, 379.0, 609.0]),
           st.sampled_from([1.0, 7.0, 19.0, 27.0, 37.0, 40.0, 59.0, 149.0, 176.0, 207.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_138(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_138']['n_samples'] += 1
        self.data['tests']['test_138']['samples'].append(x_test)
        self.data['tests']['test_138']['y_expected'].append(y_expected[0])
        self.data['tests']['test_138']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 17.0, 18.0, 41.0, 45.0, 57.0, 75.0, 79.0, 86.0, 127.0]),
           st.sampled_from([15.0, 22.0, 34.0, 45.0, 61.0, 180.0, 777.0, 1500.0, 1623.0, 1650.0]),
           st.floats(min_value=0.2362, max_value=1.4863, exclude_min=True, allow_nan=False),
           st.sampled_from([0.231, 0.296, 0.305, 0.318, 0.506, 0.57, 0.756, 0.847, 0.9, 0.984]),
           st.sampled_from([0.066, 0.382, 0.413, 0.504, 0.519, 0.722, 0.737, 0.85, 0.882, 1.0]),
           st.floats(min_value=30.137, max_value=1015.109, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 20.0, 58.0, 94.0, 127.0, 200.0, 412.0, 532.0, 583.0, 10359.0]),
           st.floats(min_value=28.2, max_value=33.49, allow_nan=False),
           st.sampled_from([13.0, 15.0, 19.0, 24.0, 29.0, 40.0, 59.0, 120.0, 268.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_139(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_139']['n_samples'] += 1
        self.data['tests']['test_139']['samples'].append(x_test)
        self.data['tests']['test_139']['y_expected'].append(y_expected[0])
        self.data['tests']['test_139']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([39.0, 62.0, 89.0, 91.0, 96.0, 195.0, 233.0, 271.0, 382.0, 418.0]),
           st.sampled_from([21.0, 125.0, 455.0, 575.0, 1048.0, 1098.0, 1188.0, 2480.0, 4806.0, 5720.0]),
           st.floats(min_value=0.2362, max_value=1.4863, exclude_min=True, allow_nan=False),
           st.sampled_from([0.067, 0.124, 0.239, 0.261, 0.278, 0.284, 0.393, 0.444, 0.538, 0.579]),
           st.sampled_from([0.339, 0.544, 0.55, 0.599, 0.665, 0.692, 0.832, 0.867, 0.955, 0.956]),
           st.floats(min_value=30.137, max_value=1015.109, exclude_min=True, allow_nan=False),
           st.sampled_from([100.0, 169.0, 232.0, 364.0, 421.0, 433.0, 534.0, 572.0, 582.0, 976.0]),
           st.floats(min_value=33.51, max_value=9253.4, exclude_min=True, allow_nan=False),
           st.sampled_from([10.0, 118.0, 219.0, 224.0, 307.0, 351.0, 368.0, 385.0, 563.0, 773.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_140(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_140']['n_samples'] += 1
        self.data['tests']['test_140']['samples'].append(x_test)
        self.data['tests']['test_140']['y_expected'].append(y_expected[0])
        self.data['tests']['test_140']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([21.0, 124.0, 125.0, 166.0, 197.0, 232.0, 332.0, 336.0, 348.0, 382.0]),
           st.sampled_from([77.0, 391.0, 1056.0, 1210.0, 1837.0, 1969.0, 3684.0, 3825.0, 3888.0, 4488.0]),
           st.floats(min_value=6.4871, max_value=112.5896, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4092, max_value=0.4984, allow_nan=False),
           st.sampled_from([0.254, 0.359, 0.462, 0.786, 0.795, 0.834, 0.845, 0.91, 0.918, 0.92]),
           st.floats(min_value=30.137, max_value=30.857, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 73.0, 221.0, 567.0, 573.0, 937.0, 1820.0, 1971.0, 2182.0, 2736.0]),
           st.sampled_from([219.0, 432.0, 447.0, 1005.0, 1497.0, 2040.0, 2681.0, 2875.0, 3424.0, 4191.0]),
           st.sampled_from([25.0, 39.0, 56.0, 74.0, 76.0, 90.0, 158.0, 351.0, 479.0, 533.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_141(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_141']['n_samples'] += 1
        self.data['tests']['test_141']['samples'].append(x_test)
        self.data['tests']['test_141']['y_expected'].append(y_expected[0])
        self.data['tests']['test_141']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([22.0, 49.0, 65.0, 75.0, 77.0, 129.0, 138.0, 172.0, 278.0, 288.0]),
           st.sampled_from([8.0, 11.0, 95.0, 138.0, 161.0, 257.0, 549.0, 892.0, 4225.0, 6072.0]),
           st.floats(min_value=6.4871, max_value=112.5896, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4987, max_value=0.5989, exclude_min=True, allow_nan=False),
           st.sampled_from([0.34, 0.406, 0.413, 0.494, 0.577, 0.603, 0.609, 0.614, 0.688, 0.996]),
           st.floats(min_value=30.137, max_value=30.857, exclude_min=True, allow_nan=False),
           st.sampled_from([29.0, 34.0, 53.0, 80.0, 84.0, 91.0, 110.0, 177.0, 412.0, 534.0]),
           st.sampled_from([13.0, 41.0, 47.0, 62.0, 124.0, 545.0, 627.0, 652.0, 662.0, 4890.0]),
           st.sampled_from([1.0, 10.0, 19.0, 20.0, 24.0, 26.0, 40.0, 45.0, 176.0, 268.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_142(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_142']['n_samples'] += 1
        self.data['tests']['test_142']['samples'].append(x_test)
        self.data['tests']['test_142']['y_expected'].append(y_expected[0])
        self.data['tests']['test_142']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=8.3, exclude_min=True, allow_nan=False),
           st.sampled_from([2.0, 14.0, 25.0, 35.0, 40.0, 252.0, 259.0, 286.0, 333.0, 535.0]),
           st.sampled_from([44.0, 55.0, 62.0, 63.0, 92.0, 107.0, 145.0, 232.0, 398.0, 1072.0]),
           st.floats(min_value=6.4871, max_value=112.5896, exclude_min=True, allow_nan=False),
           st.sampled_from([0.177, 0.348, 0.41, 0.46, 0.51, 0.547, 0.583, 0.643, 0.822, 0.9]),
           st.sampled_from([0.273, 0.364, 0.379, 0.519, 0.555, 0.558, 0.603, 0.614, 0.996, 1.0]),
           st.floats(min_value=33.741, max_value=1017.992, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 32.0, 39.0, 49.0, 57.0, 98.0, 110.0, 168.0, 366.0, 583.0]),
           st.sampled_from([23.0, 32.0, 58.0, 71.0, 596.0, 657.0, 667.0, 790.0, 1264.0, 1630.0]),
           st.sampled_from([10.0, 20.0, 21.0, 22.0, 24.0, 59.0, 65.0, 176.0, 207.0, 579.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_143(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_143']['n_samples'] += 1
        self.data['tests']['test_143']['samples'].append(x_test)
        self.data['tests']['test_143']['y_expected'].append(y_expected[0])
        self.data['tests']['test_143']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([3.0, 87.0, 114.0, 127.0, 162.0, 222.0, 348.0, 404.0, 420.0, 491.0]),
           st.sampled_from([470.0, 1359.0, 1526.0, 1710.0, 1715.0, 2560.0, 2655.0, 2842.0, 11232.0, 25748.0]),
           st.floats(min_value=0.2362, max_value=0.2858, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1155, max_value=0.1313, allow_nan=False),
           st.sampled_from([0.285, 0.294, 0.477, 0.481, 0.495, 0.642, 0.706, 0.766, 0.834, 0.862]),
           st.sampled_from([1.07, 2.07, 2.48, 2.73, 3.14, 5.51, 6.5, 18.0, 22.5, 127.0]),
           st.sampled_from([43.0, 223.0, 273.0, 422.0, 582.0, 643.0, 928.0, 1490.0, 2152.0, 3464.0]),
           st.sampled_from([312.0, 469.0, 828.0, 904.0, 1161.0, 1503.0, 1774.0, 2239.0, 4141.0, 4204.0]),
           st.sampled_from([11.0, 93.0, 138.0, 304.0, 388.0, 424.0, 544.0, 632.0, 636.0, 654.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_144(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_144']['n_samples'] += 1
        self.data['tests']['test_144']['samples'].append(x_test)
        self.data['tests']['test_144']['y_expected'].append(y_expected[0])
        self.data['tests']['test_144']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([12.0, 13.0, 21.0, 60.0, 79.0, 146.0, 172.0, 285.0, 286.0, 350.0]),
           st.sampled_from([28.0, 54.0, 84.0, 238.0, 464.0, 2312.0, 2640.0, 3555.0, 5762.0, 142290.0]),
           st.floats(min_value=0.2362, max_value=0.2858, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1316, max_value=0.1655, exclude_min=True, allow_nan=False),
           st.sampled_from([0.062, 0.16, 0.238, 0.257, 0.267, 0.319, 0.377, 0.458, 0.494, 0.6]),
           st.sampled_from([1.97, 2.83, 3.14, 3.71, 4.81, 6.63, 7.36, 7.84, 8.24, 8.77]),
           st.sampled_from([10.0, 34.0, 70.0, 162.0, 331.0, 345.0, 370.0, 671.0, 1398.0, 5066.0]),
           st.sampled_from([48.0, 64.0, 110.0, 134.0, 143.0, 693.0, 694.0, 1485.0, 4287.0, 10347.0]),
           st.sampled_from([9.0, 16.0, 17.0, 29.0, 105.0, 149.0, 164.0, 227.0, 256.0, 839.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_145(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_145']['n_samples'] += 1
        self.data['tests']['test_145']['samples'].append(x_test)
        self.data['tests']['test_145']['y_expected'].append(y_expected[0])
        self.data['tests']['test_145']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([9.0, 16.0, 20.0, 22.0, 28.0, 48.0, 60.0, 67.0, 315.0, 326.0]),
           st.sampled_from([44.0, 63.0, 77.0, 238.0, 336.0, 432.0, 1147.0, 1428.0, 3255.0, 140752.0]),
           st.floats(min_value=0.4847, max_value=0.9274, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2515, max_value=0.3013, allow_nan=False),
           st.sampled_from([0.195, 0.238, 0.332, 0.365, 0.374, 0.504, 0.534, 0.583, 0.65, 0.859]),
           st.sampled_from([2.29, 2.37, 4.5, 4.67, 4.81, 5.5, 6.5, 6.63, 7.1, 19.63]),
           st.sampled_from([8.0, 16.0, 217.0, 247.0, 345.0, 400.0, 449.0, 458.0, 519.0, 7772.0]),
           st.sampled_from([36.0, 38.0, 42.0, 48.0, 144.0, 229.0, 693.0, 951.0, 1485.0, 18149.0]),
           st.sampled_from([8.0, 17.0, 29.0, 34.0, 44.0, 59.0, 105.0, 106.0, 213.0, 901.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_146(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_146']['n_samples'] += 1
        self.data['tests']['test_146']['samples'].append(x_test)
        self.data['tests']['test_146']['y_expected'].append(y_expected[0])
        self.data['tests']['test_146']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=82.6, max_value=102.99, allow_nan=False),
           st.sampled_from([49.0, 329.0, 372.0, 528.0, 648.0, 777.0, 2392.0, 3542.0, 6766.0, 12275.0]),
           st.floats(min_value=2.6987, max_value=2.9213, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2515, max_value=0.3013, allow_nan=False),
           st.sampled_from([0.317, 0.319, 0.457, 0.542, 0.708, 0.825, 0.878, 0.922, 0.954, 0.958]),
           st.sampled_from([1.74, 2.25, 3.05, 3.44, 3.54, 4.15, 4.3, 8.25, 8.93, 16.83]),
           st.sampled_from([365.0, 397.0, 504.0, 641.0, 872.0, 928.0, 984.0, 1027.0, 1657.0, 7164.0]),
           st.sampled_from([207.0, 281.0, 492.0, 544.0, 1009.0, 1190.0, 1698.0, 2800.0, 3857.0, 3983.0]),
           st.sampled_from([6.0, 8.0, 86.0, 121.0, 194.0, 265.0, 296.0, 443.0, 678.0, 697.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_147(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_147']['n_samples'] += 1
        self.data['tests']['test_147']['samples'].append(x_test)
        self.data['tests']['test_147']['y_expected'].append(y_expected[0])
        self.data['tests']['test_147']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=103.01, max_value=193.0, exclude_min=True, allow_nan=False),
           st.sampled_from([25.0, 368.0, 548.0, 790.0, 1430.0, 1729.0, 1840.0, 2240.0, 2448.0, 5727.0]),
           st.floats(min_value=2.6987, max_value=2.9213, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2515, max_value=0.3013, allow_nan=False),
           st.sampled_from([0.385, 0.461, 0.491, 0.569, 0.752, 0.821, 0.865, 0.886, 0.932, 0.938]),
           st.floats(min_value=3.059, max_value=3.573, allow_nan=False),
           st.sampled_from([189.0, 386.0, 431.0, 505.0, 506.0, 591.0, 594.0, 596.0, 598.0, 969.0]),
           st.sampled_from([289.0, 542.0, 678.0, 769.0, 924.0, 1711.0, 2305.0, 2914.0, 3312.0, 4194.0]),
           st.sampled_from([54.0, 103.0, 122.0, 166.0, 220.0, 236.0, 358.0, 429.0, 506.0, 650.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_148(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_148']['n_samples'] += 1
        self.data['tests']['test_148']['samples'].append(x_test)
        self.data['tests']['test_148']['y_expected'].append(y_expected[0])
        self.data['tests']['test_148']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=103.01, max_value=193.0, exclude_min=True, allow_nan=False),
           st.sampled_from([8.0, 36.0, 54.0, 63.0, 169.0, 210.0, 850.0, 1769.0, 2592.0, 3969.0]),
           st.floats(min_value=2.6987, max_value=2.9213, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2515, max_value=0.3013, allow_nan=False),
           st.sampled_from([0.16, 0.248, 0.267, 0.269, 0.317, 0.519, 0.594, 0.6, 0.619, 0.655]),
           st.floats(min_value=3.576, max_value=993.86, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 13.0, 34.0, 308.0, 375.0, 449.0, 728.0, 793.0, 5066.0, 7772.0]),
           st.sampled_from([16.0, 44.0, 76.0, 283.0, 454.0, 509.0, 1215.0, 4287.0, 17452.0, 18149.0]),
           st.sampled_from([4.0, 16.0, 17.0, 20.0, 29.0, 106.0, 136.0, 153.0, 256.0, 691.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_149(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_149']['n_samples'] += 1
        self.data['tests']['test_149']['samples'].append(x_test)
        self.data['tests']['test_149']['y_expected'].append(y_expected[0])
        self.data['tests']['test_149']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 21.0, 27.0, 57.0, 115.0, 131.0, 132.0, 136.0, 325.0, 552.0]),
           st.sampled_from([57.0, 67.0, 81.0, 84.0, 107.0, 114.0, 277.0, 282.0, 830.0, 858.0]),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1747, max_value=0.2053, allow_nan=False),
           st.sampled_from([0.424, 0.472, 0.537, 0.544, 0.695, 0.743, 0.764, 0.802, 0.816, 0.92]),
           st.sampled_from([4.86, 6.33, 11.0, 13.29, 15.0, 15.17, 27.13, 29.0, 29.32, 537.0]),
           st.sampled_from([69.0, 95.0, 139.0, 159.0, 479.0, 537.0, 583.0, 628.0, 746.0, 5328.0]),
           st.sampled_from([13.0, 23.0, 36.0, 79.0, 147.0, 213.0, 255.0, 317.0, 627.0, 5784.0]),
           st.floats(min_value=72.6, max_value=90.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_150(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_150']['n_samples'] += 1
        self.data['tests']['test_150']['samples'].append(x_test)
        self.data['tests']['test_150']['y_expected'].append(y_expected[0])
        self.data['tests']['test_150']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([113.0, 118.0, 163.0, 182.0, 214.0, 272.0, 376.0, 398.0, 492.0, 500.0]),
           st.sampled_from([31.0, 252.0, 319.0, 1008.0, 1089.0, 1232.0, 1590.0, 2116.0, 4410.0, 4788.0]),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1747, max_value=0.2053, allow_nan=False),
           st.sampled_from([0.403, 0.487, 0.492, 0.572, 0.618, 0.708, 0.83, 0.842, 0.887, 0.915]),
           st.floats(min_value=2.4, max_value=2.749, allow_nan=False),
           st.sampled_from([210.0, 213.0, 217.0, 240.0, 246.0, 253.0, 278.0, 862.0, 1314.0, 2207.0]),
           st.sampled_from([167.0, 225.0, 252.0, 327.0, 533.0, 804.0, 1971.0, 2349.0, 2531.0, 4379.0]),
           st.floats(min_value=90.51, max_value=714.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_151(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_151']['n_samples'] += 1
        self.data['tests']['test_151']['samples'].append(x_test)
        self.data['tests']['test_151']['y_expected'].append(y_expected[0])
        self.data['tests']['test_151']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 7.0, 20.0, 22.0, 34.0, 49.0, 105.0, 285.0, 461.0, 463.0]),
           st.sampled_from([65.0, 480.0, 780.0, 2312.0, 2640.0, 2730.0, 3555.0, 6192.0, 12390.0, 19278.0]),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.1747, max_value=0.2053, allow_nan=False),
           st.sampled_from([0.207, 0.248, 0.255, 0.371, 0.437, 0.457, 0.533, 0.555, 0.627, 1.0]),
           st.floats(min_value=2.751, max_value=993.2, exclude_min=True, allow_nan=False),
           st.sampled_from([7.0, 23.0, 34.0, 52.0, 162.0, 218.0, 308.0, 370.0, 4213.0, 9199.0]),
           st.sampled_from([42.0, 70.0, 71.0, 100.0, 173.0, 206.0, 250.0, 2588.0, 7279.0, 23092.0]),
           st.floats(min_value=90.51, max_value=714.8, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_152(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_152']['n_samples'] += 1
        self.data['tests']['test_152']['samples'].append(x_test)
        self.data['tests']['test_152']['y_expected'].append(y_expected[0])
        self.data['tests']['test_152']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([95.0, 108.0, 177.0, 189.0, 217.0, 220.0, 263.0, 331.0, 397.0, 433.0]),
           st.floats(min_value=13944.6, max_value=17428.99, allow_nan=False),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2056, max_value=0.2247, exclude_min=True, allow_nan=False),
           st.sampled_from([0.286, 0.51, 0.548, 0.58, 0.695, 0.697, 0.844, 0.847, 0.882, 0.963]),
           st.sampled_from([1.62, 2.35, 4.05, 4.23, 4.63, 5.04, 5.07, 6.11, 6.44, 16.91]),
           st.sampled_from([15.0, 143.0, 186.0, 439.0, 525.0, 692.0, 1032.0, 1212.0, 2144.0, 3459.0]),
           st.sampled_from([104.0, 535.0, 663.0, 700.0, 1222.0, 1455.0, 1625.0, 1654.0, 3131.0, 4548.0]),
           st.sampled_from([62.0, 99.0, 102.0, 168.0, 221.0, 262.0, 354.0, 425.0, 514.0, 564.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_153(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_153']['n_samples'] += 1
        self.data['tests']['test_153']['samples'].append(x_test)
        self.data['tests']['test_153']['y_expected'].append(y_expected[0])
        self.data['tests']['test_153']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([11.0, 63.0, 132.0, 134.0, 178.0, 186.0, 213.0, 297.0, 298.0, 509.0]),
           st.floats(min_value=17429.01, max_value=42741.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2056, max_value=0.2247, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.4751, max_value=0.5783, allow_nan=False),
           st.sampled_from([1.73, 1.98, 3.2, 3.6, 4.81, 4.9, 4.92, 6.07, 7.1, 7.64]),
           st.sampled_from([86.0, 397.0, 465.0, 510.0, 672.0, 742.0, 942.0, 1400.0, 2142.0, 2207.0]),
           st.sampled_from([304.0, 351.0, 626.0, 753.0, 997.0, 1128.0, 1922.0, 2049.0, 2052.0, 2114.0]),
           st.sampled_from([74.0, 82.0, 166.0, 234.0, 276.0, 305.0, 328.0, 331.0, 364.0, 438.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_154(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_154']['n_samples'] += 1
        self.data['tests']['test_154']['samples'].append(x_test)
        self.data['tests']['test_154']['y_expected'].append(y_expected[0])
        self.data['tests']['test_154']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([20.0, 39.0, 47.0, 68.0, 70.0, 71.0, 79.0, 96.0, 167.0, 463.0]),
           st.floats(min_value=17429.01, max_value=42741.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=3.8122, max_value=110.4497, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.2056, max_value=0.2247, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5786, max_value=0.6628, exclude_min=True, allow_nan=False),
           st.sampled_from([1.57, 1.79, 3.21, 3.5, 3.93, 4.5, 4.69, 5.5, 6.27, 46.93]),
           st.sampled_from([12.0, 14.0, 16.0, 20.0, 21.0, 52.0, 181.0, 247.0, 339.0, 1058.0]),
           st.sampled_from([15.0, 24.0, 28.0, 37.0, 70.0, 651.0, 762.0, 951.0, 1958.0, 8387.0]),
           st.sampled_from([29.0, 32.0, 34.0, 36.0, 59.0, 227.0, 259.0, 400.0, 674.0, 839.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_155(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_155']['n_samples'] += 1
        self.data['tests']['test_155']['samples'].append(x_test)
        self.data['tests']['test_155']['y_expected'].append(y_expected[0])
        self.data['tests']['test_155']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([32.0, 50.0, 73.0, 79.0, 135.0, 137.0, 145.0, 149.0, 469.0, 547.0]),
           st.sampled_from([868.0, 3250.0, 4158.0, 4977.0, 8835.0, 9672.0, 12367.0, 22680.0, 26243.0, 26367.0]),
           st.floats(min_value=0.2362, max_value=0.4476, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.5948, max_value=0.7279, allow_nan=False),
           st.floats(min_value=5.828, max_value=7.034, allow_nan=False),
           st.sampled_from([427.0, 431.0, 441.0, 442.0, 1681.0, 2041.0, 10947.0, 17721.0, 27820.0, 33017.0]),
           st.sampled_from([639.0, 2077.0, 5838.0, 5908.0, 15491.0, 16177.0, 23457.0, 23547.0, 25163.0, 46133.0]),
           st.sampled_from([58.0, 60.0, 215.0, 296.0, 1028.0, 1488.0, 1634.0, 1641.0, 1644.0, 2333.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_156(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_156']['n_samples'] += 1
        self.data['tests']['test_156']['samples'].append(x_test)
        self.data['tests']['test_156']['y_expected'].append(y_expected[0])
        self.data['tests']['test_156']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([84.0, 97.0, 128.0, 193.0, 205.0, 213.0, 227.0, 292.0, 299.0, 407.0]),
           st.sampled_from([261.0, 855.0, 960.0, 1086.0, 1134.0, 1870.0, 2004.0, 2414.0, 3888.0, 8064.0]),
           st.floats(min_value=0.2362, max_value=0.4476, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7282, max_value=0.7825, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.828, max_value=7.034, allow_nan=False),
           st.sampled_from([76.0, 198.0, 282.0, 298.0, 430.0, 564.0, 711.0, 873.0, 1046.0, 7366.0]),
           st.sampled_from([233.0, 286.0, 382.0, 590.0, 633.0, 1100.0, 1105.0, 1965.0, 2091.0, 2852.0]),
           st.floats(min_value=819.0, max_value=1023.49, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_157(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_157']['n_samples'] += 1
        self.data['tests']['test_157']['samples'].append(x_test)
        self.data['tests']['test_157']['y_expected'].append(y_expected[0])
        self.data['tests']['test_157']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([32.0, 93.0, 135.0, 137.0, 140.0, 145.0, 158.0, 469.0, 472.0, 547.0]),
           st.sampled_from([868.0, 896.0, 899.0, 4158.0, 6912.0, 8835.0, 12367.0, 22680.0, 25619.0, 78352.0]),
           st.floats(min_value=0.2362, max_value=0.4476, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.7282, max_value=0.7825, exclude_min=True, allow_nan=False),
           st.floats(min_value=5.828, max_value=7.034, allow_nan=False),
           st.sampled_from([401.0, 416.0, 431.0, 441.0, 442.0, 3374.0, 17721.0, 19430.0, 27820.0, 33017.0]),
           st.sampled_from([639.0, 647.0, 665.0, 668.0, 1718.0, 5908.0, 16177.0, 20513.0, 23301.0, 23547.0]),
           st.floats(min_value=1023.51, max_value=1461.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_158(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_158']['n_samples'] += 1
        self.data['tests']['test_158']['samples'].append(x_test)
        self.data['tests']['test_158']['y_expected'].append(y_expected[0])
        self.data['tests']['test_158']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 32.0, 66.0, 93.0, 135.0, 141.0, 158.0, 469.0, 471.0, 472.0]),
           st.sampled_from([899.0, 928.0, 4158.0, 4672.0, 24174.0, 24360.0, 26367.0, 44416.0, 72204.0, 87234.0]),
           st.floats(min_value=0.2362, max_value=0.4476, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.345, 0.472, 0.641, 0.649, 0.722, 0.742, 0.767, 0.884, 0.902, 0.904]),
           st.floats(min_value=7.037, max_value=996.629, exclude_min=True, allow_nan=False),
           st.sampled_from([401.0, 416.0, 431.0, 442.0, 1475.0, 2041.0, 5966.0, 13725.0, 14180.0, 27820.0]),
           st.sampled_from([653.0, 665.0, 668.0, 2458.0, 7968.0, 15491.0, 20513.0, 25400.0, 34874.0, 35499.0]),
           st.sampled_from([6.0, 9.0, 46.0, 60.0, 64.0, 215.0, 786.0, 1028.0, 1481.0, 2333.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_159(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_159']['n_samples'] += 1
        self.data['tests']['test_159']['samples'].append(x_test)
        self.data['tests']['test_159']['y_expected'].append(y_expected[0])
        self.data['tests']['test_159']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 47.0, 81.0, 175.0, 184.0, 263.0, 334.0, 339.0, 385.0, 439.0]),
           st.floats(min_value=4748.6, max_value=5933.99, allow_nan=False),
           st.floats(min_value=1.2936, max_value=1.3959, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.574, 0.621, 0.659, 0.663, 0.679, 0.709, 0.916, 0.942, 0.947, 0.979]),
           st.sampled_from([1.63, 2.15, 2.62, 2.68, 4.0, 4.32, 5.55, 5.86, 12.99, 34.0]),
           st.sampled_from([197.0, 200.0, 308.0, 338.0, 362.0, 512.0, 580.0, 582.0, 611.0, 781.0]),
           st.sampled_from([148.0, 325.0, 336.0, 597.0, 1091.0, 1147.0, 1440.0, 1584.0, 1821.0, 11482.0]),
           st.floats(min_value=755.4, max_value=943.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_160(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_160']['n_samples'] += 1
        self.data['tests']['test_160']['samples'].append(x_test)
        self.data['tests']['test_160']['y_expected'].append(y_expected[0])
        self.data['tests']['test_160']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([1.0, 135.0, 153.0, 189.0, 238.0, 267.0, 285.0, 346.0, 471.0, 520.0]),
           st.floats(min_value=5934.01, max_value=33545.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2936, max_value=1.3959, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.509, 0.595, 0.618, 0.635, 0.671, 0.826, 0.862, 0.874, 0.91, 0.986]),
           st.sampled_from([1.84, 2.45, 3.42, 4.67, 4.71, 5.38, 5.93, 7.58, 8.06, 8.9]),
           st.sampled_from([50.0, 74.0, 104.0, 138.0, 188.0, 554.0, 724.0, 1315.0, 1657.0, 3192.0]),
           st.sampled_from([71.0, 72.0, 209.0, 226.0, 402.0, 424.0, 892.0, 1219.0, 1672.0, 2126.0]),
           st.floats(min_value=82.6, max_value=102.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_161(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_161']['n_samples'] += 1
        self.data['tests']['test_161']['samples'].append(x_test)
        self.data['tests']['test_161']['y_expected'].append(y_expected[0])
        self.data['tests']['test_161']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([32.0, 50.0, 96.0, 104.0, 137.0, 141.0, 161.0, 347.0, 469.0, 471.0]),
           st.floats(min_value=5934.01, max_value=33545.8, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.2936, max_value=1.3959, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.345, 0.445, 0.487, 0.529, 0.591, 0.72, 0.742, 0.743, 0.745, 0.963]),
           st.sampled_from([4.25, 7.47, 13.06, 15.84, 20.66, 21.52, 28.77, 44.37, 46.95, 72.38]),
           st.sampled_from([416.0, 1681.0, 1882.0, 2041.0, 5966.0, 10947.0, 13725.0, 23025.0, 27820.0, 28093.0]),
           st.sampled_from([665.0, 666.0, 1718.0, 2458.0, 20513.0, 23547.0, 25400.0, 35499.0, 42821.0, 46133.0]),
           st.floats(min_value=103.01, max_value=271.2, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_162(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_162']['n_samples'] += 1
        self.data['tests']['test_162']['samples'].append(x_test)
        self.data['tests']['test_162']['y_expected'].append(y_expected[0])
        self.data['tests']['test_162']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([55.0, 185.0, 240.0, 307.0, 346.0, 367.0, 375.0, 398.0, 424.0, 455.0]),
           st.sampled_from([356.0, 561.0, 598.0, 665.0, 1562.0, 2145.0, 2470.0, 3556.0, 4662.0, 12561.0]),
           st.floats(min_value=1.8056, max_value=108.8444, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.52, 0.637, 0.656, 0.682, 0.838, 0.86, 0.863, 0.954, 0.956, 0.984]),
           st.sampled_from([2.76, 3.36, 4.61, 6.04, 6.31, 7.9, 8.55, 8.62, 11.57, 15.9]),
           st.sampled_from([22.0, 41.0, 298.0, 409.0, 466.0, 549.0, 650.0, 837.0, 1848.0, 2053.0]),
           st.sampled_from([20.0, 51.0, 283.0, 450.0, 466.0, 664.0, 667.0, 698.0, 2731.0, 4122.0]),
           st.floats(min_value=755.4, max_value=943.99, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_163(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_163']['n_samples'] += 1
        self.data['tests']['test_163']['samples'].append(x_test)
        self.data['tests']['test_163']['y_expected'].append(y_expected[0])
        self.data['tests']['test_163']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([4.0, 10.0, 11.0, 14.0, 18.0, 20.0, 24.0, 30.0, 43.0, 402.0]),
           st.sampled_from([36.0, 78.0, 84.0, 91.0, 329.0, 1708.0, 2666.0, 19278.0, 25935.0, 39006.0]),
           st.floats(min_value=1.2936, max_value=108.4348, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.165, 0.207, 0.338, 0.436, 0.484, 0.526, 0.538, 0.633, 0.65, 0.655]),
           st.floats(min_value=7.915, max_value=9.643, allow_nan=False),
           st.sampled_from([13.0, 16.0, 20.0, 97.0, 162.0, 350.0, 375.0, 642.0, 969.0, 7689.0]),
           st.sampled_from([9.0, 31.0, 50.0, 156.0, 163.0, 755.0, 951.0, 1814.0, 4497.0, 10347.0]),
           st.floats(min_value=944.01, max_value=1397.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_164(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [5]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_164']['n_samples'] += 1
        self.data['tests']['test_164']['samples'].append(x_test)
        self.data['tests']['test_164']['y_expected'].append(y_expected[0])
        self.data['tests']['test_164']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=27.51, max_value=182.8, exclude_min=True, allow_nan=False),
           st.sampled_from([32.0, 66.0, 96.0, 104.0, 137.0, 149.0, 161.0, 347.0, 471.0, 472.0]),
           st.sampled_from([899.0, 928.0, 4158.0, 6912.0, 8835.0, 24360.0, 26367.0, 44416.0, 78352.0, 87234.0]),
           st.floats(min_value=1.2936, max_value=108.4348, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.3016, max_value=0.4412, exclude_min=True, allow_nan=False),
           st.sampled_from([0.433, 0.472, 0.567, 0.582, 0.726, 0.743, 0.884, 0.904, 0.963, 0.991]),
           st.floats(min_value=9.646, max_value=998.716, exclude_min=True, allow_nan=False),
           st.sampled_from([401.0, 414.0, 427.0, 437.0, 1681.0, 1882.0, 10947.0, 14180.0, 17721.0, 28093.0]),
           st.sampled_from([666.0, 1718.0, 5838.0, 5908.0, 16177.0, 20513.0, 23547.0, 25400.0, 42821.0, 46133.0]),
           st.floats(min_value=944.01, max_value=1397.6, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_165(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
        y_expected = [3]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_165']['n_samples'] += 1
        self.data['tests']['test_165']['samples'].append(x_test)
        self.data['tests']['test_165']['y_expected'].append(y_expected[0])
        self.data['tests']['test_165']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
