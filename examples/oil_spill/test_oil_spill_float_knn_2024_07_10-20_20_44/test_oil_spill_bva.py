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
    request.cls.data['n_test'] = 35
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_oil_spill_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestOilSpillProperty:

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([13.0, 29.0, 57.0, 60.0, 128.0, 162.0, 183.0, 270.0, 735.0, 2971.0]),
           st.sampled_from([115.22, 165.49, 221.79, 231.84, 260.19, 263.09, 1127.41, 1446.29, 1449.85, 1506.09]),
           st.sampled_from([242.59, 287.06, 295.65, 308.71, 338.47, 365.69, 546.0, 580.94, 640.12, 1530.1]),
           st.sampled_from([10.0, 18.0, 23.0, 62.0, 73.0, 90.0, 96.0, 132.0, 140.0, 161.0]),
           st.sampled_from([129600.0, 137700.0, 147500.0, 177500.0, 287500.0, 290000.0, 497812.0, 793800.0, 2616300.0, 5685000.0]),
           st.sampled_from([23.92, 24.65, 27.23, 40.88, 41.08, 42.4, 44.06, 44.67, 47.81, 54.2]),
           st.sampled_from([3.66, 5.42, 7.08, 7.36, 7.97, 7.99, 12.73, 14.9, 16.89, 18.6]),
           st.sampled_from([1011.0, 1318.0, 1461.0, 1800.0, 2000.0, 2370.0, 3483.0, 3720.0, 9294.0, 22260.0]),
           st.sampled_from([0.09, 0.11, 0.18, 0.23, 0.24, 0.26, 0.27, 0.3, 0.32, 0.34]),
           st.sampled_from([67.6, 79.2, 81.9, 101.3, 103.8, 104.2, 121.9, 125.0, 298.3, 427.4]),
           st.sampled_from([0.19, 0.2, 0.21, 0.22, 0.24, 0.26, 0.28, 0.29, 0.35, 0.36]),
           st.sampled_from([0.2, 0.26, 0.27, 0.3, 0.32, 0.33, 0.34, 0.37, 0.39, 0.79]),
           st.sampled_from([0.34, 0.36, 0.48, 0.49, 0.63, 0.64, 0.85, 0.95, 0.96, 1.02]),
           st.sampled_from([0.08, 0.09, 0.15, 0.18, 0.2, 0.23, 0.24, 0.27, 0.32, 0.5]),
           st.floats(min_value=0.133, max_value=0.163, allow_nan=False),
           st.sampled_from([11.66, 20.47, 20.73, 21.63, 24.48, 37.51, 93.32, 136.77, 180.8, 227.35]),
           st.sampled_from([7.28, 7.3, 12.15, 12.55, 14.45, 22.56, 24.47, 34.76, 50.99, 54.12]),
           st.sampled_from([0.32, 0.33, 0.36, 0.62, 0.74, 0.82, 0.9, 0.94, 1.82, 2.14]),
           st.sampled_from([0.1, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.31, 0.34]),
           st.sampled_from([0.19, 0.2, 0.21, 0.26, 0.29, 0.31, 0.34, 0.37, 0.4, 0.42]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.19, 0.22, 0.4, 0.42, 0.44, 0.45, 0.81, 0.86, 1.02, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.51, 0.21, 0.47, 0.55, 0.69, 0.74, 0.85, 1.01, 1.08, 1.22]),
           st.sampled_from([1.64, 1.86, 3.2, 3.28, 4.36, 6.3, 6.78, 6.81, 7.06, 7.31]),
           st.sampled_from([-6.77, -3.89, -3.71, -3.38, -3.24, -2.24, -2.02, -1.63, -1.27, -0.65]),
           st.sampled_from([-1.02, -0.89, -0.75, -0.36, -0.32, -0.21, -0.16, -0.14, -0.12, -0.09]),
           st.sampled_from([1.09, 1.22, 1.94, 1.95, 1.96, 2.18, 2.2, 2.59, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.23, 1.95, 2.16, 2.17, 2.19, 2.2, 2.91, 2.92]),
           st.sampled_from([12.0, 15.0, 26.0, 30.0, 43.0, 53.0, 61.0, 96.0, 183.0, 310.0]),
           st.sampled_from([450.0, 1530.0, 1620.0, 1800.0, 1890.0, 2700.0, 2790.0, 10080.0, 10350.0, 16110.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([9.71, 13.07, 16.0, 16.37, 24.34, 24.37, 28.07, 29.44, 34.35, 46.19]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([851.47, 1060.66, 1288.6, 1400.0, 2022.5, 2704.16, 2850.0, 3471.31, 4724.79, 5650.88]),
           st.sampled_from([90.0, 200.0, 212.13, 284.6, 403.11, 492.44, 829.76, 853.81, 1612.45, 1659.52]),
           st.sampled_from([65.27, 135.0, 150.0, 176.72, 193.18, 225.0, 256.77, 268.75, 345.47, 526.56]),
           st.sampled_from([0.0, 41.98, 45.13, 55.93, 73.11, 88.6, 145.92, 249.81, 348.7, 349.26]),
           st.sampled_from([1.45, 1.67, 1.7, 2.58, 2.85, 4.54, 6.63, 6.95, 7.74, 9.33]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=10332.764, max_value=12403.079, allow_nan=False),
           st.sampled_from([36.14, 65.58, 65.73, 65.74, 65.78, 65.79, 65.91, 65.93, 65.97, 66.04]),
           st.sampled_from([5.94, 6.96, 7.18, 7.32, 7.36, 7.41, 7.55, 7.58, 15.01, 15.04]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([28.0, 66.0, 92.0, 112.0, 124.0, 148.0, 229.0, 256.0, 851.0, 32389.0]),
           st.sampled_from([25.1, 25.62, 64.45, 322.81, 334.39, 797.99, 1023.97, 1664.47, 1707.7, 1708.19]),
           st.sampled_from([232.47, 412.77, 520.4, 588.36, 618.22, 659.15, 946.09, 1228.76, 1558.81, 1892.56]),
           st.sampled_from([7.0, 29.0, 45.0, 66.0, 67.0, 84.0, 102.0, 137.0, 148.0, 175.0]),
           st.sampled_from([153900.0, 230625.0, 247500.0, 283500.0, 300000.0, 460000.0, 642500.0, 827500.0, 1287900.0, 1489218.0]),
           st.sampled_from([22.96, 28.36, 38.73, 41.91, 48.08, 48.27, 51.7, 54.75, 57.14, 58.13]),
           st.sampled_from([4.6, 6.26, 7.14, 8.39, 9.08, 9.11, 10.44, 12.99, 13.86, 14.65]),
           st.sampled_from([741.0, 1590.0, 1642.0, 2570.0, 3377.0, 3660.0, 5360.0, 9433.0, 9707.0, 62250.0]),
           st.sampled_from([0.03, 0.12, 0.13, 0.24, 0.29, 0.38, 0.44, 0.51, 0.52, 0.74]),
           st.sampled_from([64.1, 79.1, 81.8, 93.7, 96.3, 100.0, 112.2, 113.4, 114.8, 122.3]),
           st.sampled_from([0.05, 0.13, 0.26, 0.3, 0.31, 0.35, 0.38, 0.4, 0.44, 0.47]),
           st.sampled_from([0.04, 0.05, 0.06, 0.19, 0.3, 0.31, 0.48, 0.52, 0.69, 0.76]),
           st.sampled_from([0.06, 0.1, 0.33, 0.38, 0.39, 0.44, 0.52, 0.82, 0.9, 1.08]),
           st.sampled_from([0.01, 0.12, 0.15, 0.25, 0.31, 0.37, 0.42, 0.47, 0.51, 0.58]),
           st.floats(min_value=0.166, max_value=0.356, exclude_min=True, allow_nan=False),
           st.sampled_from([17.48, 17.66, 22.11, 32.61, 40.94, 51.06, 63.13, 92.08, 126.93, 139.5]),
           st.sampled_from([15.31, 15.71, 17.83, 21.17, 21.71, 23.12, 24.8, 31.77, 42.82, 49.36]),
           st.sampled_from([0.31, 0.49, 0.65, 0.94, 1.82, 1.87, 2.13, 2.15, 2.19, 2.45]),
           st.sampled_from([0.02, 0.05, 0.11, 0.16, 0.18, 0.36, 0.4, 0.41, 0.43, 0.58]),
           st.sampled_from([0.12, 0.13, 0.17, 0.28, 0.34, 0.42, 0.43, 0.63, 0.68, 0.77]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.47, 0.53, 0.57, 0.76, 0.8, 0.82, 0.91, 0.99, 1.09]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.8, -0.67, -0.08, -0.05, 0.55, 0.61, 0.75, 1.49, 1.81, 2.01]),
           st.sampled_from([1.54, 1.9, 2.39, 2.56, 3.28, 4.12, 5.03, 5.62, 5.98, 7.73]),
           st.sampled_from([-7.06, -2.76, -2.68, -2.53, -2.41, -1.98, -1.05, -0.88, -0.61, 0.92]),
           st.sampled_from([-0.96, -0.9, -0.87, -0.51, -0.38, -0.33, -0.26, -0.25, -0.24, -0.07]),
           st.sampled_from([1.11, 1.97, 1.99, 2.18, 2.2, 2.23, 2.67, 2.91, 2.96, 2.97]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.23, 0.36, 1.11, 1.23, 1.93, 1.94, 1.99, 2.01, 2.23, 2.59]),
           st.sampled_from([6.0, 13.0, 20.0, 32.0, 98.0, 99.0, 160.0, 324.0, 476.0, 619.0]),
           st.sampled_from([810.0, 1620.0, 1800.0, 3690.0, 5040.0, 5400.0, 12870.0, 17280.0, 19710.0, 20700.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([9.32, 11.51, 16.04, 25.83, 27.46, 29.19, 30.38, 31.36, 35.08, 93.27]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([400.0, 670.82, 956.07, 1140.18, 1272.79, 1793.24, 2008.11, 3170.57, 3623.55, 3959.8]),
           st.sampled_from([53.03, 127.28, 141.42, 180.0, 254.95, 282.84, 424.26, 494.97, 1030.78, 1844.59]),
           st.sampled_from([64.29, 94.25, 114.05, 115.81, 148.32, 181.82, 225.0, 250.29, 370.16, 4531.14]),
           st.sampled_from([52.36, 69.13, 74.63, 90.13, 120.0, 134.49, 136.95, 221.24, 302.87, 474.04]),
           st.sampled_from([1.91, 2.35, 3.57, 3.77, 3.82, 5.11, 6.14, 6.88, 8.07, 12.91]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=10332.764, max_value=12403.079, allow_nan=False),
           st.sampled_from([36.18, 36.39, 36.47, 36.62, 36.86, 65.45, 65.55, 65.61, 65.63, 66.03]),
           st.sampled_from([5.98, 6.0, 6.33, 6.52, 6.54, 7.24, 7.66, 7.93, 15.0, 15.31]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([69.0, 105.0, 140.0, 194.0, 346.0, 670.0, 1574.0, 4851.0, 7232.0, 28526.0]),
           st.sampled_from([13.33, 31.03, 34.36, 56.31, 77.4, 154.64, 635.24, 1290.31, 1404.46, 1697.23]),
           st.sampled_from([264.07, 315.16, 473.91, 553.73, 1041.51, 1122.96, 1151.03, 1181.57, 1542.32, 1622.32]),
           st.sampled_from([57.0, 76.0, 78.0, 97.0, 109.0, 113.0, 115.0, 133.0, 145.0, 150.0]),
           st.sampled_from([129600.0, 137500.0, 137812.0, 296718.0, 352500.0, 397500.0, 413437.0, 672500.0, 6660000.0, 11020000.0]),
           st.sampled_from([23.51, 26.78, 29.51, 32.89, 38.96, 39.46, 46.31, 50.47, 54.58, 71.33]),
           st.sampled_from([6.25, 8.43, 8.92, 9.69, 9.7, 10.23, 10.42, 12.21, 12.42, 18.01]),
           st.sampled_from([1662.0, 2276.0, 2680.0, 2880.0, 2979.0, 3150.0, 4390.0, 4560.0, 4800.0, 8792.0]),
           st.sampled_from([0.06, 0.07, 0.17, 0.22, 0.32, 0.38, 0.4, 0.44, 0.52, 0.54]),
           st.floats(min_value=193.47, max_value=231.58, allow_nan=False),
           st.sampled_from([0.02, 0.05, 0.11, 0.15, 0.17, 0.22, 0.26, 0.33, 0.53, 0.62]),
           st.sampled_from([0.19, 0.2, 0.21, 0.28, 0.31, 0.39, 0.44, 0.45, 0.5, 0.53]),
           st.sampled_from([0.15, 0.18, 0.43, 0.78, 0.84, 0.91, 0.99, 1.03, 1.09, 1.19]),
           st.sampled_from([0.1, 0.17, 0.19, 0.22, 0.3, 0.36, 0.37, 0.44, 0.52, 0.53]),
           st.sampled_from([0.02, 0.1, 0.16, 0.21, 0.35, 0.37, 0.62, 0.64, 0.85, 0.97]),
           st.sampled_from([11.13, 26.87, 29.92, 41.65, 47.18, 62.57, 68.6, 88.87, 93.41, 122.64]),
           st.sampled_from([2.62, 6.13, 10.14, 11.87, 12.11, 12.27, 13.8, 20.55, 27.89, 56.8]),
           st.floats(min_value=0.282, max_value=0.319, allow_nan=False),
           st.sampled_from([0.02, 0.11, 0.16, 0.19, 0.23, 0.28, 0.32, 0.34, 0.38, 0.55]),
           st.sampled_from([0.17, 0.19, 0.22, 0.23, 0.32, 0.39, 0.48, 0.49, 0.51, 0.76]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.21, 0.26, 0.6, 0.78, 0.89, 0.95, 1.02, 1.09, 1.11, 1.33]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.57, -0.1, 0.16, 0.29, 0.51, 0.58, 0.7, 1.93, 3.13, 4.03]),
           st.sampled_from([1.45, 2.27, 2.96, 3.06, 3.31, 3.59, 3.96, 3.97, 4.06, 10.28]),
           st.sampled_from([-5.79, -5.51, -5.34, -4.03, -3.88, -2.07, -1.82, -1.61, -1.37, -0.85]),
           st.sampled_from([-1.22, -0.87, -0.74, -0.62, -0.57, -0.48, -0.44, -0.22, -0.11, -0.08]),
           st.sampled_from([1.21, 1.22, 1.93, 2.16, 2.19, 2.21, 2.23, 2.61, 2.64, 2.93]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.36, 1.21, 1.94, 1.95, 2.16, 2.19, 2.2, 2.21, 2.64, 2.98]),
           st.sampled_from([17.0, 38.0, 67.0, 82.0, 92.0, 104.0, 141.0, 161.0, 623.0, 924.0]),
           st.sampled_from([1080.0, 1440.0, 2790.0, 3420.0, 3690.0, 3960.0, 7650.0, 13410.0, 25470.0, 33390.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([9.38, 9.6, 14.39, 15.5, 16.62, 23.96, 33.7, 33.95, 35.0, 42.99]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([742.16, 772.17, 860.23, 901.56, 926.61, 1303.84, 1320.04, 1629.97, 2205.11, 3889.09]),
           st.sampled_from([316.23, 360.56, 380.79, 414.2, 484.66, 540.0, 680.07, 970.82, 1509.81, 2050.61]),
           st.sampled_from([89.32, 147.58, 184.63, 243.17, 272.82, 284.09, 305.5, 368.93, 512.54, 792.95]),
           st.sampled_from([34.02, 55.24, 84.41, 99.78, 105.57, 129.37, 151.41, 153.84, 198.29, 233.65]),
           st.sampled_from([0.39, 2.12, 3.11, 3.59, 5.19, 6.61, 7.07, 10.31, 15.18, 16.67]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=12403.082, max_value=14592.124, exclude_min=True, allow_nan=False),
           st.sampled_from([36.0, 36.33, 36.64, 36.84, 65.59, 65.71, 65.8, 65.82, 66.03, 66.33]),
           st.sampled_from([6.26, 6.31, 6.56, 6.97, 7.28, 7.44, 7.88, 7.94, 8.03, 14.82]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([13.0, 29.0, 64.0, 71.0, 98.0, 109.0, 115.0, 183.0, 453.0, 2274.0]),
           st.sampled_from([7.7, 21.23, 115.22, 221.79, 226.81, 361.68, 389.76, 454.66, 920.2, 1123.09]),
           st.sampled_from([287.42, 305.85, 333.23, 456.23, 491.78, 544.91, 546.0, 1184.91, 1494.86, 2632.39]),
           st.sampled_from([18.0, 23.0, 42.0, 62.0, 68.0, 88.0, 110.0, 114.0, 132.0, 161.0]),
           st.sampled_from([135000.0, 137700.0, 542700.0, 793800.0, 1132500.0, 1312200.0, 2948400.0, 3002500.0, 6395000.0, 7887500.0]),
           st.sampled_from([23.11, 23.92, 24.09, 29.28, 38.65, 38.9, 41.53, 42.4, 47.59, 54.2]),
           st.sampled_from([5.46, 5.94, 6.13, 6.44, 7.08, 7.9, 7.97, 8.79, 10.48, 16.85]),
           st.sampled_from([1145.5, 1461.0, 1551.5, 1710.0, 3610.0, 6998.0, 7067.0, 8770.0, 15106.0, 32401.5]),
           st.sampled_from([0.08, 0.1, 0.15, 0.18, 0.19, 0.23, 0.26, 0.28, 0.3, 0.34]),
           st.floats(min_value=231.61, max_value=365.62, exclude_min=True, allow_nan=False),
           st.sampled_from([0.11, 0.19, 0.2, 0.21, 0.22, 0.26, 0.29, 0.34, 0.35, 0.52]),
           st.sampled_from([0.16, 0.2, 0.21, 0.25, 0.27, 0.28, 0.32, 0.33, 0.35, 0.38]),
           st.sampled_from([0.24, 0.3, 0.33, 0.34, 0.35, 0.42, 0.46, 0.48, 0.5, 0.65]),
           st.sampled_from([0.07, 0.09, 0.1, 0.15, 0.17, 0.18, 0.2, 0.21, 0.23, 0.25]),
           st.sampled_from([0.16, 0.21, 0.25, 0.31, 0.33, 0.37, 0.44, 0.6, 0.63, 0.87]),
           st.sampled_from([19.85, 20.38, 26.08, 26.35, 58.3, 64.34, 71.2, 118.11, 120.22, 235.92]),
           st.sampled_from([8.09, 9.56, 9.62, 11.96, 13.31, 19.27, 29.21, 32.19, 34.76, 43.96]),
           st.floats(min_value=0.282, max_value=0.319, allow_nan=False),
           st.sampled_from([0.12, 0.15, 0.17, 0.19, 0.22, 0.23, 0.25, 0.29, 0.32, 0.34]),
           st.sampled_from([0.13, 0.14, 0.18, 0.19, 0.23, 0.24, 0.25, 0.26, 0.27, 0.4]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.42, 0.45, 0.46, 0.75, 0.81, 0.86, 0.9, 0.91, 1.02]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.51, 0.12, 0.22, 0.55, 0.57, 0.97, 1.11, 1.55, 1.61, 1.98]),
           st.sampled_from([2.66, 2.95, 3.2, 3.28, 3.52, 3.56, 4.06, 6.78, 6.81, 8.68]),
           st.sampled_from([-5.55, -3.68, -3.53, -3.5, -3.38, -3.19, -3.1, -2.59, -2.5, -0.87]),
           st.sampled_from([-0.75, -0.63, -0.53, -0.52, -0.5, -0.28, -0.25, -0.18, -0.17, -0.07]),
           st.sampled_from([1.09, 1.94, 1.95, 1.96, 2.16, 2.17, 2.18, 2.19, 2.2, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.22, 1.94, 1.95, 2.16, 2.18, 2.19, 2.59, 2.91, 2.92]),
           st.sampled_from([11.0, 12.0, 18.0, 19.0, 21.0, 34.0, 43.0, 46.0, 48.0, 133.0]),
           st.sampled_from([1170.0, 1620.0, 2250.0, 2700.0, 2790.0, 3060.0, 4140.0, 8730.0, 10350.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([13.07, 14.7, 16.37, 18.24, 23.4, 28.07, 29.4, 34.35, 61.69, 78.14]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([403.89, 742.16, 768.96, 960.47, 1569.24, 2022.5, 2672.86, 2846.05, 2930.19, 4724.79]),
           st.sampled_from([141.42, 269.26, 300.0, 320.4, 492.44, 603.74, 721.11, 1000.0, 1612.45, 2189.79]),
           st.sampled_from([65.27, 67.5, 89.57, 150.0, 190.67, 345.47, 474.05, 526.56, 751.79, 2043.9]),
           st.sampled_from([44.75, 47.73, 65.25, 114.82, 124.06, 133.49, 135.46, 179.03, 348.7, 349.26]),
           st.sampled_from([1.47, 1.71, 2.58, 2.85, 4.0, 5.88, 6.32, 6.44, 10.72, 13.05]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=12403.082, max_value=14592.124, exclude_min=True, allow_nan=False),
           st.sampled_from([65.61, 65.67, 65.72, 65.73, 65.79, 65.8, 65.81, 65.91, 65.97, 66.04]),
           st.sampled_from([6.42, 7.18, 7.32, 7.35, 7.39, 7.55, 7.84, 8.07, 14.45, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.0, max_value=3.49, allow_nan=False),
           st.sampled_from([63.0, 67.0, 107.0, 116.0, 162.0, 323.0, 2558.0, 2971.0, 3155.0, 9264.0]),
           st.sampled_from([104.75, 231.84, 263.09, 343.63, 454.66, 714.46, 852.49, 920.2, 1020.91, 1118.08]),
           st.sampled_from([292.22, 295.65, 303.14, 456.23, 566.96, 580.94, 608.43, 1062.68, 1165.76, 1184.91]),
           st.sampled_from([10.0, 11.0, 56.0, 59.0, 60.0, 69.0, 73.0, 82.0, 90.0, 155.0]),
           st.sampled_from([73125.0, 80156.0, 125000.0, 135000.0, 137700.0, 142500.0, 234900.0, 290000.0, 320000.0, 510300.0]),
           st.sampled_from([24.65, 40.88, 41.22, 41.53, 44.06, 44.67, 46.29, 52.11, 53.69, 70.65]),
           st.sampled_from([4.33, 5.01, 5.42, 5.46, 8.2, 9.31, 11.38, 12.73, 13.54, 13.56]),
           st.sampled_from([1570.0, 1710.0, 1800.0, 1858.0, 2280.0, 3483.0, 3660.0, 4748.0, 6998.0, 32401.5]),
           st.sampled_from([0.08, 0.09, 0.1, 0.11, 0.15, 0.18, 0.27, 0.28, 0.3, 0.32]),
           st.sampled_from([71.2, 81.9, 86.5, 101.3, 101.6, 103.8, 104.5, 142.9, 159.5, 276.0]),
           st.sampled_from([0.18, 0.2, 0.23, 0.24, 0.25, 0.26, 0.33, 0.34, 0.36, 0.43]),
           st.sampled_from([0.21, 0.24, 0.25, 0.26, 0.27, 0.28, 0.3, 0.33, 0.41, 0.53]),
           st.sampled_from([0.23, 0.28, 0.34, 0.38, 0.39, 0.5, 0.56, 0.69, 0.74, 0.95]),
           st.sampled_from([0.08, 0.12, 0.13, 0.15, 0.17, 0.18, 0.2, 0.23, 0.3, 0.5]),
           st.sampled_from([0.16, 0.2, 0.21, 0.29, 0.35, 0.37, 0.4, 0.44, 0.54, 0.87]),
           st.sampled_from([18.19, 20.47, 24.37, 28.58, 29.43, 58.3, 64.34, 125.35, 136.77, 431.19]),
           st.sampled_from([7.3, 8.32, 10.98, 11.96, 34.76, 39.88, 49.55, 50.37, 54.12, 82.87]),
           st.floats(min_value=0.322, max_value=0.777, exclude_min=True, allow_nan=False),
           st.sampled_from([0.14, 0.16, 0.17, 0.19, 0.21, 0.24, 0.25, 0.28, 0.29, 0.33]),
           st.sampled_from([0.13, 0.14, 0.19, 0.24, 0.25, 0.26, 0.27, 0.32, 0.42, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.36, 0.39, 0.4, 0.48, 0.5, 0.91, 0.92, 1.02, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.12, 0.47, 0.74, 0.84, 0.89, 0.97, 1.08, 1.11, 1.44, 1.98]),
           st.sampled_from([2.34, 2.64, 2.95, 3.34, 3.52, 3.84, 4.36, 5.76, 5.99, 7.31]),
           st.sampled_from([-3.89, -3.71, -3.64, -3.53, -3.5, -2.68, -2.5, -2.25, -2.24, -1.27]),
           st.sampled_from([-0.89, -0.63, -0.52, -0.51, -0.34, -0.33, -0.25, -0.18, -0.17, -0.16]),
           st.sampled_from([1.09, 1.1, 1.23, 1.94, 1.95, 2.16, 2.18, 2.2, 2.59, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.94, 2.17, 2.18, 2.19, 2.2, 2.59, 2.91, 2.92]),
           st.sampled_from([6.0, 11.0, 15.0, 20.0, 21.0, 26.0, 28.0, 29.0, 96.0, 465.0]),
           st.sampled_from([630.0, 1080.0, 1530.0, 2610.0, 2790.0, 5490.0, 6750.0, 8100.0, 10350.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([13.07, 14.7, 19.31, 24.34, 27.38, 29.44, 31.97, 40.95, 80.59, 108.27]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([538.52, 540.0, 650.0, 707.11, 851.47, 1288.6, 1569.24, 2101.07, 2850.0, 5650.88]),
           st.sampled_from([90.0, 269.26, 284.6, 320.4, 360.0, 492.44, 655.21, 813.94, 853.81, 1612.45]),
           st.sampled_from([182.83, 193.18, 225.0, 268.75, 291.6, 334.28, 368.74, 526.56, 1245.07, 2043.9]),
           st.sampled_from([41.66, 45.13, 60.97, 61.45, 65.25, 73.11, 88.6, 124.72, 135.46, 179.03]),
           st.sampled_from([2.64, 2.7, 4.06, 4.54, 4.6, 5.88, 7.33, 11.32, 11.89, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=12403.082, max_value=14592.124, exclude_min=True, allow_nan=False),
           st.sampled_from([36.18, 36.36, 65.61, 65.66, 65.72, 65.8, 65.82, 65.93, 66.04, 66.33]),
           st.sampled_from([6.07, 6.16, 6.3, 6.46, 7.18, 7.26, 7.29, 7.39, 7.95, 15.04]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 17.0, 63.0, 84.0, 95.0, 96.0, 183.0, 1201.0, 2274.0, 3155.0]),
           st.sampled_from([115.22, 165.49, 327.33, 335.02, 474.38, 497.9, 714.46, 852.49, 920.2, 1562.53]),
           st.sampled_from([287.42, 305.85, 308.71, 365.69, 469.39, 544.91, 624.0, 1337.49, 1494.86, 1507.76]),
           st.sampled_from([10.0, 18.0, 22.0, 41.0, 42.0, 62.0, 90.0, 124.0, 140.0, 155.0]),
           st.sampled_from([118125.0, 142500.0, 147500.0, 150468.0, 177500.0, 251100.0, 285000.0, 769500.0, 2616300.0, 6395000.0]),
           st.sampled_from([23.11, 23.92, 24.09, 38.65, 41.53, 42.37, 43.9, 49.35, 50.33, 54.2]),
           st.sampled_from([3.95, 4.33, 6.24, 6.44, 6.92, 7.9, 11.54, 13.56, 13.97, 16.89]),
           st.sampled_from([1400.0, 1461.0, 1570.0, 1710.0, 1800.0, 2000.0, 3660.0, 3720.0, 7067.0, 22260.0]),
           st.floats(min_value=0.095, max_value=0.113, allow_nan=False),
           st.sampled_from([83.0, 86.1, 97.0, 104.5, 113.4, 152.4, 162.1, 166.5, 195.2, 298.3]),
           st.sampled_from([0.13, 0.15, 0.19, 0.22, 0.23, 0.25, 0.29, 0.33, 0.34, 0.43]),
           st.sampled_from([0.2, 0.22, 0.24, 0.25, 0.27, 0.33, 0.39, 0.53, 0.67, 0.79]),
           st.sampled_from([0.23, 0.24, 0.36, 0.39, 0.42, 0.62, 0.63, 0.66, 0.96, 1.02]),
           st.sampled_from([0.09, 0.13, 0.14, 0.15, 0.17, 0.18, 0.2, 0.23, 0.27, 0.32]),
           st.sampled_from([0.15, 0.16, 0.19, 0.21, 0.25, 0.29, 0.33, 0.4, 0.54, 0.63]),
           st.sampled_from([16.06, 20.47, 20.73, 21.74, 26.08, 29.0, 55.25, 125.35, 207.31, 747.64]),
           st.sampled_from([8.32, 9.14, 9.22, 12.15, 16.73, 19.62, 22.56, 29.21, 43.96, 59.38]),
           st.sampled_from([0.33, 0.38, 0.82, 0.84, 0.88, 0.9, 0.97, 1.06, 1.84, 1.97]),
           st.sampled_from([0.12, 0.15, 0.22, 0.23, 0.25, 0.27, 0.28, 0.31, 0.32, 0.34]),
           st.sampled_from([0.14, 0.18, 0.22, 0.23, 0.25, 0.26, 0.31, 0.42, 0.43, 0.49]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.22, 0.36, 0.44, 0.46, 0.75, 0.86, 0.91, 0.92, 0.97, 1.02]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.12, 0.26, 0.55, 0.7, 0.85, 1.11, 1.47, 1.98, 1.99, 2.5]),
           st.sampled_from([2.33, 2.34, 2.64, 3.2, 3.21, 3.34, 3.79, 4.02, 6.78, 11.08]),
           st.sampled_from([-6.01, -4.78, -3.62, -3.19, -3.1, -2.59, -1.95, -1.92, -1.27, -0.65]),
           st.sampled_from([-1.02, -0.89, -0.75, -0.59, -0.41, -0.3, -0.23, -0.22, -0.17, -0.09]),
           st.sampled_from([1.09, 1.1, 1.23, 1.94, 1.96, 2.17, 2.18, 2.19, 2.2, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.22, 1.23, 1.94, 1.95, 1.96, 2.18, 2.19, 2.2, 2.59, 2.92]),
           st.sampled_from([11.0, 12.0, 13.0, 21.0, 34.0, 36.0, 43.0, 48.0, 61.0, 202.0]),
           st.sampled_from([450.0, 630.0, 720.0, 900.0, 2160.0, 2610.0, 3060.0, 4140.0, 6750.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.6, 16.37, 16.47, 18.24, 24.37, 28.07, 29.3, 29.4, 34.35, 46.19]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([650.0, 685.42, 768.96, 851.47, 1060.66, 1281.6, 1400.0, 1569.24, 2022.5, 2534.42]),
           st.sampled_from([90.0, 300.0, 316.23, 320.4, 492.44, 636.4, 813.94, 1170.0, 1612.45, 2189.79]),
           st.sampled_from([150.0, 190.67, 193.18, 236.27, 268.75, 368.74, 526.56, 751.79, 763.16, 1209.38]),
           st.sampled_from([0.0, 41.66, 61.45, 84.35, 85.79, 124.72, 133.49, 144.97, 255.81, 477.23]),
           st.sampled_from([1.37, 1.45, 1.71, 2.58, 2.64, 3.73, 3.9, 4.06, 6.32, 13.33]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.49, 36.5, 65.61, 65.72, 65.79, 65.82, 65.84, 65.91, 66.04, 66.06]),
           st.sampled_from([6.42, 6.96, 7.22, 7.28, 7.32, 7.55, 7.65, 8.07, 14.45, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 50.0, 67.0, 71.0, 107.0, 128.0, 162.0, 183.0, 364.0, 3155.0]),
           st.sampled_from([115.22, 260.19, 343.63, 361.68, 369.85, 441.43, 454.66, 1118.08, 1123.09, 1449.85]),
           st.sampled_from([295.65, 305.85, 308.71, 333.23, 456.63, 544.91, 546.0, 1062.68, 1165.76, 1167.85]),
           st.sampled_from([11.0, 18.0, 22.0, 51.0, 62.0, 66.0, 68.0, 69.0, 132.0, 135.0]),
           st.floats(min_value=111312.4, max_value=121562.49, allow_nan=False),
           st.sampled_from([24.09, 24.65, 30.41, 38.65, 40.88, 42.37, 44.67, 49.35, 52.11, 60.03]),
           st.sampled_from([3.66, 4.68, 4.83, 5.46, 6.13, 6.44, 7.36, 8.2, 8.79, 10.48]),
           st.sampled_from([1011.0, 1130.0, 1461.0, 1800.0, 2370.0, 5362.0, 6998.0, 7067.0, 21569.0, 22260.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([67.6, 88.4, 97.0, 101.6, 113.4, 124.9, 214.7, 244.7, 255.4, 427.4]),
           st.sampled_from([0.11, 0.19, 0.21, 0.22, 0.23, 0.29, 0.31, 0.35, 0.36, 0.49]),
           st.sampled_from([0.2, 0.22, 0.24, 0.27, 0.29, 0.3, 0.34, 0.35, 0.37, 0.67]),
           st.sampled_from([0.33, 0.39, 0.42, 0.46, 0.5, 0.53, 0.56, 0.69, 0.74, 0.95]),
           st.sampled_from([0.07, 0.08, 0.12, 0.13, 0.15, 0.23, 0.25, 0.3, 0.32, 0.5]),
           st.sampled_from([0.08, 0.14, 0.16, 0.2, 0.29, 0.3, 0.31, 0.4, 0.42, 0.64]),
           st.sampled_from([28.58, 29.0, 29.43, 37.51, 55.25, 56.02, 165.27, 203.68, 235.92, 431.19]),
           st.sampled_from([9.56, 13.31, 19.93, 29.21, 34.76, 49.55, 50.37, 50.99, 54.12, 82.87]),
           st.sampled_from([0.34, 0.36, 0.38, 0.44, 0.78, 0.97, 1.89, 1.97, 2.01, 2.14]),
           st.sampled_from([0.1, 0.17, 0.21, 0.22, 0.26, 0.27, 0.31, 0.32, 0.33, 0.34]),
           st.sampled_from([0.13, 0.14, 0.22, 0.23, 0.26, 0.32, 0.37, 0.42, 0.43, 0.52]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.22, 0.42, 0.44, 0.47, 0.48, 0.51, 0.81, 0.87, 0.97]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.11, 0.21, 0.29, 0.56, 0.59, 0.69, 0.84, 0.85, 1.08, 1.48]),
           st.sampled_from([1.64, 1.98, 2.33, 3.21, 3.34, 3.44, 3.84, 6.78, 7.06, 11.08]),
           st.sampled_from([-5.18, -3.89, -3.71, -3.5, -3.07, -2.59, -2.24, -2.02, -1.95, -0.65]),
           st.sampled_from([-0.59, -0.52, -0.41, -0.36, -0.32, -0.3, -0.28, -0.25, -0.23, -0.22]),
           st.sampled_from([1.09, 1.22, 1.23, 1.95, 1.96, 2.17, 2.18, 2.19, 2.2, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.22, 1.23, 1.94, 1.95, 2.16, 2.18, 2.2, 2.91, 2.92]),
           st.sampled_from([12.0, 19.0, 26.0, 36.0, 38.0, 39.0, 43.0, 61.0, 78.0, 164.0]),
           st.sampled_from([630.0, 1170.0, 1530.0, 1620.0, 1800.0, 2160.0, 2250.0, 2700.0, 4140.0, 10080.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([14.7, 21.91, 23.4, 29.44, 31.97, 32.3, 40.67, 40.95, 56.57, 138.68]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([0.0, 471.7, 650.0, 768.96, 1288.6, 1400.0, 2022.5, 2101.07, 5650.88, 6041.52]),
           st.sampled_from([0.0, 180.0, 250.0, 269.26, 492.44, 721.11, 829.76, 1612.45, 2189.79, 3059.41]),
           st.sampled_from([150.0, 170.58, 184.94, 204.18, 256.77, 334.28, 345.47, 751.79, 1209.38, 2043.9]),
           st.sampled_from([41.66, 44.75, 45.13, 52.22, 114.82, 117.4, 145.92, 146.03, 249.81, 460.42]),
           st.sampled_from([1.45, 1.67, 2.64, 3.2, 4.06, 4.6, 9.33, 11.89, 13.05, 13.33]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.14, 36.36, 65.58, 65.61, 65.73, 65.77, 65.82, 65.91, 66.15, 66.18]),
           st.sampled_from([5.94, 6.1, 6.46, 6.89, 7.26, 7.28, 7.29, 7.35, 7.41, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([52.0, 63.0, 64.0, 96.0, 162.0, 323.0, 364.0, 735.0, 2274.0, 2971.0]),
           st.sampled_from([120.26, 226.81, 320.75, 327.33, 343.63, 441.43, 474.38, 497.9, 964.23, 1020.91]),
           st.sampled_from([292.22, 295.65, 456.23, 469.39, 491.78, 544.91, 624.0, 1062.68, 1165.76, 1494.86]),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([26.39, 30.41, 31.5, 40.42, 41.22, 41.53, 44.67, 46.29, 50.33, 52.63]),
           st.sampled_from([5.01, 5.42, 5.89, 6.74, 7.97, 8.2, 11.38, 13.54, 13.56, 16.89]),
           st.sampled_from([1858.0, 2000.0, 2280.0, 2472.0, 3340.0, 3483.0, 3720.0, 4748.0, 5362.0, 9294.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([71.2, 81.9, 86.0, 103.8, 113.4, 125.0, 129.6, 162.1, 244.7, 496.7]),
           st.sampled_from([0.11, 0.14, 0.15, 0.18, 0.21, 0.22, 0.29, 0.33, 0.34, 0.43]),
           st.sampled_from([0.24, 0.27, 0.28, 0.3, 0.32, 0.33, 0.41, 0.42, 0.51, 0.67]),
           st.sampled_from([0.24, 0.28, 0.3, 0.31, 0.34, 0.35, 0.39, 0.48, 0.84, 0.85]),
           st.sampled_from([0.07, 0.08, 0.12, 0.14, 0.17, 0.18, 0.2, 0.27, 0.32, 0.5]),
           st.sampled_from([0.15, 0.24, 0.25, 0.3, 0.31, 0.37, 0.42, 0.63, 0.64, 0.87]),
           st.sampled_from([11.66, 24.37, 26.08, 56.02, 64.34, 120.22, 125.35, 203.68, 235.92, 747.64]),
           st.sampled_from([7.64, 9.22, 9.62, 10.98, 14.45, 19.62, 19.93, 24.18, 39.88, 74.88]),
           st.sampled_from([0.34, 0.38, 0.74, 0.79, 0.92, 1.75, 1.76, 1.87, 1.91, 2.14]),
           st.sampled_from([0.15, 0.16, 0.19, 0.21, 0.22, 0.24, 0.25, 0.26, 0.27, 0.31]),
           st.floats(min_value=0.404, max_value=0.499, allow_nan=False),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.39, 0.44, 0.46, 0.54, 0.75, 0.78, 0.92, 0.96, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.25, 0.39, 0.55, 0.57, 0.78, 0.97, 1.01, 1.55, 1.98, 1.99]),
           st.sampled_from([1.64, 2.34, 2.66, 3.21, 3.36, 4.06, 4.36, 4.72, 5.05, 7.31]),
           st.sampled_from([-6.77, -3.58, -3.5, -3.38, -3.1, -2.72, -2.68, -2.34, -1.95, -1.92]),
           st.sampled_from([-0.89, -0.75, -0.74, -0.53, -0.52, -0.5, -0.43, -0.33, -0.25, -0.22]),
           st.sampled_from([1.1, 1.22, 1.23, 1.94, 1.96, 2.16, 2.17, 2.18, 2.19, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.1, 1.22, 1.94, 1.95, 1.96, 2.16, 2.17, 2.18, 2.91, 2.92]),
           st.sampled_from([19.0, 28.0, 29.0, 30.0, 34.0, 36.0, 46.0, 48.0, 78.0, 183.0]),
           st.sampled_from([630.0, 1170.0, 1260.0, 1620.0, 1800.0, 2070.0, 2610.0, 4140.0, 5490.0, 16110.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([13.07, 18.26, 21.91, 24.37, 29.44, 32.3, 34.35, 38.8, 40.95, 56.57]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 403.89, 540.0, 685.42, 742.16, 1060.66, 1253.55, 2672.86, 2930.19, 3936.41]),
           st.sampled_from([90.0, 141.42, 284.6, 484.66, 492.44, 636.4, 853.81, 1659.52, 2189.79, 3059.41]),
           st.sampled_from([0.0, 164.58, 176.72, 236.27, 256.77, 291.21, 334.28, 763.16, 878.29, 1209.38]),
           st.sampled_from([41.66, 45.13, 45.23, 55.93, 60.55, 61.45, 73.48, 135.46, 146.03, 195.92]),
           st.sampled_from([0.0, 1.45, 3.73, 3.9, 4.6, 5.88, 6.32, 8.97, 10.72, 11.32]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.31, 36.5, 65.66, 65.72, 65.77, 65.87, 65.96, 65.97, 66.18, 66.25]),
           st.sampled_from([6.29, 6.42, 6.54, 7.22, 7.28, 7.32, 7.35, 7.95, 15.02, 15.04]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([45.0, 63.0, 79.0, 81.0, 84.0, 167.0, 176.0, 229.0, 448.0, 725.0]),
           st.sampled_from([10.21, 34.36, 57.5, 171.56, 798.03, 993.85, 1260.22, 1289.87, 1331.6, 1489.67]),
           st.sampled_from([265.5, 285.93, 364.42, 441.7, 720.4, 828.01, 1170.51, 1316.48, 1483.52, 1877.84]),
           st.floats(min_value=2.0, max_value=2.49, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([25.58, 26.12, 37.76, 38.91, 41.25, 49.26, 53.9, 57.6, 70.14, 75.6]),
           st.sampled_from([5.94, 6.42, 6.87, 7.39, 8.31, 8.67, 9.76, 14.15, 15.7, 22.22]),
           st.sampled_from([986.5, 990.0, 1932.0, 1950.0, 2260.0, 2510.0, 2850.0, 3600.0, 9433.0, 24066.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([57.6, 65.9, 68.8, 108.4, 112.4, 121.8, 131.0, 132.0, 146.4, 182.3]),
           st.sampled_from([0.08, 0.18, 0.2, 0.21, 0.24, 0.31, 0.46, 0.49, 0.51, 0.6]),
           st.sampled_from([0.25, 0.31, 0.33, 0.38, 0.45, 0.46, 0.49, 0.55, 0.65, 0.72]),
           st.sampled_from([0.15, 0.4, 0.45, 0.55, 0.61, 0.74, 0.77, 0.88, 0.96, 1.01]),
           st.sampled_from([0.03, 0.14, 0.22, 0.24, 0.28, 0.3, 0.39, 0.47, 0.52, 0.56]),
           st.sampled_from([0.01, 0.23, 0.33, 0.55, 0.66, 0.67, 0.75, 0.77, 0.84, 0.92]),
           st.sampled_from([13.94, 16.0, 31.3, 37.71, 38.68, 59.94, 65.55, 84.37, 92.67, 125.51]),
           st.sampled_from([11.91, 12.02, 12.5, 14.55, 14.83, 15.94, 17.09, 17.98, 19.84, 24.81]),
           st.sampled_from([0.16, 0.17, 0.2, 0.67, 0.82, 0.95, 2.19, 2.2, 2.29, 2.54]),
           st.sampled_from([0.02, 0.1, 0.18, 0.21, 0.22, 0.23, 0.35, 0.4, 0.54, 0.58]),
           st.floats(min_value=0.501, max_value=0.554, exclude_min=True, allow_nan=False),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.19, 0.21, 0.47, 0.51, 0.56, 0.82, 0.89, 0.95, 1.05]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-1.08, -0.27, -0.22, 0.25, 0.44, 0.82, 1.01, 1.62, 1.99, 2.04]),
           st.sampled_from([2.23, 3.19, 3.72, 3.73, 4.6, 5.09, 5.62, 6.2, 6.27, 13.62]),
           st.sampled_from([-7.03, -6.25, -5.46, -5.31, -3.44, -3.3, -2.43, -2.03, -1.34, -1.31]),
           st.sampled_from([-1.14, -1.02, -0.97, -0.72, -0.63, -0.61, -0.41, -0.4, -0.28, 0.0]),
           st.sampled_from([1.09, 1.22, 1.23, 1.96, 2.16, 2.17, 2.19, 2.59, 2.62, 2.91]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.23, 1.23, 1.93, 1.94, 2.16, 2.17, 2.23, 2.24, 2.62, 2.96]),
           st.sampled_from([36.0, 64.0, 71.0, 81.0, 88.0, 89.0, 92.0, 118.0, 164.0, 704.0]),
           st.sampled_from([990.0, 1620.0, 1890.0, 2520.0, 2610.0, 4230.0, 7830.0, 8820.0, 12870.0, 95310.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([6.51, 9.73, 12.89, 18.16, 22.55, 31.11, 33.57, 34.83, 35.8, 145.54]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([419.26, 438.93, 691.47, 955.25, 1096.59, 1253.55, 1272.79, 3120.29, 4287.77, 4650.0]),
           st.sampled_from([127.28, 484.66, 547.45, 585.23, 1011.19, 1012.42, 1029.56, 1104.54, 1140.18, 8789.57]),
           st.sampled_from([67.5, 114.55, 115.4, 120.0, 125.0, 134.35, 135.42, 251.73, 314.38, 321.43]),
           st.sampled_from([58.42, 64.68, 71.14, 105.68, 109.16, 121.58, 146.15, 151.78, 191.28, 1155.84]),
           st.sampled_from([1.62, 1.92, 2.36, 2.52, 2.56, 2.95, 4.71, 7.74, 10.73, 12.55]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([35.98, 36.02, 36.03, 36.11, 36.14, 36.47, 36.78, 36.85, 65.58, 65.94]),
           st.sampled_from([5.84, 6.2, 6.23, 6.52, 7.04, 7.05, 7.36, 8.04, 14.7, 15.13]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([18.0, 112.0, 210.0, 229.0, 265.0, 350.0, 603.0, 867.0, 7100.0, 32389.0]),
           st.sampled_from([28.68, 211.51, 478.32, 685.07, 689.57, 780.77, 900.46, 1107.45, 1203.39, 1249.47]),
           st.floats(min_value=269.284, max_value=336.354, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([29.82, 33.57, 36.86, 50.36, 51.04, 51.11, 67.09, 69.92, 71.5, 73.32]),
           st.sampled_from([6.47, 6.98, 7.32, 7.34, 7.95, 8.41, 8.52, 11.73, 15.48, 15.81]),
           st.sampled_from([794.0, 1311.0, 1401.5, 1410.0, 1445.0, 2165.0, 2780.0, 8792.0, 8940.0, 9170.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([65.4, 80.0, 83.5, 84.6, 102.3, 102.7, 103.2, 109.5, 132.4, 340.5]),
           st.sampled_from([0.13, 0.14, 0.18, 0.21, 0.23, 0.28, 0.44, 0.49, 0.56, 0.59]),
           st.sampled_from([0.09, 0.2, 0.21, 0.25, 0.3, 0.6, 0.74, 0.76, 0.79, 0.83]),
           st.sampled_from([0.14, 0.22, 0.31, 0.35, 0.47, 0.58, 0.64, 0.8, 1.0, 1.13]),
           st.sampled_from([0.1, 0.16, 0.3, 0.37, 0.38, 0.41, 0.42, 0.47, 0.52, 0.56]),
           st.sampled_from([0.02, 0.13, 0.15, 0.31, 0.4, 0.45, 0.46, 0.55, 0.62, 0.73]),
           st.floats(min_value=20.383, max_value=24.273, allow_nan=False),
           st.sampled_from([1.99, 2.62, 3.57, 12.99, 16.68, 17.09, 18.13, 19.16, 29.87, 32.23]),
           st.sampled_from([0.39, 0.79, 0.99, 1.82, 1.96, 1.97, 1.98, 2.03, 2.18, 2.51]),
           st.sampled_from([0.02, 0.04, 0.12, 0.25, 0.31, 0.32, 0.36, 0.4, 0.43, 0.44]),
           st.sampled_from([0.13, 0.14, 0.19, 0.2, 0.36, 0.37, 0.56, 0.69, 0.7, 0.76]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.4, 0.43, 0.57, 0.75, 0.79, 1.03, 1.09, 1.12, 1.14]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-1.02, -0.46, -0.37, -0.24, -0.22, -0.13, 1.3, 1.5, 3.2, 5.33]),
           st.sampled_from([2.14, 2.17, 2.26, 2.52, 2.8, 2.93, 3.47, 5.0, 5.55, 5.97]),
           st.sampled_from([-6.19, -5.42, -3.4, -3.37, -2.45, -2.13, -1.58, -1.27, -1.07, -0.83]),
           st.sampled_from([-1.21, -1.07, -0.91, -0.7, -0.67, -0.53, -0.48, -0.33, -0.32, -0.26]),
           st.sampled_from([1.11, 1.21, 1.22, 1.97, 2.16, 2.21, 2.22, 2.23, 2.67, 2.94]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.09, 1.23, 1.96, 1.99, 2.17, 2.19, 2.21, 2.62, 2.67, 2.92]),
           st.sampled_from([18.0, 24.0, 27.0, 62.0, 99.0, 117.0, 141.0, 152.0, 235.0, 369.0]),
           st.sampled_from([1080.0, 2340.0, 2880.0, 3240.0, 5940.0, 16110.0, 16920.0, 18270.0, 42120.0, 51120.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([12.91, 15.44, 18.38, 20.14, 26.12, 29.83, 31.25, 44.5, 47.56, 85.29]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([430.12, 450.0, 509.12, 514.78, 756.64, 969.33, 1297.11, 1298.0, 1350.0, 3401.47]),
           st.sampled_from([154.62, 304.14, 380.79, 403.89, 720.0, 751.66, 768.96, 807.77, 855.13, 874.64]),
           st.sampled_from([81.82, 88.04, 131.62, 135.0, 136.42, 196.58, 218.2, 290.91, 331.0, 439.78]),
           st.sampled_from([40.66, 65.73, 71.71, 71.75, 73.53, 77.12, 89.44, 92.94, 132.8, 555.9]),
           st.sampled_from([1.85, 2.73, 3.35, 3.54, 3.67, 3.82, 5.44, 7.12, 7.6, 12.75]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.36, 36.45, 36.71, 65.88, 65.94, 65.98, 66.07, 66.16, 66.26, 66.27]),
           st.sampled_from([5.95, 6.46, 6.7, 7.02, 7.13, 7.52, 7.75, 7.86, 8.11, 15.42]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 16.0, 50.0, 52.0, 57.0, 95.0, 114.0, 116.0, 270.0, 2558.0]),
           st.sampled_from([21.23, 260.19, 320.75, 441.43, 474.38, 852.49, 1020.91, 1123.09, 1127.41, 1446.29]),
           st.floats(min_value=269.284, max_value=336.354, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([32.76, 34.17, 40.42, 42.16, 43.9, 44.06, 50.33, 52.11, 55.83, 60.03]),
           st.sampled_from([4.33, 4.72, 5.01, 6.24, 7.08, 7.34, 7.99, 8.79, 15.7, 16.89]),
           st.sampled_from([1461.0, 2000.0, 2280.0, 7067.0, 7430.0, 15106.0, 18030.0, 21569.0, 29780.0, 32401.5]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([71.2, 81.9, 83.0, 86.5, 97.0, 103.8, 104.2, 129.6, 214.7, 298.3]),
           st.sampled_from([0.11, 0.13, 0.14, 0.16, 0.18, 0.22, 0.26, 0.34, 0.43, 0.49]),
           st.sampled_from([0.21, 0.22, 0.28, 0.3, 0.32, 0.39, 0.41, 0.53, 0.67, 0.79]),
           st.sampled_from([0.3, 0.31, 0.57, 0.63, 0.65, 0.66, 0.74, 0.95, 0.96, 1.02]),
           st.sampled_from([0.07, 0.1, 0.11, 0.12, 0.13, 0.15, 0.18, 0.25, 0.3, 0.32]),
           st.sampled_from([0.15, 0.19, 0.2, 0.3, 0.33, 0.38, 0.4, 0.54, 0.58, 0.6]),
           st.floats(min_value=24.276, max_value=1231.066, exclude_min=True, allow_nan=False),
           st.sampled_from([14.45, 19.62, 21.67, 24.47, 32.19, 33.47, 49.55, 50.37, 54.12, 59.38]),
           st.sampled_from([0.38, 0.47, 0.78, 0.92, 1.01, 1.06, 1.75, 1.76, 1.87, 2.14]),
           st.sampled_from([0.12, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.32, 0.33, 0.34]),
           st.sampled_from([0.14, 0.19, 0.2, 0.22, 0.26, 0.28, 0.34, 0.37, 0.42, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.24, 0.4, 0.44, 0.5, 0.52, 0.81, 0.86, 0.9, 0.91]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.1, 0.22, 0.55, 0.59, 0.7, 0.73, 0.74, 0.84, 1.79, 1.98]),
           st.sampled_from([1.64, 1.86, 2.33, 2.57, 3.2, 3.84, 5.76, 7.06, 8.68, 11.08]),
           st.sampled_from([-4.78, -3.71, -3.68, -3.64, -3.19, -3.1, -2.5, -2.37, -2.02, -1.92]),
           st.sampled_from([-0.63, -0.41, -0.36, -0.34, -0.32, -0.3, -0.17, -0.16, -0.14, -0.12]),
           st.sampled_from([1.09, 1.1, 1.22, 1.94, 1.95, 2.16, 2.18, 2.19, 2.2, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.22, 1.23, 1.94, 1.95, 2.18, 2.19, 2.2, 2.91, 2.92]),
           st.sampled_from([6.0, 11.0, 14.0, 28.0, 34.0, 46.0, 78.0, 133.0, 164.0, 465.0]),
           st.sampled_from([720.0, 990.0, 1170.0, 1530.0, 1620.0, 2070.0, 2610.0, 6750.0, 8100.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([12.62, 16.0, 16.47, 18.26, 21.91, 24.34, 38.8, 58.27, 61.69, 108.27]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([0.0, 450.0, 650.0, 685.42, 1400.0, 1569.24, 2534.42, 3448.19, 3936.41, 4724.79]),
           st.sampled_from([223.61, 250.0, 282.84, 284.6, 300.0, 320.4, 655.21, 853.81, 1170.0, 2189.79]),
           st.sampled_from([0.0, 67.5, 170.58, 182.83, 193.18, 225.0, 268.75, 291.21, 623.26, 878.29]),
           st.sampled_from([0.0, 41.98, 65.25, 73.48, 88.6, 135.46, 145.92, 160.11, 195.92, 249.81]),
           st.sampled_from([1.37, 1.47, 1.67, 4.0, 4.6, 5.88, 6.32, 7.74, 10.99, 11.89]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([65.58, 65.61, 65.66, 65.67, 65.79, 65.84, 65.87, 65.93, 66.06, 66.25]),
           st.floats(min_value=7.162, max_value=7.499, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([39.0, 64.0, 68.0, 78.0, 92.0, 96.0, 103.0, 260.0, 295.0, 355.0]),
           st.sampled_from([12.48, 22.84, 23.2, 881.82, 1263.12, 1264.54, 1401.25, 1506.37, 1621.06, 1750.3]),
           st.floats(min_value=269.284, max_value=336.354, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([29.17, 30.63, 32.96, 36.19, 37.99, 38.95, 39.64, 39.91, 59.46, 82.64]),
           st.sampled_from([4.84, 6.66, 7.23, 7.37, 9.11, 11.07, 11.53, 15.78, 18.51, 18.92]),
           st.sampled_from([667.0, 1027.0, 1386.0, 1679.5, 1969.0, 2260.0, 2680.0, 3920.0, 4160.0, 28160.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([62.0, 87.5, 90.8, 95.3, 97.1, 102.8, 105.2, 112.1, 113.6, 117.0]),
           st.sampled_from([0.04, 0.16, 0.17, 0.26, 0.29, 0.35, 0.4, 0.43, 0.53, 0.63]),
           st.sampled_from([0.03, 0.14, 0.2, 0.35, 0.49, 0.5, 0.57, 0.59, 0.64, 0.66]),
           st.sampled_from([0.12, 0.39, 0.49, 0.53, 0.6, 0.66, 0.7, 0.75, 0.81, 0.86]),
           st.sampled_from([0.02, 0.09, 0.16, 0.23, 0.26, 0.27, 0.3, 0.38, 0.44, 0.47]),
           st.sampled_from([0.06, 0.32, 0.37, 0.45, 0.5, 0.57, 0.64, 0.77, 0.83, 0.86]),
           st.floats(min_value=24.276, max_value=1231.066, exclude_min=True, allow_nan=False),
           st.sampled_from([3.63, 6.7, 9.35, 15.23, 18.11, 21.56, 24.2, 24.67, 25.0, 36.9]),
           st.sampled_from([0.2, 0.31, 0.67, 0.72, 0.74, 1.04, 2.17, 2.42, 2.55, 2.57]),
           st.sampled_from([0.07, 0.17, 0.27, 0.33, 0.35, 0.4, 0.48, 0.5, 0.55, 0.65]),
           st.sampled_from([0.14, 0.27, 0.35, 0.36, 0.51, 0.61, 0.63, 0.72, 0.73, 0.77]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.22, 0.29, 0.82, 0.84, 0.95, 0.99, 1.07, 1.09, 1.28, 1.29]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.11, 0.33, 0.68, 0.72, 1.1, 1.33, 1.46, 1.78, 2.32, 4.03]),
           st.sampled_from([1.69, 1.86, 2.14, 2.15, 2.78, 3.62, 3.78, 4.96, 5.46, 10.09]),
           st.sampled_from([-5.78, -5.65, -5.47, -5.31, -3.24, -1.54, -1.42, -0.95, -0.83, 0.88]),
           st.sampled_from([-1.06, -0.75, -0.62, -0.56, -0.38, -0.22, -0.2, -0.19, -0.12, -0.09]),
           st.sampled_from([1.1, 1.21, 1.94, 2.19, 2.23, 2.59, 2.6, 2.91, 2.92, 2.95]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.0, 1.08, 1.11, 1.98, 2.01, 2.17, 2.21, 2.67, 2.92, 2.94]),
           st.sampled_from([22.0, 47.0, 48.0, 50.0, 52.0, 53.0, 82.0, 173.0, 356.0, 619.0]),
           st.sampled_from([540.0, 1170.0, 1890.0, 2790.0, 3150.0, 3330.0, 8280.0, 11610.0, 14490.0, 19530.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([10.34, 11.47, 11.51, 11.87, 12.47, 31.68, 35.8, 39.02, 42.6, 57.62]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([618.47, 630.0, 636.4, 768.96, 851.47, 865.75, 1216.55, 1484.32, 1640.12, 2874.46]),
           st.sampled_from([282.84, 403.11, 651.92, 720.0, 807.77, 1094.9, 1484.08, 2050.61, 2405.2, 5020.46]),
           st.sampled_from([74.04, 89.32, 126.19, 127.28, 165.71, 233.33, 296.15, 305.25, 344.5, 404.43]),
           st.sampled_from([42.7, 65.47, 70.95, 77.36, 80.55, 82.45, 84.94, 95.63, 132.73, 221.24]),
           st.sampled_from([2.21, 4.11, 4.52, 5.36, 5.93, 7.16, 8.86, 12.0, 32.25, 41.23]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.06, 36.21, 36.24, 36.35, 36.76, 65.49, 65.52, 65.94, 66.0, 66.1]),
           st.floats(min_value=7.501, max_value=9.088, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([28.0, 46.0, 48.0, 55.0, 102.0, 126.0, 139.0, 188.0, 507.0, 1059.0]),
           st.floats(min_value=8.336, max_value=9.939, allow_nan=False),
           st.floats(min_value=336.357, max_value=813.999, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([26.7, 27.11, 29.23, 29.94, 32.89, 38.13, 38.28, 40.53, 59.48, 60.1]),
           st.sampled_from([7.03, 7.08, 8.19, 8.34, 10.66, 11.17, 11.22, 11.62, 16.78, 19.66]),
           st.sampled_from([667.0, 1520.0, 1699.0, 2180.0, 2387.0, 3610.0, 5270.0, 5360.0, 5944.0, 15254.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([55.8, 85.4, 104.1, 105.9, 133.9, 149.7, 162.9, 188.5, 199.0, 203.6]),
           st.sampled_from([0.05, 0.06, 0.1, 0.16, 0.18, 0.22, 0.26, 0.51, 0.53, 0.54]),
           st.sampled_from([0.17, 0.28, 0.32, 0.43, 0.47, 0.69, 0.75, 0.76, 0.77, 0.79]),
           st.sampled_from([0.07, 0.23, 0.24, 0.25, 0.43, 0.68, 0.71, 0.78, 0.86, 0.95]),
           st.sampled_from([0.02, 0.05, 0.1, 0.14, 0.16, 0.22, 0.25, 0.34, 0.35, 0.58]),
           st.sampled_from([0.06, 0.13, 0.16, 0.22, 0.25, 0.53, 0.62, 0.64, 0.67, 0.73]),
           st.sampled_from([14.18, 16.05, 27.51, 31.77, 46.36, 59.76, 83.32, 88.56, 94.94, 216.83]),
           st.sampled_from([4.0, 9.32, 10.0, 11.24, 11.99, 14.16, 17.14, 24.76, 40.32, 41.51]),
           st.sampled_from([0.62, 0.81, 0.97, 1.07, 1.87, 1.95, 1.98, 2.06, 2.45, 2.51]),
           st.sampled_from([0.04, 0.12, 0.23, 0.24, 0.27, 0.38, 0.39, 0.42, 0.47, 0.52]),
           st.sampled_from([0.06, 0.12, 0.16, 0.19, 0.2, 0.37, 0.43, 0.44, 0.47, 0.56]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.21, 0.26, 0.4, 0.43, 0.54, 0.83, 0.89, 1.03, 1.05, 1.16]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.26, 0.06, 0.68, 0.78, 1.18, 1.2, 1.43, 1.49, 1.68, 5.33]),
           st.sampled_from([1.78, 2.39, 2.62, 3.54, 4.5, 4.58, 5.04, 5.53, 7.55, 9.24]),
           st.sampled_from([-6.57, -5.63, -5.62, -3.42, -3.35, -2.54, -2.17, -1.67, -0.84, -0.76]),
           st.sampled_from([-1.25, -1.06, -0.77, -0.58, -0.55, -0.54, -0.4, -0.34, -0.27, -0.14]),
           st.sampled_from([0.0, 1.93, 1.97, 1.98, 1.99, 2.2, 2.24, 2.65, 2.92, 2.95]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.08, 1.23, 2.18, 2.2, 2.6, 2.63, 2.65, 2.9, 2.92, 2.93]),
           st.sampled_from([8.0, 14.0, 33.0, 35.0, 44.0, 56.0, 67.0, 141.0, 241.0, 356.0]),
           st.sampled_from([540.0, 1170.0, 2880.0, 5490.0, 6390.0, 9810.0, 12330.0, 13410.0, 18270.0, 33390.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([12.58, 16.83, 18.78, 20.69, 20.94, 29.83, 33.18, 34.96, 44.05, 44.25]),
           st.floats(min_value=93.2, max_value=100.49, allow_nan=False),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([364.01, 451.56, 492.44, 603.74, 651.92, 900.0, 912.41, 1360.15, 1400.89, 2140.09]),
           st.sampled_from([90.0, 106.07, 265.17, 380.79, 382.43, 414.2, 569.21, 807.77, 832.17, 1872.78]),
           st.sampled_from([5.89, 88.04, 144.73, 169.71, 178.61, 187.17, 196.96, 232.34, 764.74, 1134.59]),
           st.sampled_from([59.06, 65.47, 67.08, 87.92, 89.49, 120.72, 131.33, 136.04, 155.87, 166.81]),
           st.sampled_from([0.33, 1.49, 2.06, 2.94, 3.07, 3.35, 3.87, 7.9, 12.75, 13.33]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([35.99, 36.36, 36.42, 36.45, 36.82, 66.13, 66.19, 66.32, 66.41, 66.45]),
           st.sampled_from([5.81, 6.14, 6.37, 6.59, 6.89, 7.31, 7.53, 7.91, 8.08, 15.36]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([59.0, 60.0, 114.0, 115.0, 116.0, 323.0, 364.0, 453.0, 735.0, 9264.0]),
           st.floats(min_value=8.336, max_value=9.939, allow_nan=False),
           st.floats(min_value=336.357, max_value=813.999, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([34.17, 38.65, 40.42, 42.37, 44.06, 46.29, 53.53, 53.69, 55.83, 70.65]),
           st.sampled_from([4.68, 5.46, 6.74, 6.92, 7.89, 7.9, 7.99, 8.79, 11.38, 18.6]),
           st.sampled_from([1800.0, 2000.0, 3483.0, 3620.0, 3720.0, 8770.0, 9294.0, 15106.0, 18030.0, 22260.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([71.2, 81.9, 88.7, 103.8, 107.1, 124.9, 152.4, 159.5, 214.7, 427.4]),
           st.sampled_from([0.16, 0.19, 0.21, 0.22, 0.24, 0.28, 0.31, 0.33, 0.43, 0.49]),
           st.sampled_from([0.2, 0.21, 0.25, 0.26, 0.27, 0.28, 0.29, 0.35, 0.38, 0.39]),
           st.sampled_from([0.36, 0.46, 0.49, 0.5, 0.53, 0.54, 0.74, 0.84, 0.96, 1.02]),
           st.sampled_from([0.08, 0.1, 0.11, 0.12, 0.14, 0.17, 0.21, 0.23, 0.3, 0.35]),
           st.sampled_from([0.16, 0.21, 0.26, 0.3, 0.31, 0.33, 0.37, 0.42, 0.58, 0.82]),
           st.sampled_from([19.85, 24.37, 24.48, 29.0, 56.02, 58.3, 99.59, 136.77, 203.68, 747.64]),
           st.sampled_from([7.3, 9.22, 10.98, 11.96, 12.15, 19.27, 25.06, 32.19, 50.37, 73.4]),
           st.sampled_from([0.3, 0.33, 0.34, 0.74, 0.78, 0.79, 0.82, 0.9, 1.01, 1.06]),
           st.sampled_from([0.1, 0.12, 0.14, 0.15, 0.16, 0.23, 0.27, 0.28, 0.33, 0.34]),
           st.sampled_from([0.19, 0.2, 0.21, 0.25, 0.26, 0.28, 0.31, 0.42, 0.43, 0.49]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.19, 0.22, 0.24, 0.4, 0.45, 0.54, 0.75, 0.86, 0.88, 0.96]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.11, 0.21, 0.26, 0.39, 0.56, 0.57, 0.69, 1.01, 1.08, 1.99]),
           st.sampled_from([3.21, 3.34, 3.44, 3.79, 4.36, 4.72, 5.05, 5.76, 6.54, 6.81]),
           st.sampled_from([-7.76, -4.78, -3.62, -3.38, -3.36, -3.24, -3.1, -2.34, -2.02, -0.65]),
           st.sampled_from([-1.02, -0.53, -0.52, -0.51, -0.5, -0.48, -0.34, -0.17, -0.14, -0.09]),
           st.sampled_from([1.09, 1.1, 1.22, 1.94, 2.16, 2.17, 2.18, 2.2, 2.59, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.1, 1.22, 1.94, 1.96, 2.16, 2.17, 2.19, 2.2, 2.91, 2.92]),
           st.sampled_from([12.0, 18.0, 21.0, 26.0, 38.0, 39.0, 43.0, 48.0, 96.0, 183.0]),
           st.sampled_from([450.0, 900.0, 990.0, 1260.0, 1620.0, 2070.0, 5490.0, 6750.0, 8730.0, 10080.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.6, 12.62, 16.37, 18.24, 21.97, 24.37, 34.35, 78.14, 80.59, 87.16]),
           st.floats(min_value=100.51, max_value=109.0, exclude_min=True, allow_nan=False),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([450.0, 685.42, 707.11, 1253.55, 1281.6, 2101.07, 2704.16, 3471.31, 3936.41, 4724.79]),
           st.sampled_from([90.0, 180.0, 200.0, 212.13, 223.61, 320.4, 484.66, 603.74, 813.94, 1612.45]),
           st.sampled_from([67.5, 150.0, 170.58, 176.72, 204.18, 345.47, 368.74, 474.05, 878.29, 1209.38]),
           st.sampled_from([45.13, 65.25, 84.35, 114.82, 124.72, 145.92, 146.03, 179.03, 195.92, 349.26]),
           st.sampled_from([1.37, 1.71, 3.9, 5.88, 6.63, 7.74, 8.97, 11.89, 13.33, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([36.18, 36.36, 36.49, 65.58, 65.67, 65.91, 65.92, 66.06, 66.18, 66.25]),
           st.sampled_from([5.94, 6.3, 6.89, 7.39, 7.65, 7.84, 8.07, 14.45, 14.92, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([53.0, 54.0, 64.0, 90.0, 128.0, 145.0, 159.0, 173.0, 231.0, 649.0]),
           st.floats(min_value=9.942, max_value=386.569, exclude_min=True, allow_nan=False),
           st.floats(min_value=336.357, max_value=813.999, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.51, max_value=38.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=121562.51, max_value=14360250.0, exclude_min=True, allow_nan=False),
           st.sampled_from([29.92, 34.37, 36.05, 37.07, 39.41, 40.91, 41.43, 50.06, 50.33, 51.29]),
           st.sampled_from([5.18, 5.29, 5.42, 7.81, 8.1, 11.46, 11.59, 12.61, 15.09, 16.65]),
           st.sampled_from([937.0, 1290.0, 1387.0, 1715.0, 1780.0, 2403.0, 3340.0, 3730.0, 3920.0, 8190.0]),
           st.floats(min_value=0.116, max_value=0.24, exclude_min=True, allow_nan=False),
           st.sampled_from([41.0, 55.9, 65.8, 92.5, 107.4, 118.8, 125.9, 132.0, 188.5, 247.0]),
           st.sampled_from([0.02, 0.05, 0.15, 0.18, 0.21, 0.22, 0.27, 0.31, 0.49, 0.66]),
           st.sampled_from([0.16, 0.24, 0.33, 0.37, 0.38, 0.39, 0.44, 0.47, 0.59, 0.68]),
           st.sampled_from([0.06, 0.24, 0.46, 0.47, 0.57, 0.67, 0.69, 0.82, 1.01, 1.1]),
           st.sampled_from([0.01, 0.05, 0.11, 0.15, 0.3, 0.31, 0.38, 0.39, 0.43, 0.47]),
           st.sampled_from([0.04, 0.15, 0.18, 0.23, 0.25, 0.38, 0.42, 0.49, 0.69, 0.85]),
           st.sampled_from([11.95, 16.8, 23.34, 31.7, 41.53, 60.01, 69.8, 73.66, 89.57, 121.05]),
           st.sampled_from([7.88, 10.75, 11.22, 18.52, 19.79, 21.59, 23.04, 23.61, 29.03, 43.91]),
           st.sampled_from([0.54, 0.74, 0.86, 1.01, 1.15, 1.2, 1.91, 2.01, 2.16, 2.3]),
           st.sampled_from([0.12, 0.2, 0.21, 0.23, 0.26, 0.28, 0.36, 0.41, 0.48, 0.52]),
           st.sampled_from([0.03, 0.12, 0.14, 0.3, 0.31, 0.5, 0.56, 0.57, 0.59, 0.76]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.46, 0.75, 0.82, 0.87, 0.89, 0.94, 0.96, 1.01, 1.14, 1.2]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.39, -0.38, -0.07, -0.03, 0.07, 0.45, 1.54, 1.64, 1.68, 2.62]),
           st.sampled_from([1.85, 1.99, 2.52, 3.07, 4.95, 4.99, 5.12, 8.2, 13.24, 31.09]),
           st.sampled_from([-5.65, -3.3, -2.93, -2.91, -2.4, -2.03, -1.82, -1.76, -0.98, -0.89]),
           st.sampled_from([-1.22, -1.06, -0.77, -0.72, -0.4, -0.37, -0.29, -0.25, -0.24, -0.19]),
           st.sampled_from([1.1, 1.22, 1.24, 2.01, 2.22, 2.23, 2.6, 2.9, 2.93, 2.95]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.0, 1.1, 1.22, 1.96, 1.97, 2.21, 2.6, 2.62, 2.67, 2.96]),
           st.sampled_from([7.0, 19.0, 50.0, 51.0, 65.0, 90.0, 107.0, 114.0, 117.0, 882.0]),
           st.sampled_from([900.0, 1440.0, 1620.0, 2520.0, 3780.0, 3960.0, 4860.0, 4950.0, 6300.0, 7470.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([16.16, 32.0, 33.62, 45.3, 49.15, 49.84, 60.43, 68.33, 73.76, 114.88]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([318.2, 865.75, 886.0, 969.33, 1104.54, 1274.75, 1484.08, 1553.64, 1882.15, 4419.56]),
           st.sampled_from([187.5, 335.41, 339.58, 453.11, 559.02, 569.21, 670.82, 1068.88, 1662.08, 3095.56]),
           st.sampled_from([152.42, 220.0, 232.17, 243.17, 246.69, 283.28, 300.0, 300.52, 639.88, 1134.59]),
           st.sampled_from([39.69, 54.08, 59.88, 111.99, 125.81, 126.56, 127.61, 153.84, 191.45, 1648.8]),
           st.sampled_from([1.33, 3.3, 3.51, 3.64, 3.67, 3.86, 3.95, 4.11, 5.41, 22.02]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=18376.667, max_value=22457.958, allow_nan=False),
           st.sampled_from([35.95, 36.2, 65.39, 65.67, 65.83, 66.04, 66.14, 66.29, 66.35, 66.42]),
           st.sampled_from([5.81, 5.94, 6.61, 6.94, 7.21, 7.38, 7.74, 8.13, 14.7, 15.27]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=3.51, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 29.0, 63.0, 107.0, 128.0, 162.0, 183.0, 188.0, 270.0, 364.0]),
           st.sampled_from([231.84, 260.19, 273.61, 320.75, 327.33, 369.85, 389.76, 1123.09, 1127.41, 1562.53]),
           st.sampled_from([242.59, 276.56, 303.14, 305.06, 305.85, 456.23, 456.63, 546.0, 566.96, 1184.91]),
           st.sampled_from([11.0, 33.0, 42.0, 52.0, 68.0, 69.0, 82.0, 90.0, 124.0, 140.0]),
           st.sampled_from([118125.0, 125000.0, 150468.0, 287500.0, 290000.0, 320000.0, 470000.0, 1312200.0, 3002500.0, 7887500.0]),
           st.sampled_from([22.79, 24.09, 24.65, 29.28, 40.46, 40.88, 41.08, 41.22, 46.29, 52.63]),
           st.sampled_from([4.83, 5.46, 6.13, 6.24, 6.74, 8.79, 10.48, 11.38, 13.54, 16.89]),
           st.sampled_from([1145.5, 1318.0, 1850.0, 2000.0, 2280.0, 3340.0, 6998.0, 15880.0, 18030.0, 22260.0]),
           st.sampled_from([0.09, 0.1, 0.11, 0.15, 0.19, 0.23, 0.24, 0.25, 0.27, 0.32]),
           st.sampled_from([88.4, 101.3, 103.8, 107.1, 113.4, 124.9, 195.2, 255.4, 298.3, 496.7]),
           st.sampled_from([0.13, 0.14, 0.15, 0.16, 0.2, 0.22, 0.23, 0.35, 0.36, 0.52]),
           st.sampled_from([0.16, 0.22, 0.29, 0.3, 0.32, 0.38, 0.42, 0.51, 0.67, 0.79]),
           st.sampled_from([0.23, 0.24, 0.33, 0.36, 0.49, 0.63, 0.64, 0.66, 0.84, 0.96]),
           st.sampled_from([0.07, 0.08, 0.1, 0.12, 0.13, 0.18, 0.23, 0.25, 0.32, 0.5]),
           st.sampled_from([0.08, 0.16, 0.29, 0.3, 0.33, 0.34, 0.35, 0.38, 0.42, 0.55]),
           st.sampled_from([11.66, 20.38, 26.35, 41.65, 71.2, 99.59, 120.22, 136.77, 165.27, 207.31]),
           st.sampled_from([8.32, 9.14, 10.98, 11.96, 16.73, 19.27, 19.93, 39.88, 49.55, 54.12]),
           st.sampled_from([0.47, 0.62, 0.78, 0.88, 0.94, 1.75, 1.76, 1.84, 1.91, 2.14]),
           st.sampled_from([0.1, 0.12, 0.14, 0.16, 0.17, 0.19, 0.24, 0.25, 0.27, 0.28]),
           st.sampled_from([0.13, 0.19, 0.2, 0.21, 0.31, 0.32, 0.34, 0.37, 0.4, 0.52]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.19, 0.36, 0.47, 0.5, 0.54, 0.75, 0.88, 0.91, 0.92, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.51, 0.12, 0.26, 0.47, 0.55, 0.69, 1.11, 1.44, 1.61, 1.99]),
           st.sampled_from([1.64, 2.33, 3.32, 3.34, 3.52, 3.84, 4.02, 7.31, 8.68, 10.63]),
           st.sampled_from([-6.77, -6.01, -5.55, -5.18, -3.62, -3.54, -3.36, -2.24, -1.95, -0.87]),
           st.sampled_from([-0.89, -0.75, -0.63, -0.59, -0.5, -0.43, -0.32, -0.28, -0.22, -0.12]),
           st.sampled_from([1.09, 1.23, 1.94, 1.95, 1.96, 2.18, 2.19, 2.2, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.22, 1.23, 1.94, 2.17, 2.18, 2.19, 2.2, 2.92]),
           st.sampled_from([6.0, 15.0, 17.0, 19.0, 21.0, 39.0, 61.0, 133.0, 150.0, 183.0]),
           st.sampled_from([1620.0, 2160.0, 2610.0, 2700.0, 4140.0, 8100.0, 8730.0, 10350.0, 16110.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.81, 16.37, 24.37, 29.44, 32.3, 34.35, 40.67, 46.19, 78.14, 80.59]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 685.42, 707.11, 960.47, 1253.55, 1288.6, 1400.0, 2022.5, 2846.05, 2850.0]),
           st.sampled_from([212.13, 282.84, 316.23, 484.66, 603.74, 636.4, 655.21, 721.11, 1170.0, 3059.41]),
           st.sampled_from([0.0, 67.5, 120.21, 182.83, 204.18, 236.27, 291.21, 334.28, 878.29, 1245.07]),
           st.sampled_from([41.98, 45.13, 55.93, 65.25, 73.11, 124.72, 144.97, 146.03, 179.03, 477.23]),
           st.sampled_from([1.47, 1.7, 1.94, 4.6, 5.88, 6.63, 6.95, 7.33, 7.74, 11.89]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=22457.961, max_value=22636.027, exclude_min=True, allow_nan=False),
           st.sampled_from([36.49, 36.5, 65.79, 65.8, 65.81, 65.9, 65.93, 66.04, 66.15, 66.25]),
           st.sampled_from([5.94, 6.07, 6.1, 6.96, 7.28, 7.39, 7.58, 7.84, 7.85, 8.07]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([29.0, 57.0, 59.0, 95.0, 109.0, 114.0, 116.0, 128.0, 354.0, 364.0]),
           st.sampled_from([165.49, 221.79, 227.23, 263.09, 320.75, 369.85, 386.31, 441.43, 1123.09, 1506.09]),
           st.sampled_from([292.22, 365.69, 456.63, 546.0, 566.96, 580.94, 630.8, 1167.85, 1296.07, 1534.94]),
           st.sampled_from([10.0, 18.0, 42.0, 62.0, 66.0, 90.0, 96.0, 124.0, 140.0, 155.0]),
           st.floats(min_value=73687.2, max_value=74530.99, allow_nan=False),
           st.sampled_from([23.92, 24.65, 34.17, 40.46, 41.22, 42.16, 44.06, 53.53, 53.69, 54.2]),
           st.sampled_from([4.33, 5.01, 5.42, 5.94, 6.13, 7.89, 7.9, 7.97, 11.54, 13.56]),
           st.sampled_from([880.5, 1130.0, 1461.0, 1800.0, 1850.0, 3620.0, 5362.0, 8770.0, 22260.0, 29780.0]),
           st.sampled_from([0.08, 0.09, 0.1, 0.15, 0.23, 0.24, 0.25, 0.26, 0.27, 0.34]),
           st.sampled_from([67.6, 86.5, 104.5, 121.9, 124.9, 126.4, 244.7, 255.4, 298.3, 496.7]),
           st.sampled_from([0.11, 0.13, 0.15, 0.16, 0.18, 0.26, 0.31, 0.33, 0.36, 0.49]),
           st.sampled_from([0.25, 0.26, 0.29, 0.33, 0.38, 0.41, 0.42, 0.51, 0.67, 0.79]),
           st.sampled_from([0.24, 0.28, 0.3, 0.31, 0.42, 0.53, 0.63, 0.65, 0.74, 0.95]),
           st.sampled_from([0.08, 0.11, 0.14, 0.17, 0.18, 0.2, 0.23, 0.27, 0.35, 0.5]),
           st.sampled_from([0.14, 0.15, 0.2, 0.24, 0.25, 0.31, 0.34, 0.38, 0.63, 0.64]),
           st.sampled_from([16.06, 20.47, 21.63, 37.51, 41.65, 58.3, 118.11, 165.27, 203.68, 207.31]),
           st.sampled_from([7.3, 8.32, 11.96, 16.73, 19.93, 24.18, 32.19, 33.47, 54.12, 59.38]),
           st.sampled_from([0.3, 0.62, 0.65, 0.78, 0.79, 1.0, 1.76, 1.82, 1.84, 1.91]),
           st.sampled_from([0.1, 0.15, 0.19, 0.23, 0.25, 0.26, 0.27, 0.28, 0.31, 0.33]),
           st.sampled_from([0.13, 0.18, 0.19, 0.21, 0.23, 0.27, 0.34, 0.43, 0.49, 0.52]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.22, 0.4, 0.44, 0.46, 0.48, 0.5, 0.9, 0.91, 0.97, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.25, 0.69, 0.7, 0.73, 0.78, 1.08, 1.22, 1.55, 1.98, 2.5]),
           st.sampled_from([2.34, 2.66, 3.32, 3.84, 4.02, 4.35, 4.79, 7.06, 7.31, 10.63]),
           st.sampled_from([-6.77, -3.89, -3.58, -2.72, -2.34, -2.24, -2.02, -1.92, -1.39, -0.87]),
           st.sampled_from([-0.89, -0.48, -0.43, -0.36, -0.32, -0.23, -0.22, -0.21, -0.17, -0.07]),
           st.sampled_from([1.09, 1.22, 1.94, 1.96, 2.17, 2.18, 2.19, 2.59, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.22, 1.23, 1.94, 1.96, 2.17, 2.19, 2.59, 2.91]),
           st.sampled_from([13.0, 18.0, 19.0, 22.0, 28.0, 36.0, 38.0, 43.0, 61.0, 164.0]),
           st.floats(min_value=684.0, max_value=764.99, allow_nan=False),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([13.07, 16.47, 18.24, 18.26, 31.97, 46.19, 48.75, 56.57, 80.59, 138.68]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([450.0, 471.7, 540.0, 608.28, 685.42, 707.11, 742.16, 1288.6, 3471.31, 3936.41]),
           st.sampled_from([90.0, 200.0, 212.13, 223.61, 269.26, 492.44, 603.74, 636.4, 761.58, 1170.0]),
           st.sampled_from([135.0, 150.0, 164.58, 170.58, 184.94, 204.18, 236.27, 291.21, 368.74, 878.29]),
           st.sampled_from([50.12, 65.25, 85.79, 144.97, 145.92, 146.03, 160.11, 220.85, 348.7, 460.42]),
           st.sampled_from([0.0, 1.37, 1.47, 1.67, 1.94, 2.64, 9.33, 10.72, 11.32, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.sampled_from([36.18, 65.61, 65.72, 65.73, 65.74, 65.79, 65.87, 65.9, 65.97, 66.25]),
           st.sampled_from([5.94, 6.29, 6.3, 6.96, 7.18, 7.26, 7.35, 7.55, 7.58, 14.45]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([24.0, 67.0, 131.0, 158.0, 178.0, 210.0, 229.0, 235.0, 521.0, 2339.0]),
           st.sampled_from([37.0, 144.92, 226.62, 279.57, 882.35, 927.46, 987.03, 1039.62, 1172.7, 1351.19]),
           st.sampled_from([335.33, 369.72, 533.94, 626.25, 652.73, 739.4, 1033.9, 1283.61, 1778.77, 1789.38]),
           st.sampled_from([2.0, 4.0, 18.0, 21.0, 34.0, 36.0, 111.0, 124.0, 166.0, 169.0]),
           st.floats(min_value=73687.2, max_value=74530.99, allow_nan=False),
           st.sampled_from([23.17, 29.96, 31.23, 33.39, 33.83, 35.8, 41.27, 41.31, 52.53, 74.27]),
           st.sampled_from([6.33, 7.43, 7.79, 8.06, 8.3, 8.45, 8.49, 8.98, 14.15, 15.03]),
           st.sampled_from([1477.0, 1489.5, 1620.0, 1960.0, 2190.0, 3160.0, 3992.5, 9170.0, 10070.0, 18590.0]),
           st.sampled_from([0.02, 0.11, 0.13, 0.16, 0.18, 0.21, 0.28, 0.32, 0.39, 0.4]),
           st.sampled_from([63.4, 88.5, 93.7, 95.2, 110.5, 115.7, 116.3, 116.4, 118.8, 128.0]),
           st.sampled_from([0.08, 0.13, 0.17, 0.18, 0.21, 0.22, 0.25, 0.29, 0.4, 0.44]),
           st.sampled_from([0.1, 0.19, 0.23, 0.27, 0.37, 0.49, 0.54, 0.55, 0.62, 0.7]),
           st.sampled_from([0.09, 0.1, 0.25, 0.58, 0.9, 0.96, 0.97, 1.09, 1.1, 1.12]),
           st.sampled_from([0.05, 0.06, 0.1, 0.11, 0.14, 0.18, 0.19, 0.23, 0.35, 0.38]),
           st.sampled_from([0.07, 0.13, 0.17, 0.23, 0.27, 0.31, 0.33, 0.46, 0.72, 0.76]),
           st.sampled_from([13.19, 13.94, 20.04, 20.63, 24.12, 35.01, 54.1, 61.72, 121.05, 1437.91]),
           st.sampled_from([2.41, 14.06, 14.58, 15.71, 15.8, 16.26, 17.0, 32.23, 43.0, 351.0]),
           st.sampled_from([0.2, 0.6, 0.85, 1.2, 1.77, 1.82, 2.0, 2.08, 2.09, 2.6]),
           st.sampled_from([0.09, 0.19, 0.21, 0.25, 0.29, 0.32, 0.35, 0.37, 0.39, 0.52]),
           st.sampled_from([0.02, 0.08, 0.12, 0.28, 0.33, 0.37, 0.57, 0.61, 0.72, 0.76]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.31, 0.39, 0.45, 0.75, 0.93, 0.99, 1.03, 1.16, 1.19, 1.33]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-1.15, -0.04, 0.54, 0.86, 0.92, 0.94, 1.72, 2.96, 3.04, 5.72]),
           st.sampled_from([1.54, 1.82, 2.68, 2.86, 3.1, 3.15, 3.71, 4.88, 5.54, 16.67]),
           st.sampled_from([-6.5, -6.47, -6.37, -5.83, -2.98, -2.89, -2.71, -2.45, -1.69, -1.42]),
           st.sampled_from([-1.25, -1.02, -0.84, -0.78, -0.75, -0.66, -0.49, -0.33, -0.14, -0.08]),
           st.sampled_from([1.22, 1.94, 1.96, 2.21, 2.22, 2.23, 2.63, 2.64, 2.91, 2.98]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.0, 1.23, 1.93, 1.95, 1.96, 1.99, 2.19, 2.21, 2.67, 2.95]),
           st.sampled_from([3.0, 12.0, 41.0, 74.0, 93.0, 117.0, 118.0, 241.0, 291.0, 356.0]),
           st.floats(min_value=765.01, max_value=19674.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([8.65, 10.98, 11.71, 22.55, 22.94, 23.99, 27.23, 29.0, 29.34, 44.05]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([400.0, 450.0, 860.23, 917.88, 1030.78, 1138.42, 1520.69, 1775.53, 1822.36, 3560.41]),
           st.sampled_from([447.21, 456.21, 524.79, 626.5, 651.92, 1044.03, 1650.0, 1844.59, 4465.49, 8789.57]),
           st.sampled_from([64.35, 86.78, 111.31, 163.64, 195.08, 227.93, 233.33, 368.18, 455.78, 651.51]),
           st.sampled_from([51.09, 54.08, 64.68, 78.01, 87.7, 92.58, 140.92, 173.0, 555.9, 753.07]),
           st.sampled_from([2.19, 2.57, 2.64, 2.66, 3.02, 3.07, 7.75, 10.88, 15.3, 18.71]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.sampled_from([36.14, 36.59, 36.69, 36.78, 65.45, 65.5, 65.68, 65.74, 65.79, 66.17]),
           st.sampled_from([6.03, 6.34, 6.5, 7.1, 7.4, 8.18, 14.9, 15.03, 15.14, 15.36]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([72.0, 81.0, 99.0, 108.0, 119.0, 180.0, 184.0, 191.0, 2012.0, 4851.0]),
           st.sampled_from([10.69, 65.0, 124.9, 144.84, 491.81, 755.12, 858.28, 951.92, 1424.86, 1583.82]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.sampled_from([10.0, 25.0, 34.0, 42.0, 62.0, 112.0, 141.0, 148.0, 156.0, 161.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([27.3, 28.89, 29.53, 31.53, 37.55, 39.8, 47.56, 47.74, 52.47, 56.32]),
           st.sampled_from([5.6, 5.92, 6.49, 7.37, 7.93, 8.17, 10.26, 11.13, 16.65, 17.02]),
           st.sampled_from([741.0, 1641.0, 1950.0, 2075.0, 3435.0, 4857.5, 5080.0, 5360.0, 18590.0, 20830.0]),
           st.sampled_from([0.03, 0.1, 0.16, 0.19, 0.31, 0.32, 0.34, 0.35, 0.53, 0.63]),
           st.sampled_from([73.0, 78.9, 80.0, 93.3, 96.8, 97.5, 99.7, 110.3, 158.0, 217.4]),
           st.sampled_from([0.15, 0.19, 0.25, 0.27, 0.3, 0.42, 0.5, 0.51, 0.61, 0.63]),
           st.sampled_from([0.06, 0.23, 0.28, 0.31, 0.34, 0.35, 0.38, 0.39, 0.43, 0.6]),
           st.sampled_from([0.09, 0.11, 0.23, 0.32, 0.42, 0.51, 0.56, 0.77, 0.84, 1.13]),
           st.sampled_from([0.2, 0.32, 0.33, 0.34, 0.39, 0.51, 0.52, 0.53, 0.56, 0.65]),
           st.sampled_from([0.09, 0.22, 0.3, 0.4, 0.43, 0.45, 0.51, 0.67, 0.85, 0.86]),
           st.sampled_from([22.6, 26.79, 30.25, 39.56, 49.28, 49.32, 59.71, 62.57, 65.71, 142.82]),
           st.sampled_from([13.22, 14.23, 20.85, 20.98, 22.66, 23.37, 25.42, 26.13, 30.96, 31.7]),
           st.sampled_from([0.19, 0.56, 0.84, 0.86, 0.88, 0.97, 1.06, 1.12, 1.91, 2.1]),
           st.sampled_from([0.02, 0.05, 0.24, 0.25, 0.33, 0.38, 0.47, 0.51, 0.54, 0.55]),
           st.sampled_from([0.18, 0.28, 0.31, 0.32, 0.43, 0.47, 0.57, 0.59, 0.61, 0.64]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.35, 0.48, 0.56, 0.75, 0.89, 0.91, 1.09, 1.13, 1.16, 1.27]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-1.08, 0.24, 0.36, 0.37, 0.92, 1.38, 1.86, 1.95, 2.0, 2.09]),
           st.floats(min_value=1.647, max_value=1.698, allow_nan=False),
           st.sampled_from([-6.6, -5.0, -3.27, -3.22, -2.99, -2.45, -2.43, -2.36, -1.05, -0.94]),
           st.sampled_from([-1.21, -1.05, -1.02, -0.97, -0.75, -0.66, -0.52, -0.51, -0.5, -0.45]),
           st.sampled_from([1.11, 1.24, 1.93, 2.19, 2.22, 2.24, 2.64, 2.91, 2.96, 2.98]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.09, 1.21, 2.17, 2.23, 2.61, 2.91, 2.92, 2.94, 2.96, 2.97]),
           st.sampled_from([19.0, 34.0, 99.0, 228.0, 281.0, 310.0, 476.0, 623.0, 704.0, 882.0]),
           st.sampled_from([1350.0, 2970.0, 3060.0, 4140.0, 4320.0, 4770.0, 6930.0, 8100.0, 25470.0, 34290.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.floats(min_value=11.937, max_value=13.658, allow_nan=False),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([150.0, 324.5, 582.16, 764.85, 850.0, 990.0, 1051.19, 1113.24, 2205.11, 3623.55]),
           st.sampled_from([364.01, 403.89, 493.24, 649.0, 768.96, 1012.42, 1234.91, 1872.78, 2450.0, 4816.64]),
           st.floats(min_value=358.971, max_value=448.713, allow_nan=False),
           st.sampled_from([36.64, 54.64, 57.02, 70.52, 89.49, 90.03, 98.14, 99.78, 116.77, 156.73]),
           st.sampled_from([1.29, 4.29, 4.32, 4.56, 4.68, 4.73, 6.14, 7.6, 8.41, 20.68]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=59.922, max_value=65.914, allow_nan=False),
           st.sampled_from([6.01, 6.21, 6.64, 6.76, 7.09, 7.1, 7.2, 7.28, 7.66, 15.09]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([31.0, 57.0, 59.0, 64.0, 96.0, 162.0, 270.0, 453.0, 1201.0, 2971.0]),
           st.sampled_from([165.49, 231.84, 327.33, 335.02, 389.76, 454.66, 852.49, 1438.13, 1506.09, 1562.53]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.sampled_from([11.0, 22.0, 33.0, 51.0, 52.0, 73.0, 82.0, 90.0, 96.0, 132.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([22.79, 31.5, 44.67, 47.59, 49.35, 52.11, 53.69, 55.83, 60.03, 70.65]),
           st.sampled_from([4.68, 4.72, 5.89, 5.94, 6.74, 7.36, 7.89, 7.9, 8.79, 13.56]),
           st.sampled_from([1318.0, 1461.0, 1570.0, 1800.0, 1858.0, 2370.0, 3340.0, 4187.0, 4748.0, 15106.0]),
           st.sampled_from([0.1, 0.11, 0.18, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.32]),
           st.sampled_from([88.7, 101.3, 103.8, 113.4, 126.3, 126.4, 195.2, 255.4, 276.0, 427.4]),
           st.sampled_from([0.11, 0.13, 0.14, 0.18, 0.2, 0.23, 0.31, 0.33, 0.34, 0.49]),
           st.sampled_from([0.14, 0.16, 0.24, 0.27, 0.28, 0.3, 0.33, 0.39, 0.41, 0.51]),
           st.sampled_from([0.28, 0.33, 0.35, 0.42, 0.53, 0.62, 0.63, 0.65, 0.69, 1.02]),
           st.sampled_from([0.07, 0.13, 0.14, 0.17, 0.2, 0.21, 0.24, 0.27, 0.32, 0.35]),
           st.sampled_from([0.14, 0.2, 0.24, 0.3, 0.33, 0.34, 0.37, 0.38, 0.55, 0.6]),
           st.sampled_from([11.66, 20.47, 21.74, 26.08, 29.0, 93.32, 136.77, 165.27, 203.68, 235.92]),
           st.sampled_from([8.32, 12.15, 14.72, 22.56, 24.47, 25.06, 43.96, 49.55, 50.99, 73.4]),
           st.sampled_from([0.15, 0.36, 0.38, 0.78, 0.82, 0.84, 0.9, 1.01, 1.87, 2.14]),
           st.sampled_from([0.1, 0.12, 0.14, 0.15, 0.22, 0.24, 0.26, 0.29, 0.32, 0.33]),
           st.sampled_from([0.13, 0.18, 0.23, 0.25, 0.26, 0.27, 0.37, 0.42, 0.43, 0.49]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.22, 0.36, 0.39, 0.45, 0.5, 0.54, 0.75, 0.78, 0.97, 1.02]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.12, 0.22, 0.25, 0.29, 0.55, 0.7, 0.74, 0.97, 1.61, 1.99]),
           st.floats(min_value=1.647, max_value=1.698, allow_nan=False),
           st.sampled_from([-7.76, -5.55, -3.68, -3.54, -3.38, -3.1, -2.72, -2.37, -1.92, -1.39]),
           st.sampled_from([-1.02, -0.59, -0.53, -0.36, -0.33, -0.32, -0.25, -0.21, -0.18, -0.14]),
           st.sampled_from([1.1, 1.23, 1.94, 1.95, 2.17, 2.19, 2.2, 2.59, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.94, 1.95, 1.96, 2.18, 2.19, 2.59, 2.91, 2.92]),
           st.sampled_from([6.0, 13.0, 18.0, 20.0, 29.0, 30.0, 53.0, 78.0, 164.0, 183.0]),
           st.sampled_from([630.0, 1170.0, 1530.0, 1890.0, 2160.0, 2700.0, 2790.0, 4140.0, 5490.0, 10080.0]),
           st.sampled_from([0.0, 0.01]),
           st.floats(min_value=13.661, max_value=99.174, exclude_min=True, allow_nan=False),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 471.7, 1400.0, 2022.5, 2704.16, 2846.05, 2850.0, 2930.19, 3936.41, 4724.79]),
           st.sampled_from([141.42, 223.61, 269.26, 284.6, 300.0, 603.74, 636.4, 721.11, 813.94, 1659.52]),
           st.floats(min_value=358.971, max_value=448.713, allow_nan=False),
           st.sampled_from([41.98, 50.12, 60.55, 60.97, 61.45, 124.72, 220.85, 348.7, 460.42, 477.23]),
           st.sampled_from([0.0, 1.45, 2.7, 2.85, 4.0, 4.06, 7.33, 7.74, 13.05, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=59.922, max_value=65.914, allow_nan=False),
           st.sampled_from([6.07, 6.54, 6.96, 7.24, 7.28, 7.36, 7.85, 7.95, 8.07, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([73.0, 97.0, 110.0, 112.0, 130.0, 192.0, 213.0, 244.0, 276.0, 1574.0]),
           st.sampled_from([97.15, 510.33, 1096.68, 1229.13, 1245.35, 1469.75, 1501.54, 1582.84, 1596.73, 1601.62]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.sampled_from([71.0, 91.0, 102.0, 103.0, 121.0, 124.0, 130.0, 154.0, 164.0, 171.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([25.19, 33.83, 34.31, 35.09, 38.72, 40.17, 40.58, 43.03, 50.92, 70.5]),
           st.sampled_from([5.04, 6.75, 7.16, 7.52, 8.88, 8.94, 10.62, 12.11, 18.22, 22.22]),
           st.sampled_from([847.0, 1270.0, 1820.0, 1929.0, 2038.0, 2435.0, 4400.0, 6040.0, 6797.0, 13520.0]),
           st.sampled_from([0.1, 0.15, 0.23, 0.25, 0.37, 0.4, 0.57, 0.6, 0.61, 0.74]),
           st.sampled_from([76.2, 78.7, 81.7, 82.2, 86.8, 90.3, 111.9, 137.0, 143.1, 165.2]),
           st.sampled_from([0.05, 0.08, 0.14, 0.15, 0.16, 0.3, 0.37, 0.42, 0.49, 0.5]),
           st.sampled_from([0.13, 0.28, 0.29, 0.32, 0.4, 0.43, 0.48, 0.49, 0.51, 0.83]),
           st.sampled_from([0.34, 0.35, 0.37, 0.4, 0.5, 0.52, 0.57, 0.69, 0.73, 1.13]),
           st.sampled_from([0.01, 0.05, 0.06, 0.07, 0.13, 0.18, 0.22, 0.36, 0.52, 0.61]),
           st.sampled_from([0.05, 0.14, 0.22, 0.36, 0.4, 0.48, 0.54, 0.73, 0.75, 0.85]),
           st.sampled_from([11.43, 12.0, 15.41, 38.16, 67.32, 71.79, 78.02, 83.83, 86.5, 90.23]),
           st.sampled_from([11.87, 11.95, 14.95, 15.91, 18.55, 21.62, 26.33, 34.22, 35.63, 37.48]),
           st.sampled_from([0.16, 0.28, 0.36, 0.39, 0.61, 0.83, 0.86, 0.9, 2.1, 2.31]),
           st.sampled_from([0.08, 0.23, 0.27, 0.32, 0.33, 0.34, 0.44, 0.45, 0.48, 0.54]),
           st.sampled_from([0.15, 0.27, 0.33, 0.38, 0.49, 0.5, 0.52, 0.61, 0.68, 0.71]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.17, 0.25, 0.43, 0.49, 0.75, 0.83, 0.99, 1.17, 1.25, 1.33]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.92, -0.02, 0.69, 0.87, 0.9, 1.01, 1.52, 1.55, 2.96, 3.1]),
           st.floats(min_value=1.701, max_value=9.244, exclude_min=True, allow_nan=False),
           st.sampled_from([-6.46, -5.48, -5.46, -3.57, -3.51, -2.8, -2.73, -2.72, -2.36, -1.06]),
           st.sampled_from([-1.06, -1.0, -0.92, -0.67, -0.57, -0.56, -0.32, -0.2, -0.12, -0.09]),
           st.sampled_from([1.24, 1.95, 1.96, 1.99, 2.01, 2.18, 2.21, 2.59, 2.65, 2.94]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.09, 1.95, 1.98, 2.01, 2.19, 2.2, 2.24, 2.65, 2.67, 2.98]),
           st.sampled_from([8.0, 13.0, 17.0, 19.0, 24.0, 48.0, 83.0, 88.0, 139.0, 281.0]),
           st.sampled_from([3510.0, 4680.0, 11160.0, 12690.0, 16110.0, 18270.0, 20700.0, 42120.0, 45630.0, 95310.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([19.97, 23.21, 26.88, 27.88, 29.2, 32.12, 33.85, 40.6, 50.36, 150.0]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([150.0, 510.06, 685.42, 750.0, 788.39, 873.21, 886.0, 1739.25, 1950.0, 3170.57]),
           st.sampled_from([380.79, 493.24, 500.0, 531.51, 930.05, 969.33, 1054.75, 1431.78, 2450.0, 5020.46]),
           st.floats(min_value=358.971, max_value=448.713, allow_nan=False),
           st.sampled_from([35.79, 91.35, 92.76, 100.55, 128.55, 145.99, 218.07, 309.49, 323.06, 351.13]),
           st.sampled_from([1.87, 1.89, 2.45, 2.5, 2.59, 2.89, 3.04, 4.53, 8.28, 11.45]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=59.922, max_value=65.914, allow_nan=False),
           st.sampled_from([6.24, 6.42, 6.58, 6.64, 7.3, 7.37, 8.16, 14.67, 15.11, 15.31]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([62.0, 93.0, 144.0, 189.0, 257.0, 269.0, 270.0, 390.0, 1574.0, 4810.0]),
           st.sampled_from([30.08, 30.26, 41.79, 1015.08, 1253.2, 1422.2, 1524.24, 1546.64, 1594.45, 1700.15]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.sampled_from([0.0, 13.0, 34.0, 47.0, 51.0, 102.0, 113.0, 160.0, 161.0, 164.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([29.5, 34.31, 35.16, 35.51, 37.01, 46.36, 47.56, 49.77, 60.14, 75.13]),
           st.floats(min_value=7.586, max_value=9.274, allow_nan=False),
           st.sampled_from([1085.0, 1318.0, 1330.0, 1620.0, 1754.5, 1858.0, 2580.0, 2780.0, 5270.0, 12830.0]),
           st.sampled_from([0.12, 0.25, 0.28, 0.34, 0.43, 0.45, 0.53, 0.54, 0.62, 0.67]),
           st.sampled_from([49.8, 64.9, 68.6, 74.5, 74.8, 83.0, 92.3, 93.6, 94.0, 127.4]),
           st.sampled_from([0.1, 0.12, 0.14, 0.18, 0.26, 0.32, 0.5, 0.54, 0.6, 0.61]),
           st.sampled_from([0.06, 0.39, 0.52, 0.53, 0.57, 0.59, 0.68, 0.69, 0.74, 0.76]),
           st.sampled_from([0.08, 0.16, 0.2, 0.21, 0.28, 0.45, 0.52, 0.57, 0.59, 0.66]),
           st.sampled_from([0.05, 0.06, 0.24, 0.32, 0.34, 0.35, 0.39, 0.42, 0.53, 0.56]),
           st.sampled_from([0.03, 0.06, 0.11, 0.28, 0.3, 0.6, 0.67, 0.77, 0.92, 1.1]),
           st.sampled_from([10.96, 14.35, 14.68, 19.01, 24.07, 25.43, 33.26, 48.66, 88.87, 139.5]),
           st.sampled_from([8.03, 14.29, 16.0, 16.43, 16.83, 19.56, 23.5, 30.27, 45.51, 61.45]),
           st.sampled_from([0.37, 0.57, 0.71, 0.82, 0.85, 0.9, 1.74, 1.86, 1.94, 2.42]),
           st.sampled_from([0.06, 0.2, 0.28, 0.3, 0.31, 0.37, 0.38, 0.45, 0.52, 0.55]),
           st.sampled_from([0.07, 0.27, 0.28, 0.37, 0.4, 0.41, 0.48, 0.49, 0.69, 0.71]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.2, 0.22, 0.31, 0.41, 0.85, 0.92, 0.93, 1.02, 1.05, 1.2]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.9, -0.82, -0.11, 0.47, 1.22, 1.24, 1.29, 2.15, 2.33, 2.74]),
           st.sampled_from([1.51, 2.39, 2.88, 3.08, 3.12, 3.55, 3.96, 4.02, 4.04, 13.81]),
           st.sampled_from([-6.13, -6.04, -5.62, -3.45, -2.94, -2.64, -2.25, -2.1, -1.61, 1.28]),
           st.sampled_from([-1.14, -1.07, -1.02, -0.94, -0.83, -0.71, -0.68, -0.26, -0.2, -0.09]),
           st.sampled_from([0.0, 1.09, 1.95, 1.96, 1.97, 2.16, 2.61, 2.93, 2.96, 2.98]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.1, 1.11, 1.23, 1.93, 1.98, 2.17, 2.24, 2.65, 2.91, 2.94]),
           st.sampled_from([8.0, 25.0, 26.0, 44.0, 48.0, 56.0, 65.0, 70.0, 117.0, 122.0]),
           st.sampled_from([810.0, 900.0, 990.0, 1620.0, 3240.0, 5580.0, 6300.0, 6390.0, 7560.0, 19530.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([7.47, 10.98, 13.02, 14.58, 15.41, 16.31, 18.95, 28.11, 28.73, 46.8]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([381.84, 487.5, 514.78, 691.47, 776.21, 1026.16, 1104.54, 1123.61, 1204.16, 3802.96]),
           st.sampled_from([70.71, 180.0, 269.26, 324.5, 353.55, 538.52, 540.0, 559.02, 873.21, 2892.63]),
           st.floats(min_value=448.716, max_value=2277.668, exclude_min=True, allow_nan=False),
           st.sampled_from([47.43, 57.85, 60.17, 70.72, 104.52, 111.11, 117.95, 122.03, 150.6, 351.13]),
           st.sampled_from([2.49, 2.68, 3.07, 3.27, 4.93, 5.22, 7.81, 9.41, 9.42, 15.18]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=59.922, max_value=65.914, allow_nan=False),
           st.sampled_from([6.41, 6.49, 6.84, 7.08, 7.16, 7.6, 7.94, 8.03, 14.86, 14.98]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([13.0, 16.0, 29.0, 50.0, 59.0, 128.0, 364.0, 735.0, 2971.0, 3155.0]),
           st.sampled_from([226.81, 227.23, 361.68, 386.31, 454.66, 474.38, 497.9, 852.49, 1123.09, 1438.13]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.sampled_from([2.0, 10.0, 18.0, 23.0, 68.0, 88.0, 96.0, 97.0, 114.0, 140.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([23.11, 23.92, 31.5, 38.65, 38.9, 41.22, 41.53, 52.63, 55.83, 60.03]),
           st.floats(min_value=9.277, max_value=12.359, exclude_min=True, allow_nan=False),
           st.sampled_from([1400.0, 1551.5, 1858.0, 2280.0, 3483.0, 3610.0, 3660.0, 6998.0, 7430.0, 17380.0]),
           st.sampled_from([0.08, 0.15, 0.18, 0.19, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28]),
           st.sampled_from([79.2, 86.0, 86.1, 101.6, 103.8, 129.6, 162.1, 276.0, 402.1, 496.7]),
           st.sampled_from([0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.24, 0.31, 0.36, 0.43]),
           st.sampled_from([0.16, 0.22, 0.3, 0.32, 0.33, 0.37, 0.38, 0.51, 0.53, 0.67]),
           st.sampled_from([0.3, 0.33, 0.34, 0.35, 0.36, 0.39, 0.49, 0.63, 0.69, 0.85]),
           st.sampled_from([0.08, 0.09, 0.1, 0.13, 0.15, 0.18, 0.2, 0.21, 0.23, 0.35]),
           st.sampled_from([0.15, 0.19, 0.26, 0.3, 0.33, 0.34, 0.44, 0.54, 0.6, 0.63]),
           st.sampled_from([18.19, 20.73, 24.37, 24.48, 26.35, 29.43, 120.22, 125.35, 149.87, 207.31]),
           st.sampled_from([7.28, 8.32, 13.31, 19.93, 22.56, 24.47, 43.96, 50.99, 73.4, 74.88]),
           st.sampled_from([0.33, 0.47, 0.74, 0.88, 0.94, 0.97, 1.76, 1.82, 1.84, 1.97]),
           st.sampled_from([0.1, 0.12, 0.15, 0.19, 0.22, 0.27, 0.28, 0.31, 0.32, 0.33]),
           st.sampled_from([0.19, 0.2, 0.21, 0.23, 0.24, 0.25, 0.26, 0.27, 0.42, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.36, 0.4, 0.46, 0.52, 0.54, 0.75, 0.87, 0.88, 0.9, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.26, 0.47, 0.55, 0.56, 0.73, 1.01, 1.22, 1.48, 1.79, 1.99]),
           st.sampled_from([1.64, 1.86, 2.57, 2.95, 3.56, 4.79, 5.99, 6.78, 7.31, 10.63]),
           st.sampled_from([-5.18, -3.89, -3.71, -3.54, -3.5, -3.36, -2.72, -1.95, -1.92, -1.27]),
           st.sampled_from([-0.75, -0.51, -0.36, -0.34, -0.32, -0.3, -0.21, -0.16, -0.09, -0.07]),
           st.sampled_from([1.09, 1.1, 1.22, 1.23, 1.94, 1.95, 1.96, 2.18, 2.2, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.23, 1.94, 2.16, 2.17, 2.18, 2.19, 2.59, 2.91, 2.92]),
           st.sampled_from([10.0, 12.0, 13.0, 28.0, 32.0, 36.0, 38.0, 39.0, 183.0, 310.0]),
           st.sampled_from([450.0, 630.0, 990.0, 1080.0, 1530.0, 2160.0, 2700.0, 8100.0, 8730.0, 10080.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.6, 16.37, 16.47, 24.37, 28.07, 38.8, 40.67, 58.27, 77.39, 87.16]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 471.7, 685.42, 1569.24, 2672.86, 2850.0, 3471.31, 3936.41, 4724.79, 5650.88]),
           st.sampled_from([223.61, 269.26, 360.0, 403.11, 603.74, 655.21, 761.58, 829.76, 1000.0, 3059.41]),
           st.floats(min_value=448.716, max_value=2277.668, exclude_min=True, allow_nan=False),
           st.sampled_from([41.66, 45.23, 50.12, 55.93, 88.6, 144.97, 145.92, 195.92, 348.7, 349.26]),
           st.sampled_from([1.45, 1.67, 2.64, 3.2, 6.32, 6.63, 8.69, 8.97, 10.72, 13.05]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=59.922, max_value=65.914, allow_nan=False),
           st.sampled_from([6.46, 6.89, 7.18, 7.24, 7.29, 7.35, 7.55, 8.07, 14.92, 14.95]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([25.0, 67.0, 71.0, 72.0, 91.0, 216.0, 253.0, 320.0, 1366.0, 1655.0]),
           st.sampled_from([5.7, 22.11, 25.1, 43.68, 122.83, 152.8, 575.19, 858.28, 874.62, 1208.25]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.floats(min_value=73.2, max_value=91.49, allow_nan=False),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([35.04, 35.51, 39.3, 39.35, 40.0, 40.08, 42.22, 42.69, 61.69, 70.14]),
           st.sampled_from([2.33, 4.58, 5.23, 6.29, 7.38, 8.17, 8.65, 8.94, 15.0, 19.06]),
           st.sampled_from([1360.0, 1505.0, 1752.0, 1870.0, 2583.0, 2620.0, 3340.0, 7780.0, 8110.0, 10041.0]),
           st.sampled_from([0.09, 0.17, 0.18, 0.22, 0.28, 0.31, 0.36, 0.39, 0.47, 0.74]),
           st.sampled_from([60.6, 62.8, 63.9, 82.2, 84.6, 89.3, 97.1, 122.6, 128.0, 146.4]),
           st.sampled_from([0.05, 0.08, 0.1, 0.14, 0.2, 0.38, 0.41, 0.48, 0.49, 0.5]),
           st.sampled_from([0.07, 0.2, 0.25, 0.33, 0.35, 0.42, 0.48, 0.49, 0.59, 0.68]),
           st.sampled_from([0.09, 0.15, 0.31, 0.43, 0.45, 0.58, 0.61, 0.62, 0.72, 1.08]),
           st.sampled_from([0.03, 0.06, 0.07, 0.1, 0.19, 0.22, 0.28, 0.29, 0.46, 0.52]),
           st.sampled_from([0.14, 0.4, 0.42, 0.54, 0.61, 0.65, 0.66, 0.76, 0.8, 0.84]),
           st.sampled_from([12.47, 25.83, 29.47, 40.19, 53.15, 69.31, 76.63, 93.1, 105.43, 120.98]),
           st.sampled_from([2.41, 11.61, 11.76, 11.95, 15.91, 21.68, 28.55, 44.15, 61.45, 351.0]),
           st.sampled_from([0.21, 0.39, 0.55, 0.61, 0.98, 1.0, 1.77, 1.97, 2.19, 2.34]),
           st.sampled_from([0.19, 0.22, 0.23, 0.29, 0.3, 0.32, 0.33, 0.43, 0.47, 0.5]),
           st.sampled_from([0.02, 0.03, 0.05, 0.16, 0.2, 0.26, 0.28, 0.3, 0.36, 0.71]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.23, 0.28, 0.37, 0.46, 0.49, 0.59, 0.76, 0.98, 1.08, 1.27]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.63, -0.16, -0.06, -0.03, 1.11, 1.4, 1.43, 1.72, 1.98, 2.04]),
           st.sampled_from([1.65, 1.97, 2.33, 2.55, 2.82, 3.74, 4.26, 4.39, 5.15, 6.09]),
           st.sampled_from([-6.01, -5.39, -5.33, -4.69, -3.44, -3.28, -3.14, -1.94, -1.43, -1.34]),
           st.sampled_from([-1.23, -1.02, -0.92, -0.85, -0.37, -0.35, -0.32, -0.24, -0.19, 0.0]),
           st.sampled_from([0.0, 1.22, 1.93, 1.97, 1.99, 2.16, 2.21, 2.22, 2.62, 2.9]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([0.0, 1.11, 1.24, 1.99, 2.18, 2.22, 2.6, 2.61, 2.62, 2.9]),
           st.sampled_from([5.0, 8.0, 26.0, 52.0, 60.0, 63.0, 78.0, 85.0, 104.0, 172.0]),
           st.sampled_from([630.0, 5040.0, 6660.0, 7470.0, 8820.0, 12150.0, 14490.0, 16920.0, 17280.0, 20700.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([6.81, 9.09, 10.96, 13.18, 26.62, 30.27, 44.25, 49.8, 56.8, 63.44]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([300.0, 471.7, 742.16, 814.98, 956.07, 1000.0, 1012.42, 1557.24, 1588.24, 1659.52]),
           st.sampled_from([291.55, 335.41, 447.21, 538.52, 540.0, 640.31, 738.24, 926.18, 975.72, 2362.39]),
           st.sampled_from([35.36, 129.41, 130.02, 145.38, 161.15, 227.78, 247.55, 327.22, 396.49, 447.31]),
           st.sampled_from([46.54, 55.48, 68.03, 92.94, 97.07, 102.1, 119.29, 181.45, 225.16, 292.9]),
           st.sampled_from([1.92, 2.3, 4.52, 5.14, 5.4, 6.25, 7.65, 8.39, 12.0, 14.63]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=65.917, max_value=66.023, exclude_min=True, allow_nan=False),
           st.sampled_from([6.29, 6.4, 6.51, 6.83, 6.89, 7.93, 14.8, 14.98, 15.08, 15.44]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_24(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_24']['n_samples'] += 1
        self.data['tests']['test_24']['samples'].append(x_test)
        self.data['tests']['test_24']['y_expected'].append(y_expected[0])
        self.data['tests']['test_24']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 29.0, 31.0, 71.0, 96.0, 116.0, 270.0, 1201.0, 2558.0, 2971.0]),
           st.sampled_from([7.7, 115.22, 273.61, 454.66, 497.9, 903.02, 920.2, 1020.91, 1118.08, 1506.09]),
           st.floats(min_value=292.244, max_value=365.054, allow_nan=False),
           st.floats(min_value=91.51, max_value=109.2, exclude_min=True, allow_nan=False),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([22.79, 24.09, 27.23, 40.88, 41.08, 42.37, 44.67, 46.29, 53.53, 54.2]),
           st.sampled_from([4.72, 5.01, 5.46, 6.44, 6.92, 9.31, 11.54, 12.73, 15.7, 16.85]),
           st.sampled_from([1145.5, 1570.0, 1710.0, 3620.0, 4748.0, 7067.0, 15106.0, 15880.0, 18030.0, 32401.5]),
           st.sampled_from([0.08, 0.09, 0.1, 0.15, 0.18, 0.19, 0.22, 0.24, 0.28, 0.3]),
           st.sampled_from([86.0, 86.5, 107.1, 113.4, 121.9, 124.9, 125.0, 126.4, 214.7, 427.4]),
           st.sampled_from([0.15, 0.16, 0.18, 0.19, 0.22, 0.29, 0.31, 0.34, 0.35, 0.36]),
           st.sampled_from([0.21, 0.27, 0.29, 0.3, 0.38, 0.39, 0.41, 0.42, 0.53, 0.67]),
           st.sampled_from([0.23, 0.24, 0.34, 0.36, 0.46, 0.62, 0.63, 0.65, 0.66, 1.02]),
           st.sampled_from([0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.18, 0.27, 0.35, 0.5]),
           st.sampled_from([0.14, 0.15, 0.16, 0.2, 0.25, 0.3, 0.31, 0.42, 0.44, 0.6]),
           st.sampled_from([11.66, 17.07, 20.38, 20.47, 24.37, 29.43, 120.22, 136.77, 180.8, 431.19]),
           st.sampled_from([8.32, 9.14, 9.22, 14.45, 14.72, 19.62, 25.06, 34.76, 54.12, 74.88]),
           st.sampled_from([0.34, 0.36, 0.44, 0.79, 0.92, 0.94, 1.75, 1.82, 1.85, 1.97]),
           st.sampled_from([0.14, 0.16, 0.19, 0.24, 0.25, 0.27, 0.28, 0.32, 0.33, 0.34]),
           st.sampled_from([0.13, 0.14, 0.18, 0.23, 0.28, 0.31, 0.32, 0.34, 0.37, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.4, 0.42, 0.48, 0.78, 0.86, 0.87, 0.91, 0.97, 1.02]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.21, 0.26, 0.39, 0.47, 0.59, 0.69, 0.78, 0.84, 1.01, 1.55]),
           st.sampled_from([1.98, 2.67, 3.36, 3.52, 3.84, 5.99, 6.78, 7.31, 8.68, 10.98]),
           st.sampled_from([-7.76, -3.86, -3.58, -3.53, -3.38, -3.24, -3.19, -3.07, -2.37, -1.92]),
           st.sampled_from([-1.02, -0.89, -0.74, -0.51, -0.5, -0.41, -0.32, -0.18, -0.14, -0.07]),
           st.sampled_from([1.09, 1.23, 1.94, 1.96, 2.17, 2.18, 2.19, 2.2, 2.59, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.1, 1.22, 1.23, 1.94, 1.95, 1.96, 2.18, 2.19, 2.59, 2.91]),
           st.sampled_from([17.0, 21.0, 28.0, 32.0, 36.0, 38.0, 39.0, 46.0, 78.0, 96.0]),
           st.sampled_from([630.0, 900.0, 990.0, 1080.0, 2070.0, 2610.0, 2790.0, 3060.0, 8100.0, 10350.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([9.71, 18.26, 23.4, 24.34, 29.3, 34.35, 48.75, 58.27, 78.14, 87.16]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([0.0, 403.89, 450.0, 608.28, 851.47, 1281.6, 1569.24, 2930.19, 4724.79, 5650.88]),
           st.sampled_from([90.0, 250.0, 316.23, 320.4, 484.66, 492.44, 603.74, 636.4, 829.76, 1000.0]),
           st.sampled_from([0.0, 67.5, 120.21, 135.0, 176.72, 204.18, 291.6, 453.21, 526.56, 763.16]),
           st.sampled_from([50.12, 52.22, 84.35, 124.72, 144.97, 145.92, 160.11, 249.81, 255.81, 460.42]),
           st.sampled_from([1.37, 1.67, 3.9, 4.0, 4.6, 5.88, 6.44, 6.63, 11.32, 11.89]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.floats(min_value=65.917, max_value=66.023, exclude_min=True, allow_nan=False),
           st.sampled_from([6.3, 6.54, 6.96, 7.18, 7.55, 7.58, 7.84, 7.95, 8.07, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_25(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_25']['n_samples'] += 1
        self.data['tests']['test_25']['samples'].append(x_test)
        self.data['tests']['test_25']['y_expected'].append(y_expected[0])
        self.data['tests']['test_25']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([16.0, 17.0, 64.0, 95.0, 107.0, 128.0, 453.0, 735.0, 1201.0, 2558.0]),
           st.sampled_from([7.7, 21.23, 221.79, 227.23, 343.63, 361.68, 474.38, 920.2, 1020.91, 1438.13]),
           st.floats(min_value=365.057, max_value=365.224, exclude_min=True, allow_nan=False),
           st.sampled_from([23.0, 42.0, 52.0, 66.0, 68.0, 88.0, 110.0, 124.0, 140.0, 155.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([22.79, 38.9, 41.22, 42.16, 42.4, 47.81, 49.35, 53.69, 54.2, 70.65]),
           st.sampled_from([3.66, 6.74, 7.97, 10.48, 11.38, 12.73, 13.54, 13.97, 15.7, 16.85]),
           st.sampled_from([1850.0, 1858.0, 2000.0, 3340.0, 3720.0, 17380.0, 18030.0, 21569.0, 22260.0, 29780.0]),
           st.sampled_from([0.09, 0.1, 0.18, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.34]),
           st.sampled_from([79.2, 83.0, 86.1, 86.5, 101.3, 124.9, 152.4, 162.1, 187.0, 402.1]),
           st.sampled_from([0.13, 0.15, 0.2, 0.21, 0.23, 0.24, 0.26, 0.33, 0.34, 0.35]),
           st.sampled_from([0.14, 0.2, 0.25, 0.27, 0.29, 0.33, 0.37, 0.38, 0.51, 0.67]),
           st.sampled_from([0.3, 0.39, 0.42, 0.46, 0.5, 0.53, 0.64, 0.85, 0.96, 1.02]),
           st.sampled_from([0.07, 0.11, 0.12, 0.14, 0.18, 0.24, 0.3, 0.32, 0.35, 0.5]),
           st.sampled_from([0.08, 0.15, 0.2, 0.3, 0.35, 0.37, 0.42, 0.44, 0.55, 0.64]),
           st.sampled_from([11.66, 19.85, 20.38, 20.73, 21.74, 24.37, 29.4, 29.43, 149.87, 207.31]),
           st.sampled_from([7.3, 7.64, 8.09, 9.22, 12.11, 14.72, 16.73, 24.18, 33.47, 54.12]),
           st.sampled_from([0.15, 0.33, 0.47, 0.84, 0.88, 0.94, 0.97, 1.06, 1.84, 2.01]),
           st.sampled_from([0.1, 0.14, 0.16, 0.17, 0.19, 0.23, 0.24, 0.26, 0.33, 0.34]),
           st.sampled_from([0.13, 0.18, 0.2, 0.22, 0.23, 0.28, 0.29, 0.31, 0.43, 0.49]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.24, 0.4, 0.46, 0.5, 0.78, 0.81, 0.87, 0.92, 1.02]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.1, 0.12, 0.29, 0.55, 0.73, 1.08, 1.11, 1.48, 1.99, 2.5]),
           st.sampled_from([2.57, 2.95, 3.36, 3.79, 4.02, 4.35, 4.72, 4.79, 10.98, 11.08]),
           st.sampled_from([-7.76, -6.77, -5.55, -3.68, -3.5, -3.1, -2.51, -2.37, -0.87, -0.65]),
           st.sampled_from([-0.75, -0.74, -0.63, -0.51, -0.33, -0.28, -0.25, -0.14, -0.12, -0.07]),
           st.sampled_from([1.09, 1.23, 1.94, 1.95, 2.16, 2.17, 2.18, 2.2, 2.59, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.1, 1.94, 1.95, 1.96, 2.16, 2.18, 2.19, 2.2, 2.59, 2.92]),
           st.sampled_from([11.0, 19.0, 21.0, 22.0, 26.0, 28.0, 32.0, 48.0, 150.0, 465.0]),
           st.sampled_from([900.0, 1260.0, 1800.0, 2070.0, 2160.0, 2700.0, 6750.0, 8100.0, 8730.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([14.7, 19.31, 21.97, 24.34, 24.37, 29.3, 40.67, 58.27, 78.14, 87.16]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 471.7, 540.0, 608.28, 742.16, 851.47, 2101.07, 2534.42, 2846.05, 2850.0]),
           st.sampled_from([127.28, 141.42, 250.0, 269.26, 300.0, 316.23, 484.66, 492.44, 636.4, 829.76]),
           st.sampled_from([120.21, 170.58, 182.83, 184.94, 190.67, 193.18, 345.47, 623.26, 878.29, 2043.9]),
           st.sampled_from([41.66, 47.73, 124.72, 145.92, 160.11, 195.92, 249.81, 255.81, 349.26, 477.23]),
           st.sampled_from([0.0, 1.47, 1.67, 2.7, 3.2, 6.95, 7.33, 7.74, 8.97, 10.99]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.sampled_from([36.14, 36.18, 65.61, 65.67, 65.72, 65.73, 65.79, 65.8, 65.82, 66.06]),
           st.sampled_from([6.29, 6.54, 6.89, 7.29, 7.32, 7.41, 7.58, 7.65, 14.92, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_26(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_26']['n_samples'] += 1
        self.data['tests']['test_26']['samples'].append(x_test)
        self.data['tests']['test_26']['y_expected'].append(y_expected[0])
        self.data['tests']['test_26']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=18.51, max_value=85.2, exclude_min=True, allow_nan=False),
           st.sampled_from([26.0, 37.0, 135.0, 211.0, 276.0, 346.0, 363.0, 649.0, 725.0, 743.0]),
           st.sampled_from([13.1, 30.26, 45.17, 153.39, 1292.48, 1315.02, 1355.46, 1388.78, 1426.96, 1444.54]),
           st.floats(min_value=365.897, max_value=837.631, exclude_min=True, allow_nan=False),
           st.sampled_from([3.0, 37.0, 41.0, 63.0, 65.0, 116.0, 148.0, 154.0, 170.0, 176.0]),
           st.floats(min_value=74531.01, max_value=14322624.8, exclude_min=True, allow_nan=False),
           st.sampled_from([24.95, 29.15, 29.53, 32.69, 37.34, 47.33, 47.74, 48.37, 57.6, 70.21]),
           st.sampled_from([1.23, 7.52, 7.59, 8.24, 8.36, 9.79, 10.74, 11.65, 16.51, 21.29]),
           st.sampled_from([1290.0, 1380.0, 1572.0, 2090.0, 2403.0, 2490.0, 2650.0, 3160.0, 3670.0, 9080.0]),
           st.sampled_from([0.02, 0.07, 0.08, 0.16, 0.19, 0.2, 0.23, 0.24, 0.42, 0.62]),
           st.sampled_from([64.4, 67.3, 71.0, 74.0, 77.2, 100.3, 101.6, 135.4, 187.7, 342.2]),
           st.sampled_from([0.1, 0.17, 0.25, 0.32, 0.33, 0.4, 0.42, 0.51, 0.59, 0.62]),
           st.sampled_from([0.14, 0.31, 0.34, 0.37, 0.42, 0.47, 0.54, 0.68, 0.7, 0.79]),
           st.sampled_from([0.09, 0.11, 0.15, 0.26, 0.42, 0.54, 0.56, 0.57, 0.64, 0.93]),
           st.sampled_from([0.01, 0.02, 0.07, 0.11, 0.12, 0.13, 0.22, 0.25, 0.32, 0.39]),
           st.sampled_from([0.07, 0.09, 0.14, 0.19, 0.22, 0.26, 0.4, 0.69, 0.81, 0.86]),
           st.sampled_from([10.32, 11.13, 18.07, 19.94, 21.5, 28.87, 45.98, 64.4, 64.86, 110.49]),
           st.sampled_from([2.81, 4.35, 11.92, 12.12, 12.23, 19.76, 20.65, 23.67, 28.03, 58.22]),
           st.sampled_from([0.27, 0.34, 0.68, 0.72, 0.94, 0.96, 1.12, 2.02, 2.04, 2.13]),
           st.sampled_from([0.02, 0.08, 0.09, 0.13, 0.17, 0.22, 0.31, 0.41, 0.5, 0.55]),
           st.sampled_from([0.04, 0.09, 0.23, 0.25, 0.26, 0.29, 0.33, 0.38, 0.47, 0.7]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.22, 0.23, 0.24, 0.48, 0.76, 1.02, 1.2, 1.21, 1.25, 1.27]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.63, -0.05, 1.14, 1.17, 1.54, 1.59, 1.94, 2.09, 2.18, 2.52]),
           st.sampled_from([1.87, 2.34, 2.71, 3.3, 3.31, 3.35, 5.15, 6.89, 15.4, 16.46]),
           st.sampled_from([-7.18, -7.08, -6.18, -5.39, -3.38, -3.37, -3.3, -1.83, -0.8, -0.61]),
           st.sampled_from([-1.27, -1.21, -1.19, -0.97, -0.96, -0.68, -0.4, -0.29, -0.13, 0.0]),
           st.sampled_from([1.11, 1.21, 2.01, 2.17, 2.23, 2.24, 2.63, 2.93, 2.94, 2.98]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.1, 1.22, 1.96, 1.99, 2.62, 2.65, 2.92, 2.94, 2.95, 2.98]),
           st.sampled_from([11.0, 34.0, 38.0, 58.0, 77.0, 87.0, 89.0, 241.0, 302.0, 310.0]),
           st.sampled_from([630.0, 1620.0, 1980.0, 2250.0, 2790.0, 3060.0, 4770.0, 5040.0, 12330.0, 51120.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([9.09, 13.22, 20.17, 22.37, 26.35, 26.43, 27.24, 30.2, 45.88, 46.67]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([500.0, 510.06, 540.83, 610.33, 764.85, 1049.57, 1328.53, 1609.97, 1822.36, 7274.96]),
           st.sampled_from([360.0, 403.11, 450.0, 456.21, 458.91, 738.24, 1012.42, 1140.18, 1662.08, 2050.61]),
           st.sampled_from([106.07, 123.75, 147.06, 150.0, 153.56, 182.92, 200.0, 221.26, 232.17, 524.86]),
           st.sampled_from([25.98, 46.48, 55.24, 57.02, 72.28, 81.8, 110.25, 138.35, 173.0, 340.8]),
           st.sampled_from([0.53, 1.51, 2.52, 2.55, 3.02, 3.04, 3.2, 6.72, 8.49, 15.3]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=19088.936, max_value=23348.294, allow_nan=False),
           st.sampled_from([36.19, 36.47, 36.55, 65.68, 65.95, 65.98, 66.06, 66.09, 66.17, 66.28]),
           st.sampled_from([5.86, 6.52, 6.53, 7.3, 7.58, 8.01, 14.76, 14.81, 14.86, 15.32]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_27(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_27']['n_samples'] += 1
        self.data['tests']['test_27']['samples'].append(x_test)
        self.data['tests']['test_27']['y_expected'].append(y_expected[0])
        self.data['tests']['test_27']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([5.0, 6.0, 10.0, 13.0, 16.0, 23.0, 47.0, 118.0, 125.0, 140.0]),
           st.sampled_from([13.0, 17.0, 29.0, 50.0, 64.0, 114.0, 270.0, 354.0, 364.0, 3155.0]),
           st.sampled_from([21.23, 104.75, 115.22, 120.26, 165.49, 361.68, 369.85, 386.31, 805.86, 1506.09]),
           st.sampled_from([276.56, 287.06, 295.65, 305.06, 544.91, 624.0, 630.8, 1167.85, 1184.91, 1337.49]),
           st.floats(min_value=78.0, max_value=97.49, allow_nan=False),
           st.sampled_from([105300.0, 137700.0, 142500.0, 150468.0, 160000.0, 251100.0, 287500.0, 497812.0, 793800.0, 2948400.0]),
           st.sampled_from([24.65, 30.41, 40.46, 41.22, 43.9, 44.67, 50.33, 51.39, 52.11, 60.03]),
           st.sampled_from([3.66, 5.89, 5.94, 6.13, 6.92, 7.08, 9.31, 10.48, 13.54, 16.89]),
           st.sampled_from([1011.0, 1145.5, 1400.0, 1461.0, 1858.0, 3620.0, 3660.0, 21569.0, 22260.0, 32401.5]),
           st.sampled_from([0.08, 0.09, 0.1, 0.11, 0.15, 0.22, 0.24, 0.25, 0.27, 0.34]),
           st.sampled_from([70.0, 81.9, 83.0, 86.5, 103.8, 121.9, 152.4, 159.5, 195.2, 427.4]),
           st.sampled_from([0.14, 0.15, 0.16, 0.17, 0.22, 0.25, 0.28, 0.29, 0.34, 0.35]),
           st.sampled_from([0.22, 0.26, 0.3, 0.32, 0.33, 0.34, 0.35, 0.37, 0.51, 0.79]),
           st.sampled_from([0.33, 0.38, 0.53, 0.54, 0.64, 0.66, 0.74, 0.84, 0.96, 1.02]),
           st.sampled_from([0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.17, 0.24, 0.5]),
           st.sampled_from([0.2, 0.21, 0.26, 0.29, 0.3, 0.31, 0.33, 0.64, 0.82, 0.87]),
           st.sampled_from([16.06, 19.85, 20.73, 24.37, 29.4, 37.51, 55.25, 118.11, 235.92, 431.19]),
           st.sampled_from([7.64, 8.09, 11.96, 13.1, 16.73, 19.27, 49.55, 50.37, 54.12, 73.4]),
           st.sampled_from([0.15, 0.3, 0.36, 0.38, 0.65, 0.88, 0.92, 1.84, 1.89, 2.14]),
           st.sampled_from([0.1, 0.16, 0.17, 0.22, 0.23, 0.24, 0.25, 0.28, 0.31, 0.32]),
           st.sampled_from([0.13, 0.14, 0.21, 0.24, 0.28, 0.29, 0.31, 0.34, 0.42, 0.52]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.19, 0.22, 0.24, 0.39, 0.44, 0.47, 0.97, 1.02, 1.1]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.floats(min_value=-0.674, max_value=-0.396, allow_nan=False),
           st.sampled_from([3.36, 3.52, 4.06, 4.35, 4.36, 5.05, 5.99, 7.06, 7.31, 10.63]),
           st.sampled_from([-5.18, -3.68, -3.38, -3.24, -3.19, -2.24, -1.95, -1.63, -1.27, -0.65]),
           st.sampled_from([-1.02, -0.89, -0.74, -0.43, -0.36, -0.3, -0.25, -0.22, -0.17, -0.14]),
           st.sampled_from([1.09, 1.22, 1.23, 1.95, 1.96, 2.17, 2.18, 2.19, 2.2, 2.91]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.23, 1.94, 1.95, 1.96, 2.16, 2.17, 2.18, 2.19, 2.59]),
           st.sampled_from([12.0, 15.0, 17.0, 18.0, 36.0, 39.0, 46.0, 61.0, 78.0, 133.0]),
           st.sampled_from([720.0, 990.0, 1080.0, 1170.0, 1620.0, 2250.0, 2610.0, 3060.0, 6750.0, 8730.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([9.71, 16.0, 16.47, 18.24, 18.26, 21.39, 31.97, 40.67, 61.69, 87.16]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([0.0, 471.7, 608.28, 742.16, 1288.6, 1400.0, 2930.19, 3448.19, 3936.41, 5650.88]),
           st.sampled_from([0.0, 90.0, 141.42, 282.84, 484.66, 603.74, 636.4, 761.58, 813.94, 829.76]),
           st.sampled_from([67.5, 110.77, 135.0, 164.58, 225.0, 256.77, 345.47, 368.74, 623.26, 1209.38]),
           st.sampled_from([41.98, 44.75, 45.23, 52.22, 55.93, 60.55, 73.11, 124.06, 124.72, 195.92]),
           st.sampled_from([1.45, 1.47, 1.67, 2.7, 2.85, 3.9, 4.51, 6.44, 7.33, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.722, max_value=65.664, allow_nan=False),
           st.sampled_from([5.94, 6.1, 7.22, 7.26, 7.28, 7.35, 7.84, 7.95, 8.07, 15.01]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_28(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_28']['n_samples'] += 1
        self.data['tests']['test_28']['samples'].append(x_test)
        self.data['tests']['test_28']['y_expected'].append(y_expected[0])
        self.data['tests']['test_28']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([14.0, 24.0, 52.0, 79.0, 98.0, 105.0, 135.0, 163.0, 180.0, 257.0]),
           st.sampled_from([67.0, 124.0, 129.0, 170.0, 191.0, 281.0, 350.0, 405.0, 3624.0, 7232.0]),
           st.sampled_from([11.74, 43.56, 49.27, 80.87, 102.09, 475.5, 687.35, 798.03, 844.44, 956.77]),
           st.sampled_from([1.0, 178.25, 195.42, 435.8, 523.0, 530.8, 581.31, 894.53, 1260.37, 1472.03]),
           st.floats(min_value=78.0, max_value=97.49, allow_nan=False),
           st.sampled_from([80156.0, 136406.0, 155000.0, 447500.0, 477900.0, 877500.0, 1287900.0, 5742900.0, 6257500.0, 35891720.0]),
           st.sampled_from([33.35, 33.47, 35.81, 35.96, 37.94, 38.9, 39.91, 41.29, 52.32, 54.58]),
           st.sampled_from([6.17, 6.59, 6.94, 7.35, 8.3, 8.58, 10.66, 10.7, 11.76, 15.33]),
           st.sampled_from([810.0, 2218.0, 2350.0, 2456.0, 2540.0, 3423.5, 3780.0, 5360.0, 5944.0, 21289.5]),
           st.sampled_from([0.03, 0.1, 0.18, 0.27, 0.31, 0.34, 0.37, 0.39, 0.58, 0.62]),
           st.sampled_from([68.3, 79.3, 85.3, 90.6, 99.3, 105.1, 118.7, 126.4, 132.0, 165.6]),
           st.sampled_from([0.02, 0.11, 0.17, 0.19, 0.2, 0.27, 0.28, 0.48, 0.5, 0.54]),
           st.sampled_from([0.13, 0.14, 0.25, 0.3, 0.32, 0.38, 0.54, 0.57, 0.68, 0.78]),
           st.sampled_from([0.1, 0.39, 0.43, 0.45, 0.47, 0.49, 0.81, 0.82, 0.87, 0.94]),
           st.sampled_from([0.01, 0.08, 0.09, 0.12, 0.15, 0.22, 0.23, 0.26, 0.31, 0.35]),
           st.sampled_from([0.15, 0.41, 0.45, 0.49, 0.52, 0.67, 0.69, 0.74, 0.79, 1.12]),
           st.sampled_from([19.37, 20.41, 24.99, 25.55, 38.03, 56.16, 74.0, 139.5, 160.27, 391.19]),
           st.sampled_from([2.62, 9.57, 17.68, 17.98, 18.52, 28.23, 30.44, 30.96, 35.16, 35.24]),
           st.sampled_from([0.15, 0.66, 0.73, 0.97, 1.0, 1.11, 1.14, 1.96, 2.05, 2.17]),
           st.sampled_from([0.03, 0.04, 0.11, 0.13, 0.14, 0.33, 0.35, 0.42, 0.51, 0.65]),
           st.sampled_from([0.05, 0.21, 0.26, 0.33, 0.34, 0.36, 0.43, 0.45, 0.48, 0.72]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.27, 0.49, 0.53, 0.78, 0.87, 0.88, 0.92, 0.97, 1.01, 1.29]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.floats(min_value=-0.393, max_value=0.829, exclude_min=True, allow_nan=False),
           st.sampled_from([2.48, 2.6, 2.74, 3.49, 3.59, 4.16, 4.79, 7.09, 11.5, 13.29]),
           st.sampled_from([-6.39, -5.65, -3.49, -3.4, -3.12, -2.43, -1.51, -1.46, -1.43, -1.2]),
           st.sampled_from([-1.37, -1.25, -1.02, -0.98, -0.9, -0.87, -0.84, -0.75, -0.61, -0.12]),
           st.sampled_from([1.09, 1.1, 1.94, 1.95, 1.99, 2.18, 2.19, 2.23, 2.64, 2.9]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.1, 1.11, 1.21, 1.95, 1.96, 1.99, 2.18, 2.61, 2.65, 2.92]),
           st.sampled_from([22.0, 28.0, 30.0, 32.0, 46.0, 61.0, 79.0, 96.0, 104.0, 291.0]),
           st.sampled_from([540.0, 1080.0, 1620.0, 2250.0, 3330.0, 3690.0, 5490.0, 9540.0, 12150.0, 20700.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([10.34, 13.04, 16.47, 20.35, 27.1, 28.31, 39.24, 52.58, 55.23, 112.45]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([487.5, 510.06, 569.21, 807.77, 814.98, 853.81, 912.41, 996.24, 1150.0, 2452.04]),
           st.sampled_from([391.51, 403.89, 424.26, 768.96, 813.94, 890.22, 1011.19, 1012.42, 4816.64, 8789.57]),
           st.sampled_from([145.78, 193.37, 195.83, 227.78, 227.93, 229.1, 232.42, 300.09, 637.97, 1245.62]),
           st.sampled_from([35.79, 36.74, 54.91, 61.91, 88.49, 103.73, 105.58, 126.14, 155.48, 300.07]),
           st.sampled_from([1.63, 3.38, 3.58, 4.14, 4.47, 5.44, 5.53, 6.62, 7.81, 14.51]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.floats(min_value=59.722, max_value=65.664, allow_nan=False),
           st.sampled_from([5.91, 6.24, 7.48, 7.61, 7.75, 7.79, 14.55, 14.65, 14.66, 15.27]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_29(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_29']['n_samples'] += 1
        self.data['tests']['test_29']['samples'].append(x_test)
        self.data['tests']['test_29']['y_expected'].append(y_expected[0])
        self.data['tests']['test_29']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([7.0, 31.0, 36.0, 37.0, 97.0, 109.0, 179.0, 181.0, 188.0, 192.0]),
           st.sampled_from([24.0, 48.0, 65.0, 83.0, 86.0, 91.0, 214.0, 219.0, 346.0, 1059.0]),
           st.sampled_from([7.7, 14.12, 36.05, 77.0, 79.17, 79.84, 153.81, 949.89, 1596.73, 1702.82]),
           st.sampled_from([348.0, 398.5, 547.92, 720.4, 891.77, 969.47, 1341.12, 1409.43, 1469.83, 1483.52]),
           st.floats(min_value=78.0, max_value=97.49, allow_nan=False),
           st.sampled_from([135000.0, 192500.0, 250312.0, 291600.0, 322500.0, 327500.0, 397500.0, 630000.0, 702500.0, 874800.0]),
           st.sampled_from([24.06, 32.65, 32.86, 34.99, 36.4, 43.95, 44.1, 48.53, 50.47, 74.27]),
           st.sampled_from([3.6, 5.42, 5.45, 6.07, 8.24, 8.56, 8.7, 8.93, 11.15, 11.56]),
           st.sampled_from([778.0, 1505.0, 2043.0, 2410.0, 2560.0, 2948.0, 2960.0, 3600.0, 5680.0, 6361.0]),
           st.sampled_from([0.02, 0.03, 0.06, 0.07, 0.17, 0.2, 0.21, 0.3, 0.36, 0.38]),
           st.sampled_from([55.7, 61.0, 71.8, 73.9, 95.1, 99.4, 133.8, 139.5, 149.6, 151.4]),
           st.sampled_from([0.07, 0.08, 0.2, 0.24, 0.27, 0.35, 0.44, 0.5, 0.54, 0.61]),
           st.sampled_from([0.05, 0.26, 0.27, 0.29, 0.34, 0.36, 0.41, 0.59, 0.68, 0.77]),
           st.sampled_from([0.12, 0.25, 0.55, 0.75, 0.93, 0.94, 0.95, 1.01, 1.05, 1.19]),
           st.sampled_from([0.01, 0.05, 0.12, 0.23, 0.28, 0.34, 0.36, 0.41, 0.52, 0.58]),
           st.sampled_from([0.01, 0.08, 0.13, 0.22, 0.31, 0.54, 0.64, 0.72, 0.85, 0.87]),
           st.sampled_from([16.73, 21.34, 22.51, 24.1, 26.66, 40.79, 47.55, 110.49, 166.37, 180.43]),
           st.sampled_from([8.25, 8.85, 9.66, 12.13, 17.75, 22.75, 22.87, 25.46, 36.77, 49.87]),
           st.sampled_from([0.31, 0.38, 0.62, 0.71, 1.24, 1.91, 1.99, 2.45, 2.48, 2.55]),
           st.floats(min_value=0.072, max_value=0.084, allow_nan=False),
           st.sampled_from([0.02, 0.08, 0.12, 0.15, 0.2, 0.22, 0.23, 0.29, 0.36, 0.53]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.25, 0.26, 0.41, 0.79, 0.8, 0.85, 0.87, 1.04, 1.05]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.07, 0.25, 0.3, 0.8, 0.99, 1.12, 1.35, 1.46, 1.68, 1.75]),
           st.sampled_from([2.61, 2.81, 2.92, 3.02, 3.17, 3.68, 3.9, 4.05, 4.32, 4.46]),
           st.sampled_from([-5.15, -3.93, -3.76, -3.21, -3.16, -3.12, -2.78, -2.37, -2.1, -2.01]),
           st.sampled_from([-1.27, -1.14, -1.02, -0.83, -0.59, -0.58, -0.49, -0.33, -0.21, -0.11]),
           st.sampled_from([1.09, 1.22, 1.24, 1.96, 1.97, 2.18, 2.19, 2.65, 2.91, 2.96]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.11, 1.21, 1.23, 2.17, 2.19, 2.6, 2.63, 2.9, 2.91, 2.98]),
           st.sampled_from([8.0, 20.0, 24.0, 39.0, 48.0, 57.0, 69.0, 75.0, 77.0, 1695.0]),
           st.sampled_from([1080.0, 1170.0, 3240.0, 6390.0, 6930.0, 7290.0, 9540.0, 11160.0, 19710.0, 51120.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([15.91, 19.83, 22.08, 22.55, 24.29, 39.42, 44.05, 49.63, 57.23, 201.47]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([460.98, 715.89, 806.23, 956.07, 1063.01, 1162.97, 1202.08, 1691.89, 2015.56, 2050.0]),
           st.sampled_from([364.01, 391.51, 569.21, 626.5, 795.5, 912.41, 970.82, 1044.03, 1662.08, 3150.0]),
           st.sampled_from([63.64, 125.81, 145.38, 218.56, 242.16, 244.38, 263.56, 270.72, 308.83, 500.16]),
           st.sampled_from([37.21, 50.88, 52.39, 75.96, 76.26, 88.5, 90.0, 129.99, 131.33, 147.71]),
           st.sampled_from([1.57, 2.51, 2.56, 2.72, 2.87, 2.89, 5.2, 5.22, 6.51, 10.14]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.667, max_value=65.823, exclude_min=True, allow_nan=False),
           st.sampled_from([5.85, 5.95, 6.13, 6.35, 6.39, 6.65, 7.03, 7.04, 7.66, 15.04]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_30(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_30']['n_samples'] += 1
        self.data['tests']['test_30']['samples'].append(x_test)
        self.data['tests']['test_30']['y_expected'].append(y_expected[0])
        self.data['tests']['test_30']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0, 3.0, 4.0, 16.0, 18.0, 28.0, 47.0, 118.0, 125.0, 140.0]),
           st.sampled_from([29.0, 54.0, 60.0, 84.0, 95.0, 96.0, 98.0, 183.0, 188.0, 270.0]),
           st.sampled_from([21.23, 120.26, 221.79, 227.23, 335.02, 343.63, 386.31, 474.38, 1123.09, 1446.29]),
           st.sampled_from([303.14, 305.06, 305.85, 308.71, 309.86, 333.23, 456.23, 566.96, 640.12, 1507.76]),
           st.floats(min_value=78.0, max_value=97.49, allow_nan=False),
           st.sampled_from([80156.0, 125000.0, 142500.0, 160000.0, 240000.0, 285000.0, 320000.0, 510300.0, 793800.0, 882900.0]),
           st.sampled_from([24.09, 30.41, 38.9, 41.53, 42.37, 47.81, 50.73, 52.63, 53.53, 53.69]),
           st.sampled_from([4.83, 5.01, 6.74, 6.92, 7.36, 7.44, 7.99, 8.79, 11.54, 15.7]),
           st.sampled_from([1551.5, 1710.0, 2000.0, 2280.0, 2370.0, 3483.0, 3620.0, 3720.0, 18030.0, 32401.5]),
           st.sampled_from([0.08, 0.1, 0.11, 0.19, 0.23, 0.25, 0.28, 0.3, 0.32, 0.34]),
           st.sampled_from([67.6, 71.2, 79.2, 86.5, 97.0, 104.2, 124.9, 142.9, 298.3, 427.4]),
           st.sampled_from([0.13, 0.17, 0.18, 0.19, 0.2, 0.23, 0.29, 0.36, 0.43, 0.52]),
           st.sampled_from([0.16, 0.21, 0.24, 0.27, 0.28, 0.32, 0.38, 0.39, 0.41, 0.79]),
           st.sampled_from([0.28, 0.35, 0.36, 0.42, 0.48, 0.53, 0.57, 0.66, 0.69, 0.84]),
           st.sampled_from([0.07, 0.08, 0.09, 0.11, 0.14, 0.17, 0.18, 0.27, 0.3, 0.5]),
           st.sampled_from([0.21, 0.24, 0.25, 0.3, 0.33, 0.4, 0.44, 0.54, 0.55, 0.82]),
           st.sampled_from([14.91, 20.38, 20.73, 29.0, 41.65, 118.11, 125.35, 136.77, 207.31, 431.19]),
           st.sampled_from([8.32, 10.98, 12.15, 14.45, 16.73, 19.62, 19.93, 50.37, 54.12, 73.4]),
           st.sampled_from([0.33, 0.38, 0.74, 0.84, 0.88, 1.76, 1.84, 1.85, 1.87, 2.01]),
           st.floats(min_value=0.087, max_value=0.199, exclude_min=True, allow_nan=False),
           st.sampled_from([0.18, 0.21, 0.22, 0.24, 0.25, 0.29, 0.31, 0.34, 0.42, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.19, 0.24, 0.36, 0.39, 0.4, 0.46, 0.54, 0.86, 0.97]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.47, 0.57, 0.73, 0.78, 0.84, 1.08, 1.11, 1.48, 1.55, 2.5]),
           st.sampled_from([1.86, 1.98, 3.56, 4.02, 4.35, 4.36, 5.05, 5.99, 6.3, 10.63]),
           st.sampled_from([-5.18, -4.78, -3.64, -3.38, -3.36, -3.24, -3.19, -2.37, -2.02, -1.63]),
           st.sampled_from([-0.59, -0.5, -0.33, -0.28, -0.23, -0.22, -0.21, -0.17, -0.16, -0.09]),
           st.sampled_from([1.09, 1.1, 1.22, 1.94, 1.95, 2.16, 2.17, 2.18, 2.59, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.22, 1.23, 1.94, 1.95, 1.96, 2.16, 2.17, 2.18, 2.19, 2.2]),
           st.sampled_from([6.0, 12.0, 18.0, 22.0, 26.0, 30.0, 46.0, 133.0, 310.0, 465.0]),
           st.sampled_from([450.0, 630.0, 720.0, 900.0, 990.0, 1260.0, 1800.0, 2700.0, 6750.0, 10080.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.6, 14.7, 21.39, 21.91, 28.07, 29.44, 31.97, 34.35, 56.57, 87.16]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([0.0, 402.49, 538.52, 650.0, 707.11, 851.47, 1060.66, 1281.6, 2022.5, 3936.41]),
           st.sampled_from([141.42, 269.26, 282.84, 284.6, 300.0, 603.74, 636.4, 813.94, 1612.45, 1749.29]),
           st.sampled_from([65.27, 120.21, 164.58, 176.72, 225.0, 291.6, 345.47, 526.56, 623.26, 878.29]),
           st.sampled_from([41.66, 41.98, 52.22, 85.79, 114.82, 124.06, 195.92, 255.81, 348.7, 349.26]),
           st.sampled_from([2.7, 2.85, 4.51, 5.88, 6.95, 8.69, 10.72, 10.99, 11.89, 14.93]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.floats(min_value=65.667, max_value=65.823, exclude_min=True, allow_nan=False),
           st.sampled_from([5.94, 6.42, 6.89, 7.29, 7.39, 7.84, 7.85, 8.07, 14.95, 15.02]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_31(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_31']['n_samples'] += 1
        self.data['tests']['test_31']['samples'].append(x_test)
        self.data['tests']['test_31']['y_expected'].append(y_expected[0])
        self.data['tests']['test_31']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.0, 7.0, 10.0, 11.0, 16.0, 17.0, 23.0, 47.0, 72.0, 125.0]),
           st.sampled_from([29.0, 71.0, 95.0, 96.0, 98.0, 162.0, 183.0, 453.0, 735.0, 9264.0]),
           st.sampled_from([7.7, 115.22, 120.26, 320.75, 454.66, 474.38, 1118.08, 1127.41, 1438.13, 1446.29]),
           st.sampled_from([242.59, 287.42, 292.22, 305.85, 544.91, 546.0, 624.0, 1165.76, 1494.86, 1530.1]),
           st.floats(min_value=97.51, max_value=114.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=85004.8, max_value=88677.99, allow_nan=False),
           st.sampled_from([23.11, 23.92, 29.28, 40.42, 42.4, 43.9, 51.39, 52.11, 53.53, 55.83]),
           st.sampled_from([5.46, 6.24, 6.74, 7.08, 7.36, 10.48, 11.38, 11.54, 13.54, 16.85]),
           st.sampled_from([1461.0, 1570.0, 1800.0, 1850.0, 3660.0, 4748.0, 6998.0, 9294.0, 18030.0, 22260.0]),
           st.sampled_from([0.11, 0.18, 0.19, 0.22, 0.23, 0.24, 0.26, 0.27, 0.28, 0.32]),
           st.floats(min_value=125.52, max_value=146.64, allow_nan=False),
           st.sampled_from([0.14, 0.15, 0.19, 0.2, 0.24, 0.26, 0.31, 0.33, 0.36, 0.52]),
           st.sampled_from([0.21, 0.32, 0.33, 0.34, 0.35, 0.37, 0.38, 0.41, 0.53, 0.79]),
           st.sampled_from([0.23, 0.24, 0.3, 0.31, 0.34, 0.36, 0.63, 0.69, 0.84, 0.96]),
           st.sampled_from([0.07, 0.09, 0.14, 0.15, 0.2, 0.23, 0.27, 0.32, 0.35, 0.5]),
           st.sampled_from([0.15, 0.2, 0.24, 0.25, 0.34, 0.35, 0.4, 0.44, 0.64, 0.82]),
           st.sampled_from([11.66, 17.07, 20.73, 24.37, 29.0, 41.65, 64.34, 136.77, 207.31, 431.19]),
           st.sampled_from([7.64, 8.32, 9.14, 12.15, 14.45, 21.67, 22.56, 25.06, 49.55, 50.99]),
           st.sampled_from([0.3, 0.32, 0.38, 0.47, 0.65, 0.74, 0.82, 1.76, 1.89, 1.91]),
           st.sampled_from([0.1, 0.15, 0.17, 0.24, 0.25, 0.27, 0.28, 0.29, 0.31, 0.32]),
           st.sampled_from([0.19, 0.2, 0.21, 0.22, 0.23, 0.26, 0.34, 0.4, 0.42, 0.43]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.24, 0.4, 0.45, 0.46, 0.51, 0.75, 0.87, 0.88, 0.9, 0.92]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.21, 0.25, 0.56, 0.74, 0.84, 1.44, 1.47, 1.55, 1.61, 1.98]),
           st.sampled_from([1.86, 2.34, 2.67, 3.2, 3.21, 3.34, 3.56, 3.84, 4.06, 5.05]),
           st.sampled_from([-3.89, -3.54, -3.53, -3.5, -3.19, -2.72, -2.24, -1.63, -1.39, -0.65]),
           st.sampled_from([-1.02, -0.75, -0.63, -0.5, -0.33, -0.32, -0.28, -0.23, -0.21, -0.12]),
           st.sampled_from([1.09, 1.22, 1.94, 1.95, 2.16, 2.17, 2.18, 2.2, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.09, 1.1, 1.94, 1.96, 2.16, 2.18, 2.19, 2.2, 2.91, 2.92]),
           st.sampled_from([13.0, 18.0, 21.0, 22.0, 29.0, 30.0, 32.0, 34.0, 46.0, 78.0]),
           st.sampled_from([450.0, 720.0, 1170.0, 1620.0, 2070.0, 3060.0, 8730.0, 10080.0, 16110.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([10.81, 31.97, 38.8, 40.67, 46.19, 61.69, 78.14, 80.59, 87.16, 138.68]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([402.49, 403.89, 650.0, 685.42, 1060.66, 1400.0, 2101.07, 3471.31, 3936.41, 4724.79]),
           st.sampled_from([90.0, 212.13, 223.61, 250.0, 603.74, 636.4, 761.58, 1612.45, 1659.52, 2189.79]),
           st.sampled_from([110.77, 164.58, 170.58, 204.18, 236.27, 268.75, 291.6, 368.74, 1209.38, 1229.26]),
           st.sampled_from([0.0, 41.98, 44.75, 47.73, 61.45, 117.4, 124.72, 135.46, 145.92, 220.85]),
           st.sampled_from([0.0, 2.64, 2.7, 6.11, 6.44, 6.63, 8.69, 10.99, 13.05, 13.33]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.sampled_from([36.14, 36.31, 36.49, 65.58, 65.66, 65.67, 65.78, 65.82, 65.84, 65.9]),
           st.sampled_from([5.94, 6.16, 6.29, 6.46, 6.96, 7.18, 7.32, 7.36, 7.95, 8.07]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_32(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_32']['n_samples'] += 1
        self.data['tests']['test_32']['samples'].append(x_test)
        self.data['tests']['test_32']['y_expected'].append(y_expected[0])
        self.data['tests']['test_32']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([44.0, 52.0, 62.0, 122.0, 123.0, 131.0, 140.0, 169.0, 180.0, 190.0]),
           st.sampled_from([13.0, 62.0, 65.0, 153.0, 232.0, 260.0, 367.0, 725.0, 743.0, 3234.0]),
           st.sampled_from([13.3, 30.26, 35.05, 124.9, 153.39, 537.93, 902.43, 971.94, 1020.29, 1702.82]),
           st.sampled_from([437.2, 551.22, 614.24, 642.79, 836.77, 1090.43, 1234.18, 1245.62, 1255.31, 2229.45]),
           st.floats(min_value=97.51, max_value=114.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=88678.01, max_value=14333942.4, exclude_min=True, allow_nan=False),
           st.sampled_from([27.82, 30.26, 33.05, 35.42, 41.09, 42.08, 43.95, 47.75, 50.4, 51.04]),
           st.sampled_from([4.37, 4.58, 4.79, 4.84, 5.08, 8.19, 9.02, 10.03, 11.76, 15.1]),
           st.sampled_from([1159.0, 1270.0, 1370.5, 1560.0, 1790.0, 1880.0, 2310.0, 2628.5, 3890.5, 5600.0]),
           st.sampled_from([0.08, 0.14, 0.15, 0.16, 0.38, 0.47, 0.54, 0.58, 0.6, 0.67]),
           st.floats(min_value=125.52, max_value=146.64, allow_nan=False),
           st.sampled_from([0.02, 0.07, 0.13, 0.24, 0.28, 0.36, 0.46, 0.49, 0.52, 0.53]),
           st.sampled_from([0.19, 0.21, 0.22, 0.31, 0.34, 0.43, 0.44, 0.48, 0.57, 0.75]),
           st.sampled_from([0.05, 0.12, 0.18, 0.37, 0.41, 0.42, 0.78, 0.82, 0.96, 1.11]),
           st.sampled_from([0.07, 0.13, 0.14, 0.2, 0.29, 0.35, 0.38, 0.53, 0.58, 0.61]),
           st.sampled_from([0.01, 0.05, 0.07, 0.11, 0.21, 0.38, 0.39, 0.4, 0.46, 0.54]),
           st.sampled_from([30.48, 31.83, 39.17, 44.84, 51.76, 68.6, 74.36, 76.63, 218.44, 1559.95]),
           st.sampled_from([10.13, 10.66, 11.03, 11.21, 16.3, 19.08, 19.5, 21.63, 24.16, 26.88]),
           st.sampled_from([0.22, 0.62, 0.66, 0.72, 1.86, 2.1, 2.14, 2.23, 2.24, 2.26]),
           st.sampled_from([0.08, 0.1, 0.18, 0.23, 0.32, 0.36, 0.39, 0.42, 0.5, 0.65]),
           st.sampled_from([0.06, 0.07, 0.15, 0.16, 0.36, 0.41, 0.56, 0.57, 0.63, 0.71]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.18, 0.29, 0.4, 0.43, 0.53, 0.55, 0.75, 1.02, 1.11, 1.19]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-1.16, -0.9, -0.21, -0.02, 0.15, 0.66, 1.4, 1.61, 2.36, 2.74]),
           st.sampled_from([1.98, 1.99, 2.2, 2.39, 3.55, 3.74, 8.0, 8.34, 10.09, 10.46]),
           st.sampled_from([-7.21, -3.45, -3.4, -3.34, -3.27, -2.63, -1.57, -1.05, -0.97, -0.94]),
           st.sampled_from([-1.14, -0.98, -0.89, -0.64, -0.59, -0.47, -0.44, -0.36, -0.34, -0.27]),
           st.sampled_from([1.21, 1.95, 1.97, 1.99, 2.01, 2.17, 2.23, 2.64, 2.65, 2.92]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.21, 1.24, 1.98, 2.01, 2.18, 2.21, 2.24, 2.6, 2.61, 2.98]),
           st.sampled_from([5.0, 14.0, 17.0, 26.0, 62.0, 63.0, 64.0, 98.0, 141.0, 164.0]),
           st.sampled_from([1530.0, 2610.0, 2700.0, 3150.0, 3510.0, 5130.0, 8460.0, 12330.0, 33390.0, 34290.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([14.17, 16.5, 22.57, 32.0, 33.37, 34.27, 42.8, 53.65, 57.1, 60.02]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([304.14, 447.21, 850.0, 1133.58, 1171.54, 1250.0, 1489.77, 1874.17, 3010.4, 3874.68]),
           st.sampled_from([159.1, 353.55, 563.75, 630.0, 632.46, 649.0, 710.63, 969.33, 1872.78, 8789.57]),
           st.sampled_from([70.71, 84.85, 120.0, 157.08, 164.99, 165.71, 170.0, 190.91, 195.56, 288.58]),
           st.sampled_from([35.79, 51.96, 55.24, 73.14, 73.6, 81.55, 87.22, 105.09, 386.5, 387.6]),
           st.sampled_from([1.23, 3.13, 3.17, 3.77, 5.31, 5.35, 10.2, 10.7, 10.73, 21.06]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.sampled_from([35.98, 36.06, 36.67, 36.8, 65.53, 65.67, 65.79, 65.82, 66.15, 66.38]),
           st.sampled_from([6.03, 6.4, 6.74, 7.64, 7.82, 7.95, 14.6, 14.71, 14.87, 14.98]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_33(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_33']['n_samples'] += 1
        self.data['tests']['test_33']['samples'].append(x_test)
        self.data['tests']['test_33']['y_expected'].append(y_expected[0])
        self.data['tests']['test_33']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([42.0, 67.0, 98.0, 109.0, 120.0, 133.0, 146.0, 147.0, 185.0, 197.0]),
           st.sampled_from([16.0, 54.0, 78.0, 132.0, 153.0, 157.0, 164.0, 191.0, 6099.0, 32389.0]),
           st.sampled_from([28.95, 41.19, 47.2, 64.45, 408.25, 475.5, 491.81, 844.54, 1208.61, 1628.99]),
           st.sampled_from([52.55, 331.27, 555.24, 580.92, 1063.34, 1155.59, 1248.93, 1260.37, 1557.52, 1683.72]),
           st.floats(min_value=97.51, max_value=114.0, exclude_min=True, allow_nan=False),
           st.sampled_from([105468.0, 147500.0, 178200.0, 324843.0, 331875.0, 630000.0, 672500.0, 755000.0, 1406250.0, 3912187.0]),
           st.sampled_from([27.82, 27.96, 28.72, 29.49, 29.51, 38.67, 56.39, 66.69, 68.26, 73.32]),
           st.sampled_from([1.7, 5.1, 5.45, 6.17, 6.54, 8.03, 8.22, 9.11, 9.12, 18.26]),
           st.sampled_from([741.0, 778.0, 937.0, 1701.5, 1970.0, 2210.0, 3010.0, 4290.0, 4560.0, 10852.5]),
           st.sampled_from([0.03, 0.07, 0.13, 0.14, 0.26, 0.31, 0.36, 0.37, 0.43, 0.6]),
           st.floats(min_value=146.67, max_value=297.67, exclude_min=True, allow_nan=False),
           st.sampled_from([0.07, 0.14, 0.19, 0.2, 0.22, 0.36, 0.41, 0.47, 0.49, 0.54]),
           st.sampled_from([0.14, 0.17, 0.3, 0.33, 0.34, 0.41, 0.49, 0.61, 0.65, 0.83]),
           st.sampled_from([0.1, 0.14, 0.16, 0.24, 0.29, 0.51, 0.63, 0.66, 0.67, 1.23]),
           st.floats(min_value=0.066, max_value=0.079, allow_nan=False),
           st.sampled_from([0.07, 0.39, 0.53, 0.54, 0.57, 0.6, 0.63, 0.64, 0.85, 0.86]),
           st.sampled_from([10.47, 18.52, 20.07, 26.66, 28.41, 30.31, 32.94, 35.65, 49.55, 51.89]),
           st.sampled_from([10.3, 11.03, 11.71, 11.77, 20.15, 20.31, 21.59, 30.38, 31.53, 40.32]),
           st.sampled_from([0.51, 0.6, 0.61, 0.64, 0.78, 1.15, 2.01, 2.18, 2.23, 2.43]),
           st.sampled_from([0.02, 0.03, 0.1, 0.17, 0.22, 0.29, 0.33, 0.38, 0.54, 0.65]),
           st.sampled_from([0.03, 0.17, 0.22, 0.31, 0.35, 0.48, 0.5, 0.57, 0.6, 0.73]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.3, 0.59, 0.65, 0.8, 1.0, 1.03, 1.05, 1.08, 1.09, 1.27]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([-0.37, -0.33, -0.25, -0.22, 0.06, 0.56, 0.8, 1.47, 1.49, 3.04]),
           st.sampled_from([1.44, 1.88, 2.61, 3.16, 3.82, 4.08, 4.14, 6.28, 7.12, 9.71]),
           st.sampled_from([-6.67, -6.0, -3.0, -2.81, -2.8, -2.7, -1.76, -1.58, -1.24, -0.7]),
           st.sampled_from([-1.22, -1.14, -1.0, -0.94, -0.85, -0.74, -0.71, -0.6, -0.25, -0.11]),
           st.sampled_from([1.1, 1.97, 1.98, 1.99, 2.22, 2.24, 2.6, 2.62, 2.63, 2.95]),
           st.sampled_from([0.0, 0.01, 0.86, 0.87]),
           st.sampled_from([1.09, 1.11, 1.94, 1.99, 2.21, 2.22, 2.62, 2.65, 2.94, 2.96]),
           st.sampled_from([15.0, 63.0, 80.0, 83.0, 98.0, 160.0, 302.0, 346.0, 367.0, 882.0]),
           st.sampled_from([450.0, 540.0, 1260.0, 1530.0, 1980.0, 3690.0, 3780.0, 5580.0, 9810.0, 20160.0]),
           st.sampled_from([0.0, 0.01, 0.02]),
           st.sampled_from([11.87, 14.32, 17.4, 19.6, 20.82, 33.57, 51.26, 106.12, 134.68, 202.33]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([604.15, 886.4, 1000.0, 1051.19, 1140.18, 1202.08, 1303.84, 2222.84, 2800.0, 4125.0]),
           st.sampled_from([180.0, 270.42, 284.6, 583.1, 790.57, 950.0, 1097.72, 1453.44, 2362.39, 3150.0]),
           st.sampled_from([123.74, 125.0, 136.62, 165.59, 184.63, 272.69, 318.45, 334.64, 423.38, 519.94]),
           st.sampled_from([52.36, 57.3, 77.77, 80.98, 94.93, 96.63, 109.35, 174.6, 181.94, 380.86]),
           st.sampled_from([1.9, 2.49, 3.37, 3.58, 3.64, 4.9, 9.14, 11.41, 11.49, 76.63]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.sampled_from([36.12, 36.14, 36.44, 36.8, 36.85, 65.48, 65.65, 65.78, 66.08, 66.26]),
           st.sampled_from([6.1, 6.25, 6.36, 6.61, 6.71, 6.74, 7.08, 7.33, 14.89, 15.03]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_34(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_34']['n_samples'] += 1
        self.data['tests']['test_34']['samples'].append(x_test)
        self.data['tests']['test_34']['y_expected'].append(y_expected[0])
        self.data['tests']['test_34']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([2.0, 3.0, 6.0, 7.0, 9.0, 13.0, 16.0, 18.0, 28.0, 140.0]),
           st.sampled_from([50.0, 57.0, 63.0, 67.0, 107.0, 115.0, 183.0, 188.0, 323.0, 2274.0]),
           st.sampled_from([21.23, 226.81, 227.23, 273.61, 335.02, 361.68, 474.38, 964.23, 1020.91, 1449.85]),
           st.sampled_from([242.59, 292.22, 305.85, 309.86, 333.23, 338.47, 469.39, 544.91, 1337.49, 1494.86]),
           st.floats(min_value=97.51, max_value=114.0, exclude_min=True, allow_nan=False),
           st.sampled_from([129600.0, 234900.0, 240000.0, 497812.0, 769500.0, 882900.0, 1132500.0, 1312200.0, 2948400.0, 5953500.0]),
           st.sampled_from([24.09, 38.65, 40.42, 40.46, 42.4, 46.29, 49.35, 53.53, 60.03, 70.65]),
           st.sampled_from([6.24, 6.74, 7.36, 8.79, 11.38, 11.54, 13.59, 13.97, 14.9, 16.85]),
           st.sampled_from([1318.0, 1400.0, 1551.5, 2280.0, 3483.0, 3620.0, 3720.0, 4187.0, 21569.0, 29780.0]),
           st.sampled_from([0.09, 0.11, 0.15, 0.19, 0.23, 0.24, 0.25, 0.26, 0.3, 0.32]),
           st.floats(min_value=146.67, max_value=297.67, exclude_min=True, allow_nan=False),
           st.sampled_from([0.13, 0.15, 0.17, 0.19, 0.2, 0.22, 0.23, 0.28, 0.33, 0.34]),
           st.sampled_from([0.14, 0.22, 0.28, 0.29, 0.34, 0.35, 0.39, 0.42, 0.51, 0.67]),
           st.sampled_from([0.24, 0.28, 0.31, 0.35, 0.48, 0.49, 0.5, 0.64, 0.65, 0.66]),
           st.floats(min_value=0.082, max_value=0.195, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2, 0.24, 0.3, 0.33, 0.38, 0.4, 0.42, 0.54, 0.6, 0.64]),
           st.sampled_from([14.91, 17.07, 18.19, 20.38, 29.4, 55.25, 56.02, 99.59, 120.22, 149.87]),
           st.sampled_from([7.64, 8.09, 8.32, 13.31, 16.73, 19.62, 24.47, 43.96, 50.37, 54.12]),
           st.sampled_from([0.15, 0.32, 0.34, 0.65, 1.75, 1.84, 1.89, 1.91, 1.97, 2.14]),
           st.sampled_from([0.1, 0.15, 0.16, 0.19, 0.21, 0.26, 0.28, 0.32, 0.33, 0.34]),
           st.sampled_from([0.18, 0.2, 0.21, 0.22, 0.23, 0.31, 0.34, 0.43, 0.49, 0.52]),
           st.sampled_from([47.66, 55.85, 67.87, 69.09, 75.26, 85.22, 87.65, 123.47, 126.08]),
           st.sampled_from([0.0]),
           st.sampled_from([0.19, 0.36, 0.44, 0.46, 0.5, 0.51, 0.81, 0.86, 0.87, 0.9]),
           st.sampled_from([132.78, 204.34, 221.97, 239.69, 351.67, 421.21, 422.12, 2025.42, 2036.8]),
           st.sampled_from([-0.71, -0.53, -0.01, 0.18, 0.87, 0.97, 1.01, 1.83]),
           st.sampled_from([2.96, 3.01, 3.78, 3.83, 4.66, 5.07, 9.24, 12.06, 14.78]),
           st.sampled_from([0.26, 0.29, 0.56, 0.69, 0.73, 0.85, 1.01, 1.11, 1.55, 1.98]),
           st.sampled_from([1.86, 2.64, 3.36, 3.52, 4.72, 4.79, 6.3, 6.78, 6.81, 8.68]),
           st.sampled_from([-7.76, -5.55, -3.87, -3.71, -3.58, -3.5, -3.07, -2.72, -2.59, -1.92]),
           st.sampled_from([-0.63, -0.36, -0.32, -0.3, -0.22, -0.18, -0.14, -0.12, -0.09, -0.07]),
           st.sampled_from([1.09, 1.23, 1.96, 2.16, 2.18, 2.19, 2.2, 2.59, 2.91, 2.92]),
           st.sampled_from([0.0]),
           st.sampled_from([1.22, 1.23, 1.94, 1.96, 2.16, 2.18, 2.2, 2.59, 2.91, 2.92]),
           st.sampled_from([12.0, 13.0, 17.0, 19.0, 36.0, 96.0, 133.0, 150.0, 183.0, 202.0]),
           st.sampled_from([900.0, 990.0, 1080.0, 1170.0, 2160.0, 2700.0, 6750.0, 8100.0, 10350.0, 24030.0]),
           st.sampled_from([0.0, 0.01]),
           st.sampled_from([9.71, 10.6, 13.07, 16.0, 18.24, 21.39, 23.4, 31.97, 46.19, 78.14]),
           st.sampled_from([64.0, 78.0, 82.0, 85.0, 89.0, 99.0, 102.0, 133.0, 143.0]),
           st.sampled_from([39.0, 50.0, 55.0, 63.0, 67.0, 69.0, 73.0, 85.0, 86.0]),
           st.sampled_from([471.7, 538.52, 608.28, 685.42, 960.47, 2850.0, 3471.31, 4724.79, 5650.88, 6041.52]),
           st.sampled_from([0.0, 90.0, 250.0, 269.26, 282.84, 403.11, 721.11, 1000.0, 1170.0, 1659.52]),
           st.sampled_from([0.0, 135.0, 170.58, 176.72, 204.18, 236.27, 256.77, 526.56, 623.26, 763.16]),
           st.sampled_from([41.98, 47.73, 60.55, 61.45, 65.25, 88.6, 124.06, 220.85, 348.7, 349.26]),
           st.sampled_from([1.37, 1.67, 1.94, 2.7, 4.51, 5.88, 6.32, 6.63, 10.99, 11.32]),
           st.sampled_from([0.0, 1.0]),
           st.floats(min_value=23348.297, max_value=29704.329, exclude_min=True, allow_nan=False),
           st.sampled_from([65.58, 65.66, 65.77, 65.78, 65.79, 65.84, 65.9, 65.93, 65.97, 66.25]),
           st.sampled_from([5.94, 6.29, 6.54, 7.24, 7.28, 7.29, 7.58, 7.65, 7.95, 14.95]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_35(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_35']['n_samples'] += 1
        self.data['tests']['test_35']['samples'].append(x_test)
        self.data['tests']['test_35']['y_expected'].append(y_expected[0])
        self.data['tests']['test_35']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
