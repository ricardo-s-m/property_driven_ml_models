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
    request.cls.data['n_test'] = 22
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_breast_cancer_bva_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestBreastCancerProperty:

    @given(st.sampled_from([10.05, 11.46, 11.67, 12.03, 12.16, 12.36, 12.87, 13.03, 13.05, 14.5]),
           st.sampled_from([13.78, 15.34, 15.65, 16.32, 16.34, 17.68, 18.1, 20.54, 21.6, 25.22]),
           st.sampled_from([61.05, 64.73, 65.12, 65.31, 70.21, 73.87, 76.85, 82.53, 86.49, 89.78]),
           st.sampled_from([300.2, 307.3, 311.9, 396.0, 416.2, 462.0, 464.1, 469.1, 485.8, 668.7]),
           st.sampled_from([0.06883, 0.07274, 0.08276, 0.08284, 0.09434, 0.09687, 0.09933, 0.0995, 0.1024, 0.1045]),
           st.sampled_from([0.03872, 0.04043, 0.06679, 0.06718, 0.09486, 0.09697, 0.1117, 0.1147, 0.1155, 0.1296]),
           st.sampled_from([0.01857, 0.01923, 0.02224, 0.02337, 0.03581, 0.03809, 0.04505, 0.07741, 0.09263, 0.1191]),
           st.sampled_from([0.01963, 0.02036, 0.02037, 0.02471, 0.02534, 0.02645, 0.0316, 0.04908, 0.06211, 0.07857]),
           st.sampled_from([0.1203, 0.122, 0.1274, 0.1621, 0.1668, 0.1687, 0.1806, 0.1842, 0.1943, 0.1944]),
           st.sampled_from([0.05268, 0.05731, 0.06057, 0.06087, 0.06147, 0.06183, 0.06347, 0.06457, 0.06899, 0.06963]),
           st.sampled_from([0.1458, 0.1639, 0.1665, 0.1767, 0.1814, 0.1816, 0.2023, 0.2271, 0.2387, 0.3721]),
           st.sampled_from([0.4833, 0.6793, 0.6931, 0.7656, 0.7927, 0.8944, 1.182, 1.39, 1.652, 2.635]),
           st.sampled_from([0.8439, 1.199, 1.348, 1.553, 1.83, 2.087, 2.183, 2.591, 2.877, 3.33]),
           st.floats(min_value=32.2443, max_value=38.6048, allow_nan=False),
           st.floats(min_value=0.002977, max_value=0.003293, allow_nan=False),
           st.sampled_from([0.0104, 0.01082, 0.01153, 0.01179, 0.0118, 0.02172, 0.02305, 0.02736, 0.02899, 0.04112]),
           st.sampled_from([0.005325, 0.005832, 0.007276, 0.007816, 0.01451, 0.02443, 0.03125, 0.0398, 0.05189, 0.05738]),
           st.sampled_from([0.0, 0.005298, 0.005383, 0.005484, 0.00637, 0.007369, 0.00762, 0.01007, 0.01161, 0.02188]),
           st.sampled_from([0.01344, 0.01347, 0.01422, 0.02086, 0.02108, 0.02134, 0.02921, 0.03004, 0.03102, 0.03675]),
           st.sampled_from([0.001435, 0.002425, 0.002472, 0.00248, 0.002534, 0.002607, 0.002744, 0.002778, 0.002848, 0.003399]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=24.608, max_value=27.754, allow_nan=False),
           st.sampled_from([75.19, 78.44, 84.7, 86.67, 90.81, 94.52, 98.4, 100.2, 104.5, 115.9]),
           st.sampled_from([270.0, 342.9, 390.4, 483.1, 503.0, 551.3, 623.7, 630.5, 687.6, 762.6]),
           st.sampled_from([0.09439, 0.106, 0.1072, 0.1157, 0.1185, 0.1249, 0.1292, 0.1376, 0.1733, 0.185]),
           st.sampled_from([0.08614, 0.1, 0.12, 0.1379, 0.1644, 0.188, 0.2156, 0.2264, 0.2499, 0.2793]),
           st.sampled_from([0.03938, 0.09996, 0.102, 0.1067, 0.122, 0.13, 0.1632, 0.2123, 0.2962, 0.4004]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2171, 0.2372, 0.2475, 0.2488, 0.2615, 0.2668, 0.2779, 0.2787, 0.2806, 0.323]),
           st.sampled_from([0.06289, 0.06688, 0.06769, 0.07062, 0.07319, 0.07675, 0.08052, 0.08273, 0.09879, 0.1084]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.42, 14.48, 15.46, 16.02, 16.27, 17.27, 18.46, 18.82, 21.56, 22.27]),
           st.sampled_from([10.38, 15.51, 17.12, 17.57, 19.32, 20.58, 20.67, 22.07, 22.47, 24.04]),
           st.sampled_from([75.0, 85.42, 93.77, 96.73, 97.4, 98.0, 102.5, 124.8, 133.7, 142.0]),
           st.sampled_from([475.9, 559.2, 563.0, 815.8, 857.6, 982.0, 1068.0, 1155.0, 1203.0, 1214.0]),
           st.sampled_from([0.08402, 0.09215, 0.0943, 0.09463, 0.09812, 0.09898, 0.09997, 0.1082, 0.1215, 0.1425]),
           st.sampled_from([0.05761, 0.1029, 0.1159, 0.1206, 0.1314, 0.1489, 0.1555, 0.2768, 0.2776, 0.2832]),
           st.sampled_from([0.03299, 0.09769, 0.1147, 0.1367, 0.1417, 0.1525, 0.1793, 0.2133, 0.2188, 0.3189]),
           st.sampled_from([0.03334, 0.0539, 0.06018, 0.06254, 0.07507, 0.07981, 0.1043, 0.1259, 0.1377, 0.1845]),
           st.sampled_from([0.1538, 0.1848, 0.1998, 0.2091, 0.2092, 0.2108, 0.2123, 0.2149, 0.2252, 0.2521]),
           st.sampled_from([0.05407, 0.0551, 0.05866, 0.05941, 0.06049, 0.0614, 0.06218, 0.06697, 0.0687, 0.07292]),
           st.sampled_from([0.231, 0.2419, 0.2895, 0.3345, 0.3971, 0.439, 0.6361, 0.9291, 1.004, 1.176]),
           st.sampled_from([0.7452, 0.828, 0.9622, 0.976, 1.033, 1.069, 1.321, 1.452, 2.11, 2.284]),
           st.sampled_from([1.719, 2.362, 2.629, 2.844, 3.705, 3.833, 4.106, 4.174, 4.321, 5.54]),
           st.floats(min_value=32.2443, max_value=38.6048, allow_nan=False),
           st.floats(min_value=0.002977, max_value=0.003293, allow_nan=False),
           st.sampled_from([0.01174, 0.02263, 0.02785, 0.02891, 0.03374, 0.0496, 0.05121, 0.0605, 0.08297, 0.09806]),
           st.sampled_from([0.01311, 0.03112, 0.03185, 0.03576, 0.03582, 0.03988, 0.04665, 0.05081, 0.068, 0.07926]),
           st.sampled_from([0.009231, 0.01186, 0.01271, 0.01272, 0.0132, 0.01424, 0.01499, 0.01806, 0.02215, 0.02765]),
           st.sampled_from([0.01177, 0.01226, 0.01852, 0.019, 0.01925, 0.0225, 0.02337, 0.02816, 0.05628, 0.05963]),
           st.sampled_from([0.002205, 0.002686, 0.002695, 0.002747, 0.00304, 0.003727, 0.00374, 0.003892, 0.004452, 0.007646]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=27.757, max_value=32.113, exclude_min=True, allow_nan=False),
           st.sampled_from([101.7, 103.4, 104.3, 107.5, 117.7, 119.4, 125.4, 146.6, 158.3, 160.0]),
           st.sampled_from([806.9, 1175.0, 1210.0, 1269.0, 1549.0, 1606.0, 1646.0, 1671.0, 1972.0, 2022.0]),
           st.sampled_from([0.1207, 0.1275, 0.1381, 0.1497, 0.1498, 0.1514, 0.1634, 0.1777, 0.1862, 0.2184]),
           st.sampled_from([0.1551, 0.2666, 0.2733, 0.3463, 0.3725, 0.3903, 0.4186, 0.4648, 0.4827, 0.6247]),
           st.sampled_from([0.2623, 0.3378, 0.3794, 0.5036, 0.5186, 0.5344, 0.5539, 0.6451, 0.6943, 0.7242]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.1978, 0.2834, 0.2968, 0.2994, 0.3032, 0.3698, 0.39, 0.3985, 0.4677, 0.4882]),
           st.sampled_from([0.07421, 0.07568, 0.07602, 0.07849, 0.08911, 0.09223, 0.1026, 0.1189, 0.1233, 0.1243]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.95, 9.295, 10.48, 10.57, 10.9, 11.69, 12.04, 13.56, 14.95, 16.17]),
           st.sampled_from([12.84, 13.1, 13.27, 14.11, 15.79, 15.91, 16.49, 17.27, 21.7, 22.11]),
           st.sampled_from([43.79, 66.52, 74.68, 76.31, 81.92, 83.18, 84.18, 84.74, 85.09, 92.51]),
           st.sampled_from([372.7, 386.0, 399.8, 453.1, 512.0, 537.9, 542.9, 571.8, 641.2, 668.6]),
           st.sampled_from([0.06828, 0.07963, 0.08311, 0.08386, 0.08582, 0.08597, 0.08924, 0.09495, 0.09524, 0.09816]),
           st.sampled_from([0.03212, 0.05205, 0.05794, 0.0645, 0.07624, 0.09661, 0.1114, 0.1125, 0.1296, 0.1334]),
           st.sampled_from([0.005025, 0.005067, 0.006643, 0.01288, 0.01797, 0.04105, 0.04462, 0.05263, 0.06335, 0.07721]),
           st.sampled_from([0.002941, 0.007246, 0.008488, 0.0129, 0.01737, 0.0239, 0.02416, 0.03264, 0.04812, 0.04951]),
           st.sampled_from([0.1411, 0.1467, 0.1508, 0.1607, 0.1617, 0.1703, 0.1723, 0.197, 0.2013, 0.2086]),
           st.sampled_from([0.05952, 0.06083, 0.06312, 0.06409, 0.06491, 0.06552, 0.0657, 0.06782, 0.07098, 0.09575]),
           st.sampled_from([0.1408, 0.1783, 0.2113, 0.2345, 0.2406, 0.2713, 0.2784, 0.2818, 0.3278, 0.4347]),
           st.sampled_from([0.4833, 0.6931, 0.7786, 0.8163, 0.9078, 0.9961, 1.166, 1.312, 1.428, 1.597]),
           st.sampled_from([1.267, 1.373, 1.525, 1.565, 1.721, 2.308, 2.344, 2.394, 3.267, 4.021]),
           st.floats(min_value=32.2443, max_value=38.6048, allow_nan=False),
           st.floats(min_value=0.003296, max_value=0.008862, exclude_min=True, allow_nan=False),
           st.sampled_from([0.008432, 0.009514, 0.01017, 0.01233, 0.02005, 0.02845, 0.03026, 0.03378, 0.04192, 0.06669]),
           st.sampled_from([0.007004, 0.007975, 0.009127, 0.01461, 0.01465, 0.01683, 0.01855, 0.02589, 0.02662, 0.03029]),
           st.sampled_from([0.007016, 0.007039, 0.007956, 0.008231, 0.009073, 0.009215, 0.009233, 0.01112, 0.01152, 0.02919]),
           st.sampled_from([0.01449, 0.01637, 0.01848, 0.02009, 0.02027, 0.021, 0.02216, 0.02349, 0.02354, 0.03356]),
           st.sampled_from([0.001344, 0.001381, 0.002583, 0.002768, 0.003002, 0.003071, 0.004638, 0.006884, 0.007555, 0.007877]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=29.019, max_value=33.268, allow_nan=False),
           st.sampled_from([65.5, 72.22, 79.57, 80.78, 83.85, 84.58, 86.67, 96.53, 100.2, 101.2]),
           st.sampled_from([270.0, 385.2, 395.4, 435.9, 470.0, 495.2, 643.8, 719.8, 819.1, 1210.0]),
           st.sampled_from([0.08567, 0.08864, 0.1016, 0.1092, 0.1097, 0.1176, 0.1293, 0.15, 0.178, 0.2006]),
           st.sampled_from([0.09726, 0.1381, 0.1575, 0.1644, 0.1726, 0.1887, 0.1892, 0.2187, 0.2506, 0.251]),
           st.sampled_from([0.03582, 0.05233, 0.05307, 0.05524, 0.06648, 0.1277, 0.1364, 0.231, 0.2654, 0.4504]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2438, 0.2606, 0.2758, 0.2775, 0.2804, 0.2878, 0.3006, 0.3163, 0.322, 0.3455]),
           st.sampled_from([0.05521, 0.07676, 0.07875, 0.08278, 0.0849, 0.08701, 0.08797, 0.09938, 0.1043, 0.1082]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.8, 15.06, 17.35, 18.82, 19.07, 19.19, 19.68, 20.58, 20.92, 25.22]),
           st.sampled_from([17.05, 17.52, 17.89, 18.57, 20.58, 21.81, 22.49, 25.0, 25.09, 26.57]),
           st.sampled_from([88.4, 94.74, 96.42, 102.7, 102.9, 103.6, 117.5, 120.3, 122.1, 142.0]),
           st.sampled_from([642.5, 744.7, 826.8, 948.0, 991.7, 1088.0, 1308.0, 1319.0, 1482.0, 1682.0]),
           st.sampled_from([0.08772, 0.09078, 0.1008, 0.1018, 0.103, 0.1036, 0.106, 0.1084, 0.1162, 0.1398]),
           st.sampled_from([0.1023, 0.1143, 0.1157, 0.1223, 0.1298, 0.1314, 0.1348, 0.1389, 0.1497, 0.2363]),
           st.sampled_from([0.06593, 0.1569, 0.1626, 0.164, 0.1684, 0.1692, 0.1891, 0.2065, 0.2128, 0.3754]),
           st.sampled_from([0.02031, 0.02704, 0.03334, 0.05778, 0.06597, 0.06759, 0.08744, 0.1389, 0.141, 0.1878]),
           st.sampled_from([0.1598, 0.1761, 0.1814, 0.1847, 0.1928, 0.1966, 0.1998, 0.2041, 0.2085, 0.2183]),
           st.sampled_from([0.05096, 0.05176, 0.05395, 0.05484, 0.05581, 0.05592, 0.06251, 0.06544, 0.07469, 0.07799]),
           st.sampled_from([0.1938, 0.3331, 0.3704, 0.4681, 0.6003, 0.6107, 0.6191, 0.6361, 0.6534, 1.0]),
           st.sampled_from([0.8225, 0.828, 0.9489, 1.023, 1.127, 1.178, 1.189, 1.214, 1.377, 1.398]),
           st.sampled_from([1.895, 1.897, 2.183, 2.844, 2.861, 5.383, 5.801, 7.247, 7.337, 10.12]),
           st.floats(min_value=32.2443, max_value=38.6048, allow_nan=False),
           st.floats(min_value=0.003296, max_value=0.008862, exclude_min=True, allow_nan=False),
           st.sampled_from([0.01202, 0.01488, 0.018, 0.02499, 0.02772, 0.02863, 0.03345, 0.03633, 0.07217, 0.08297]),
           st.sampled_from([0.01272, 0.01998, 0.02117, 0.02749, 0.03109, 0.03342, 0.04638, 0.0473, 0.04907, 0.05546]),
           st.sampled_from([0.007671, 0.009206, 0.01038, 0.01051, 0.01269, 0.0132, 0.01343, 0.01561, 0.01822, 0.03441]),
           st.sampled_from([0.01226, 0.01341, 0.01369, 0.01389, 0.01492, 0.01575, 0.01719, 0.02105, 0.02324, 0.02545]),
           st.sampled_from([0.0017, 0.001803, 0.001948, 0.002695, 0.002759, 0.003053, 0.003755, 0.006042, 0.009208, 0.009875]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=33.271, max_value=33.328, exclude_min=True, allow_nan=False),
           st.sampled_from([106.0, 113.9, 116.2, 127.1, 134.9, 135.1, 136.5, 162.3, 166.1, 177.4]),
           st.sampled_from([698.8, 888.3, 906.5, 1298.0, 1535.0, 1670.0, 1696.0, 1946.0, 2022.0, 2027.0]),
           st.sampled_from([0.08822, 0.1111, 0.1193, 0.1263, 0.1294, 0.1392, 0.1417, 0.1436, 0.1514, 0.1515]),
           st.sampled_from([0.1943, 0.1963, 0.2275, 0.2763, 0.3094, 0.3161, 0.3391, 0.3904, 0.4462, 0.5937]),
           st.sampled_from([0.02398, 0.2489, 0.3889, 0.4819, 0.4956, 0.5274, 0.5372, 0.5703, 0.5803, 0.7892]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2749, 0.3007, 0.3379, 0.3407, 0.3415, 0.3537, 0.3643, 0.3751, 0.4432, 0.4863]),
           st.sampled_from([0.06091, 0.06387, 0.06494, 0.0782, 0.08009, 0.08294, 0.08314, 0.09614, 0.1183, 0.124]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.726, 10.26, 10.32, 11.04, 11.7, 11.74, 12.25, 12.77, 12.95, 13.46]),
           st.sampled_from([12.27, 13.14, 13.47, 14.86, 15.86, 15.9, 16.52, 17.46, 19.73, 22.44]),
           st.sampled_from([43.79, 62.5, 70.92, 75.54, 77.61, 78.78, 85.31, 87.32, 87.38, 102.0]),
           st.sampled_from([272.5, 386.8, 388.1, 466.5, 506.9, 537.3, 561.0, 575.3, 592.6, 686.9]),
           st.sampled_from([0.07557, 0.08044, 0.08223, 0.08511, 0.086, 0.08759, 0.08801, 0.09345, 0.09879, 0.1004]),
           st.sampled_from([0.03116, 0.03789, 0.05473, 0.05743, 0.06602, 0.07074, 0.07234, 0.08061, 0.1073, 0.1316]),
           st.sampled_from([0.001487, 0.007173, 0.01103, 0.01367, 0.02688, 0.03102, 0.04721, 0.05282, 0.05928, 0.08777]),
           st.sampled_from([0.01076, 0.01108, 0.02017, 0.02173, 0.02293, 0.02315, 0.02456, 0.03068, 0.03483, 0.05397]),
           st.sampled_from([0.1528, 0.1637, 0.1657, 0.173, 0.1859, 0.2031, 0.2079, 0.2197, 0.2217, 0.2743]),
           st.sampled_from([0.05581, 0.05718, 0.05912, 0.06019, 0.06066, 0.06104, 0.06217, 0.06491, 0.06833, 0.07751]),
           st.sampled_from([0.1144, 0.1745, 0.1931, 0.2315, 0.2406, 0.2525, 0.2619, 0.2976, 0.3249, 0.4302]),
           st.sampled_from([0.4706, 0.4957, 0.6594, 0.7339, 1.219, 1.486, 1.597, 1.705, 1.911, 2.509]),
           st.sampled_from([1.101, 1.164, 1.484, 1.597, 1.686, 1.973, 2.0, 2.011, 2.465, 4.021]),
           st.floats(min_value=32.2443, max_value=38.6048, allow_nan=False),
           st.floats(min_value=0.003296, max_value=0.008862, exclude_min=True, allow_nan=False),
           st.sampled_from([0.003012, 0.01415, 0.01442, 0.01449, 0.01529, 0.01779, 0.02502, 0.02589, 0.06457, 0.0659]),
           st.sampled_from([0.006564, 0.007004, 0.007816, 0.01132, 0.01311, 0.01613, 0.0231, 0.02757, 0.04615, 0.06578]),
           st.sampled_from([0.00624, 0.007483, 0.008, 0.009128, 0.01056, 0.01164, 0.01493, 0.01712, 0.01774, 0.01966]),
           st.sampled_from([0.01322, 0.01394, 0.01551, 0.01651, 0.01718, 0.01829, 0.01938, 0.02466, 0.02542, 0.03504]),
           st.sampled_from([0.001432, 0.00152, 0.001566, 0.002464, 0.002585, 0.002619, 0.002985, 0.003696, 0.004406, 0.004726]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=33.561, max_value=36.756, exclude_min=True, allow_nan=False),
           st.sampled_from([72.01, 80.92, 81.25, 82.69, 87.0, 90.82, 91.29, 95.48, 96.74, 99.66]),
           st.sampled_from([240.1, 355.2, 437.0, 508.9, 522.9, 629.6, 633.7, 634.3, 708.8, 854.3]),
           st.sampled_from([0.09023, 0.09711, 0.1038, 0.1045, 0.1139, 0.1183, 0.1225, 0.1259, 0.1467, 0.1533]),
           st.sampled_from([0.1231, 0.1477, 0.1525, 0.1766, 0.1782, 0.1843, 0.2118, 0.2187, 0.2658, 0.2793]),
           st.sampled_from([0.0688, 0.07003, 0.07116, 0.07161, 0.08615, 0.1049, 0.1423, 0.1856, 0.2604, 0.4504]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2177, 0.2213, 0.225, 0.2434, 0.2478, 0.2522, 0.2527, 0.2725, 0.2787, 0.3604]),
           st.sampled_from([0.06291, 0.06306, 0.0641, 0.06443, 0.06788, 0.07048, 0.07863, 0.08187, 0.08304, 0.09825]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.4, 13.98, 14.27, 15.13, 15.46, 16.26, 16.74, 19.0, 19.53, 20.64]),
           st.sampled_from([16.4, 19.62, 19.66, 20.25, 22.33, 22.61, 23.56, 24.49, 27.06, 27.81]),
           st.sampled_from([102.1, 104.1, 114.5, 118.6, 130.0, 130.4, 130.7, 142.0, 152.1, 186.9]),
           st.sampled_from([475.9, 582.7, 671.4, 728.2, 732.4, 858.1, 933.1, 1169.0, 1293.0, 1419.0]),
           st.sampled_from([0.09029, 0.09055, 0.09057, 0.09823, 0.1036, 0.1041, 0.1064, 0.1084, 0.1169, 0.1178]),
           st.floats(min_value=0.051531, max_value=0.059568, allow_nan=False),
           st.sampled_from([0.04686, 0.05375, 0.06593, 0.0869, 0.09447, 0.1272, 0.1445, 0.1784, 0.203, 0.4268]),
           st.sampled_from([0.03085, 0.05364, 0.0539, 0.06597, 0.0734, 0.07785, 0.08886, 0.1242, 0.1401, 0.141]),
           st.sampled_from([0.1428, 0.1505, 0.159, 0.1663, 0.1974, 0.1998, 0.2082, 0.2085, 0.2151, 0.231]),
           st.sampled_from([0.05096, 0.05504, 0.05915, 0.05961, 0.06115, 0.0614, 0.06232, 0.06466, 0.07692, 0.08104]),
           st.sampled_from([0.2976, 0.3908, 0.4681, 0.5296, 0.5959, 0.7474, 0.9622, 1.058, 1.172, 2.873]),
           st.sampled_from([0.3621, 0.6999, 0.9173, 0.9832, 1.033, 1.186, 1.24, 1.599, 2.112, 2.836]),
           st.sampled_from([1.752, 2.11, 2.563, 2.961, 3.008, 3.195, 4.158, 4.174, 5.54, 10.05]),
           st.floats(min_value=38.6051, max_value=49.195, exclude_min=True, allow_nan=False),
           st.sampled_from([0.004029, 0.004044, 0.004877, 0.005043, 0.005345, 0.005444, 0.006429, 0.007499, 0.01149, 0.01345]),
           st.sampled_from([0.009269, 0.01376, 0.01427, 0.02203, 0.02616, 0.02785, 0.02855, 0.02928, 0.03799, 0.04844]),
           st.sampled_from([0.02151, 0.02375, 0.03185, 0.03342, 0.03457, 0.03909, 0.04257, 0.04741, 0.05042, 0.05196]),
           st.sampled_from([0.01051, 0.01167, 0.01267, 0.01459, 0.01479, 0.01569, 0.01678, 0.02311, 0.02593, 0.02801]),
           st.sampled_from([0.01069, 0.01388, 0.01465, 0.01479, 0.01495, 0.01518, 0.01543, 0.01697, 0.02045, 0.02186]),
           st.sampled_from([0.001465, 0.001976, 0.002085, 0.002142, 0.002436, 0.003042, 0.004028, 0.004515, 0.004571, 0.007444]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.sampled_from([24.3, 27.68, 28.65, 30.41, 30.7, 32.01, 33.48, 34.27, 34.85, 38.25]),
           st.sampled_from([113.8, 120.4, 121.4, 123.4, 124.1, 130.0, 132.7, 144.9, 145.4, 152.1]),
           st.sampled_from([508.1, 787.9, 826.4, 897.0, 1223.0, 1269.0, 1349.0, 1403.0, 1600.0, 1821.0]),
           st.sampled_from([0.1054, 0.1111, 0.1312, 0.1368, 0.1377, 0.1381, 0.1503, 0.1559, 0.1574, 0.1794]),
           st.sampled_from([0.1581, 0.2678, 0.284, 0.2947, 0.3416, 0.3903, 0.4126, 0.4706, 0.5634, 0.5717]),
           st.floats(min_value=0.26272, max_value=0.328399, allow_nan=False),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2572, 0.2623, 0.273, 0.2866, 0.2909, 0.3029, 0.306, 0.3537, 0.4432, 0.5166]),
           st.sampled_from([0.06251, 0.06637, 0.07632, 0.07787, 0.08019, 0.08368, 0.09187, 0.09333, 0.09614, 0.1123]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.878, 9.667, 12.16, 12.23, 12.94, 13.34, 13.56, 13.87, 14.29, 16.5]),
           st.sampled_from([13.44, 13.84, 14.97, 15.68, 17.36, 17.48, 18.54, 19.29, 19.83, 20.19]),
           st.sampled_from([56.74, 60.11, 66.86, 77.32, 82.01, 82.71, 84.95, 89.79, 90.3, 94.89]),
           st.sampled_from([181.0, 241.0, 349.6, 381.9, 388.1, 426.0, 428.9, 447.8, 461.4, 464.4]),
           st.sampled_from([0.07445, 0.07449, 0.07561, 0.08043, 0.08451, 0.08582, 0.08946, 0.09245, 0.1073, 0.1078]),
           st.floats(min_value=0.059571, max_value=0.116736, exclude_min=True, allow_nan=False),
           st.sampled_from([0.00751, 0.01923, 0.01993, 0.01997, 0.02367, 0.03476, 0.03546, 0.03974, 0.06664, 0.07741]),
           st.sampled_from([0.007246, 0.02292, 0.02343, 0.02344, 0.02864, 0.03027, 0.03088, 0.0389, 0.04812, 0.05397]),
           st.sampled_from([0.1487, 0.159, 0.165, 0.1659, 0.1701, 0.1722, 0.1859, 0.1874, 0.197, 0.2015]),
           st.sampled_from([0.05243, 0.05593, 0.0566, 0.05852, 0.061, 0.06372, 0.06581, 0.06899, 0.07125, 0.07669]),
           st.sampled_from([0.1312, 0.1753, 0.2005, 0.2212, 0.2589, 0.2957, 0.3237, 0.3833, 0.4993, 0.5115]),
           st.sampled_from([0.7815, 0.9112, 0.948, 0.9823, 1.044, 1.046, 1.363, 1.409, 1.687, 1.879]),
           st.sampled_from([1.036, 1.236, 1.281, 1.429, 1.475, 1.874, 2.044, 2.105, 2.355, 3.814]),
           st.floats(min_value=38.6051, max_value=49.195, exclude_min=True, allow_nan=False),
           st.sampled_from([0.003308, 0.005012, 0.005096, 0.007762, 0.007802, 0.008263, 0.008534, 0.009006, 0.01097, 0.02177]),
           st.sampled_from([0.008539, 0.008776, 0.009216, 0.009947, 0.01123, 0.01205, 0.01469, 0.01777, 0.01903, 0.02667]),
           st.sampled_from([0.00186, 0.005949, 0.01988, 0.01993, 0.03016, 0.04017, 0.04167, 0.04305, 0.07927, 0.1435]),
           st.sampled_from([0.004832, 0.007369, 0.007497, 0.009333, 0.01121, 0.01136, 0.01162, 0.01167, 0.01493, 0.01721]),
           st.sampled_from([0.01359, 0.01528, 0.01698, 0.0172, 0.0194, 0.02538, 0.02632, 0.02719, 0.02869, 0.03799]),
           st.sampled_from([0.001392, 0.001767, 0.002582, 0.002619, 0.002671, 0.002801, 0.003009, 0.003705, 0.004572, 0.009627]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.sampled_from([18.2, 19.49, 25.05, 25.34, 27.82, 28.26, 31.89, 32.84, 34.24, 37.88]),
           st.sampled_from([71.68, 71.98, 72.01, 83.24, 85.51, 86.65, 89.0, 93.63, 98.0, 100.3]),
           st.sampled_from([303.8, 367.0, 583.0, 628.5, 632.9, 680.6, 705.6, 749.9, 767.3, 809.8]),
           st.sampled_from([0.09388, 0.09527, 0.09701, 0.1206, 0.1221, 0.1235, 0.1276, 0.1333, 0.1389, 0.1405]),
           st.sampled_from([0.07057, 0.0872, 0.08978, 0.09515, 0.1193, 0.1975, 0.2042, 0.2515, 0.2772, 0.3429]),
           st.floats(min_value=0.26272, max_value=0.328399, allow_nan=False),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2208, 0.222, 0.2226, 0.2262, 0.2346, 0.2454, 0.2642, 0.31, 0.3196, 0.323]),
           st.sampled_from([0.05871, 0.06025, 0.06743, 0.07418, 0.07626, 0.07806, 0.07858, 0.08147, 0.08284, 0.1364]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.17, 13.86, 15.12, 19.27, 19.73, 20.2, 21.16, 23.09, 23.27, 23.29]),
           st.sampled_from([19.33, 19.82, 20.66, 20.71, 21.38, 23.29, 23.98, 25.27, 25.56, 29.33]),
           st.sampled_from([92.87, 94.49, 95.55, 96.42, 100.4, 118.6, 124.8, 129.1, 135.7, 137.8]),
           st.sampled_from([642.7, 648.2, 701.9, 713.3, 719.5, 994.0, 1148.0, 1167.0, 1306.0, 1546.0]),
           st.sampled_from([0.08331, 0.08439, 0.0909, 0.1012, 0.1024, 0.1062, 0.111, 0.1172, 0.123, 0.1398]),
           st.sampled_from([0.06722, 0.09182, 0.1023, 0.1076, 0.1133, 0.1273, 0.1571, 0.1954, 0.2665, 0.2776]),
           st.sampled_from([0.02685, 0.1044, 0.1153, 0.1411, 0.1445, 0.191, 0.1932, 0.2107, 0.231, 0.2712]),
           st.sampled_from([0.05778, 0.06759, 0.08399, 0.08543, 0.08886, 0.09353, 0.09479, 0.09711, 0.1244, 0.1913]),
           st.sampled_from([0.1308, 0.1582, 0.1627, 0.1927, 0.1943, 0.195, 0.1953, 0.2106, 0.2183, 0.2556]),
           st.sampled_from([0.0551, 0.0558, 0.05613, 0.05656, 0.05796, 0.05916, 0.06183, 0.06768, 0.06777, 0.07115]),
           st.sampled_from([0.2787, 0.3331, 0.37, 0.4041, 0.4388, 0.5659, 0.6361, 0.9806, 1.111, 1.296]),
           st.sampled_from([0.6123, 0.6575, 0.6633, 0.8225, 0.8339, 0.8413, 0.8737, 0.9988, 1.077, 1.476]),
           st.sampled_from([1.903, 2.642, 2.937, 3.445, 3.477, 4.303, 4.369, 4.837, 5.865, 7.05]),
           st.floats(min_value=38.6051, max_value=49.195, exclude_min=True, allow_nan=False),
           st.sampled_from([0.004766, 0.005726, 0.005731, 0.005769, 0.005771, 0.006113, 0.006471, 0.006789, 0.007571, 0.0103]),
           st.sampled_from([0.02008, 0.02569, 0.02644, 0.03033, 0.03481, 0.03799, 0.0431, 0.04588, 0.04732, 0.1006]),
           st.sampled_from([0.01715, 0.01818, 0.02975, 0.03052, 0.04257, 0.04435, 0.04502, 0.04531, 0.04683, 0.04755]),
           st.sampled_from([0.009567, 0.009643, 0.01195, 0.01345, 0.01444, 0.01463, 0.01806, 0.01883, 0.02139, 0.02536]),
           st.sampled_from([0.01065, 0.0122, 0.01357, 0.01386, 0.01686, 0.01743, 0.0214, 0.02324, 0.02789, 0.04783]),
           st.sampled_from([0.00243, 0.002498, 0.00255, 0.002575, 0.002871, 0.002887, 0.003391, 0.003817, 0.003854, 0.01008]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.sampled_from([21.43, 22.88, 25.0, 25.73, 26.44, 26.56, 28.12, 28.45, 31.39, 33.62]),
           st.sampled_from([108.1, 113.7, 130.9, 137.9, 139.8, 142.0, 142.1, 153.2, 160.5, 195.9]),
           st.sampled_from([553.6, 741.6, 750.1, 759.4, 981.2, 1760.0, 1933.0, 2022.0, 2053.0, 2227.0]),
           st.sampled_from([0.1148, 0.1168, 0.1365, 0.1407, 0.1412, 0.1414, 0.1497, 0.1533, 0.1732, 0.1789]),
           st.sampled_from([0.1793, 0.205, 0.2057, 0.2336, 0.2947, 0.2968, 0.3463, 0.4061, 0.4244, 0.4503]),
           st.floats(min_value=0.328402, max_value=0.513121, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2268, 0.248, 0.2567, 0.2593, 0.2683, 0.3021, 0.3175, 0.3383, 0.3792, 0.4245]),
           st.sampled_from([0.05737, 0.06494, 0.07146, 0.07397, 0.07625, 0.08328, 0.08666, 0.09288, 0.09438, 0.1026]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([12.83, 13.0, 16.02, 16.07, 18.82, 19.1, 20.44, 21.37, 21.71, 23.27]),
           st.sampled_from([15.1, 15.51, 18.15, 20.26, 20.52, 21.08, 22.04, 22.13, 26.57, 28.25]),
           st.sampled_from([105.7, 109.8, 110.2, 117.3, 124.8, 126.5, 130.4, 147.3, 165.5, 171.5]),
           st.sampled_from([371.1, 572.6, 698.8, 705.6, 817.7, 928.3, 1094.0, 1148.0, 1203.0, 1217.0]),
           st.sampled_from([0.08682, 0.0926, 0.09684, 0.09867, 0.1, 0.106, 0.1088, 0.1141, 0.116, 0.1326]),
           st.sampled_from([0.04605, 0.05616, 0.07081, 0.08564, 0.1098, 0.1469, 0.1649, 0.1666, 0.1669, 0.2832]),
           st.sampled_from([0.05285, 0.09954, 0.1168, 0.1323, 0.1411, 0.1659, 0.1856, 0.191, 0.1974, 0.3635]),
           st.sampled_from([0.02847, 0.05252, 0.05613, 0.08646, 0.08886, 0.09333, 0.1043, 0.1121, 0.1279, 0.1595]),
           st.sampled_from([0.1428, 0.1582, 0.1767, 0.1784, 0.1852, 0.1953, 0.1973, 0.1995, 0.2152, 0.2395]),
           st.sampled_from([0.05557, 0.0558, 0.06071, 0.06083, 0.06113, 0.06222, 0.06232, 0.06916, 0.06924, 0.07083]),
           st.sampled_from([0.2385, 0.2711, 0.3061, 0.386, 0.4615, 0.5659, 0.6289, 0.645, 0.7049, 1.291]),
           st.sampled_from([0.6062, 0.7859, 0.9209, 0.9988, 1.03, 1.152, 1.202, 1.278, 1.679, 2.836]),
           st.sampled_from([1.344, 2.097, 2.735, 3.218, 3.598, 4.061, 4.837, 7.804, 8.077, 10.12]),
           st.floats(min_value=91.5552, max_value=181.6841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.002866, 0.00553, 0.005607, 0.006003, 0.006113, 0.006458, 0.007149, 0.009369, 0.01056, 0.01124]),
           st.sampled_from([0.01578, 0.02321, 0.02563, 0.02616, 0.03033, 0.03481, 0.03611, 0.04674, 0.05839, 0.1354]),
           st.sampled_from([0.01412, 0.02375, 0.02664, 0.02913, 0.03011, 0.03342, 0.03452, 0.03582, 0.04665, 0.0573]),
           st.sampled_from([0.009753, 0.01167, 0.01276, 0.01291, 0.01293, 0.01569, 0.01601, 0.01834, 0.02156, 0.03441]),
           st.sampled_from([0.01069, 0.01377, 0.01522, 0.01675, 0.01717, 0.01964, 0.02018, 0.02091, 0.0225, 0.03756]),
           st.sampled_from([0.001589, 0.003187, 0.003892, 0.004367, 0.00457, 0.005002, 0.005099, 0.005667, 0.005815, 0.009875]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.sampled_from([26.2, 26.58, 28.07, 28.65, 30.15, 31.47, 34.12, 35.34, 40.68, 47.16]),
           st.sampled_from([108.8, 119.1, 123.5, 132.9, 134.9, 140.9, 142.1, 149.3, 152.1, 168.2]),
           st.sampled_from([591.7, 706.0, 906.5, 909.4, 1084.0, 1227.0, 1408.0, 1651.0, 1866.0, 2360.0]),
           st.sampled_from([0.1265, 0.1377, 0.138, 0.1416, 0.1435, 0.1449, 0.1471, 0.1497, 0.1665, 0.1789]),
           st.sampled_from([0.1997, 0.2884, 0.2942, 0.3262, 0.3331, 0.3463, 0.3539, 0.3547, 0.3835, 0.4648]),
           st.sampled_from([0.3349, 0.3965, 0.3976, 0.4316, 0.4504, 0.4634, 0.5165, 0.6181, 0.7242, 0.9608]),
           st.floats(min_value=0.10864, max_value=0.135799, allow_nan=False),
           st.sampled_from([0.2572, 0.277, 0.2807, 0.3003, 0.3258, 0.3274, 0.363, 0.4066, 0.4154, 0.4824]),
           st.sampled_from([0.0761, 0.07944, 0.08019, 0.08482, 0.0895, 0.09946, 0.1019, 0.1031, 0.1059, 0.1132]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.95, 9.742, 9.777, 11.16, 12.34, 14.41, 14.42, 14.44, 14.96, 15.04]),
           st.sampled_from([12.74, 13.16, 15.73, 15.76, 17.19, 17.3, 20.76, 25.42, 29.97, 33.81]),
           st.sampled_from([65.31, 72.23, 73.16, 75.71, 77.61, 79.19, 84.52, 86.1, 86.87, 87.16]),
           st.sampled_from([227.2, 396.0, 412.5, 462.9, 465.4, 493.1, 530.6, 559.2, 566.3, 646.1]),
           st.sampled_from([0.06613, 0.06883, 0.07618, 0.07956, 0.0808, 0.08142, 0.08992, 0.09037, 0.09423, 0.09516]),
           st.sampled_from([0.04971, 0.05991, 0.06779, 0.07079, 0.07849, 0.0812, 0.08498, 0.1296, 0.1599, 0.1661]),
           st.sampled_from([0.008306, 0.01053, 0.01369, 0.01974, 0.02399, 0.02602, 0.04069, 0.06335, 0.06824, 0.1169]),
           st.sampled_from([0.01043, 0.0129, 0.01553, 0.01699, 0.01775, 0.01875, 0.01883, 0.02421, 0.02657, 0.06987]),
           st.sampled_from([0.1203, 0.1337, 0.1409, 0.1539, 0.1664, 0.1705, 0.1714, 0.1957, 0.2003, 0.2031]),
           st.sampled_from([0.05586, 0.05718, 0.06287, 0.06331, 0.06447, 0.06639, 0.06761, 0.06891, 0.08046, 0.0845]),
           st.sampled_from([0.1115, 0.2137, 0.2338, 0.2446, 0.2512, 0.2719, 0.3563, 0.3776, 0.5262, 0.6061]),
           st.sampled_from([0.3871, 0.489, 0.6745, 0.7656, 0.8652, 0.9505, 1.059, 1.918, 2.06, 2.878]),
           st.sampled_from([1.471, 1.535, 1.577, 1.648, 1.66, 1.75, 1.937, 2.087, 2.143, 2.326]),
           st.sampled_from([11.64, 16.41, 16.57, 17.4, 18.39, 21.55, 25.06, 29.84, 33.0, 41.24]),
           st.sampled_from([0.00398, 0.004477, 0.005463, 0.005501, 0.005518, 0.006011, 0.006588, 0.006719, 0.007278, 0.007389]),
           st.sampled_from([0.008491, 0.01079, 0.01515, 0.01561, 0.01735, 0.01966, 0.02222, 0.02348, 0.02417, 0.04877]),
           st.sampled_from([0.008268, 0.008732, 0.01334, 0.01529, 0.01536, 0.0184, 0.0208, 0.02176, 0.02575, 0.07683]),
           st.sampled_from([0.004821, 0.00646, 0.007638, 0.009128, 0.01038, 0.012, 0.01269, 0.01364, 0.0158, 0.01623]),
           st.sampled_from([0.01561, 0.01807, 0.01897, 0.01989, 0.02266, 0.02427, 0.02625, 0.02678, 0.02711, 0.02728]),
           st.sampled_from([0.001755, 0.001802, 0.001858, 0.002211, 0.002304, 0.002585, 0.002808, 0.003002, 0.004143, 0.006872]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=22.94, max_value=25.669, allow_nan=False),
           st.sampled_from([68.73, 76.25, 77.98, 78.28, 84.35, 86.04, 86.6, 88.54, 95.78, 102.8]),
           st.floats(min_value=685.28, max_value=810.29, allow_nan=False),
           st.floats(min_value=0.157074, max_value=0.178549, allow_nan=False),
           st.sampled_from([0.04953, 0.1247, 0.1478, 0.1644, 0.1724, 0.1879, 0.1928, 0.1949, 0.2031, 0.4202]),
           st.sampled_from([0.05186, 0.08803, 0.09203, 0.1055, 0.1246, 0.186, 0.1876, 0.1904, 0.2806, 0.5381]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2447, 0.2465, 0.2513, 0.2639, 0.2688, 0.2757, 0.2762, 0.2983, 0.3267, 0.3469]),
           st.sampled_from([0.05521, 0.05974, 0.06435, 0.06443, 0.0671, 0.07012, 0.07686, 0.08865, 0.09938, 0.1016]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([10.95, 13.0, 14.45, 15.13, 16.35, 16.65, 18.63, 20.26, 21.61, 27.42]),
           st.sampled_from([17.52, 18.52, 19.07, 19.63, 20.13, 20.67, 23.03, 24.49, 25.56, 26.29]),
           st.sampled_from([77.93, 86.18, 92.87, 97.4, 98.0, 100.3, 106.2, 112.4, 121.4, 153.5]),
           st.sampled_from([477.4, 499.0, 642.7, 712.8, 840.4, 933.1, 982.0, 1033.0, 1152.0, 1203.0]),
           st.sampled_from([0.09055, 0.0961, 0.09797, 0.09823, 0.09872, 0.1018, 0.1026, 0.1073, 0.1141, 0.115]),
           st.sampled_from([0.05761, 0.07304, 0.08468, 0.1154, 0.1402, 0.1453, 0.1479, 0.198, 0.2004, 0.2768]),
           st.sampled_from([0.09875, 0.1127, 0.1266, 0.1793, 0.1855, 0.203, 0.2065, 0.2188, 0.2195, 0.3176]),
           st.sampled_from([0.06463, 0.06638, 0.07483, 0.07488, 0.08271, 0.08886, 0.09029, 0.09333, 0.1155, 0.1878]),
           st.sampled_from([0.1506, 0.1565, 0.1582, 0.1662, 0.1696, 0.1765, 0.1893, 0.1929, 0.1943, 0.1989]),
           st.sampled_from([0.05044, 0.05478, 0.05647, 0.0577, 0.06053, 0.06097, 0.06194, 0.06277, 0.06898, 0.07369]),
           st.sampled_from([0.2873, 0.3971, 0.4768, 0.5058, 0.5858, 0.5907, 0.7128, 0.9553, 1.009, 1.072]),
           st.sampled_from([0.6062, 0.6509, 0.6857, 0.8249, 0.9004, 0.9832, 1.012, 1.078, 1.152, 1.24]),
           st.sampled_from([1.534, 1.752, 2.244, 2.45, 2.587, 2.63, 3.061, 3.564, 4.706, 6.372]),
           st.sampled_from([31.0, 33.01, 40.73, 43.5, 54.04, 56.18, 57.72, 69.65, 96.05, 130.2]),
           st.sampled_from([0.004551, 0.004714, 0.004756, 0.004821, 0.005524, 0.005872, 0.006428, 0.006717, 0.008081, 0.009087]),
           st.sampled_from([0.009105, 0.01893, 0.03057, 0.03082, 0.03368, 0.03976, 0.04061, 0.04308, 0.05328, 0.08668]),
           st.sampled_from([0.02291, 0.02855, 0.02905, 0.02975, 0.0404, 0.04232, 0.04345, 0.05501, 0.07743, 0.08079]),
           st.sampled_from([0.007671, 0.009222, 0.009767, 0.009863, 0.01043, 0.01093, 0.01195, 0.0142, 0.01508, 0.02149]),
           st.sampled_from([0.01369, 0.01414, 0.01415, 0.01495, 0.01522, 0.01591, 0.01675, 0.01686, 0.01798, 0.01998]),
           st.sampled_from([0.002365, 0.002461, 0.002846, 0.002887, 0.003727, 0.005126, 0.006193, 0.006995, 0.007098, 0.01284]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=22.94, max_value=25.669, allow_nan=False),
           st.sampled_from([102.2, 119.1, 120.4, 129.1, 130.9, 132.8, 136.1, 160.2, 180.9, 195.0]),
           st.floats(min_value=685.28, max_value=810.29, allow_nan=False),
           st.floats(min_value=0.178552, max_value=0.187361, exclude_min=True, allow_nan=False),
           st.sampled_from([0.05131, 0.1486, 0.3885, 0.4056, 0.4099, 0.4244, 0.4256, 0.5564, 0.5775, 0.5955]),
           st.sampled_from([0.2606, 0.2675, 0.2802, 0.3728, 0.3853, 0.3948, 0.4317, 0.5165, 0.583, 0.6869]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2341, 0.2576, 0.273, 0.2768, 0.2812, 0.3379, 0.3437, 0.347, 0.3591, 0.3799]),
           st.sampled_from([0.06558, 0.07277, 0.07371, 0.07425, 0.07787, 0.08482, 0.09606, 0.1026, 0.1198, 0.1275]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([9.567, 9.742, 10.05, 10.94, 11.5, 12.56, 12.58, 13.14, 13.74, 14.62]),
           st.sampled_from([13.1, 14.59, 15.5, 16.95, 16.99, 18.3, 19.54, 20.21, 21.37, 21.48]),
           st.sampled_from([56.74, 61.64, 70.41, 75.46, 78.04, 82.67, 87.19, 87.91, 90.31, 97.53]),
           st.sampled_from([349.6, 358.9, 384.8, 402.7, 412.5, 442.7, 481.6, 538.9, 542.9, 588.7]),
           st.sampled_from([0.07372, 0.07561, 0.07838, 0.07966, 0.08675, 0.0876, 0.09231, 0.09882, 0.1016, 0.1061]),
           st.floats(min_value=0.100596, max_value=0.120899, allow_nan=False),
           st.sampled_from([0.001487, 0.03393, 0.03592, 0.04057, 0.04063, 0.04249, 0.06824, 0.07741, 0.08777, 0.145]),
           st.sampled_from([0.005159, 0.01969, 0.02008, 0.02733, 0.02738, 0.0295, 0.03099, 0.04781, 0.063, 0.06615]),
           st.sampled_from([0.1705, 0.1709, 0.1721, 0.1743, 0.1744, 0.1801, 0.1897, 0.203, 0.211, 0.2116]),
           st.sampled_from([0.05266, 0.05653, 0.0566, 0.05696, 0.05808, 0.06228, 0.06503, 0.0685, 0.06908, 0.07187]),
           st.sampled_from([0.1153, 0.1312, 0.1408, 0.2213, 0.23, 0.28, 0.286, 0.2949, 0.3778, 0.3833]),
           st.sampled_from([0.4801, 0.5996, 0.6656, 0.9264, 1.042, 1.059, 1.231, 1.571, 1.916, 2.878]),
           st.sampled_from([1.355, 1.392, 1.535, 1.67, 1.687, 1.955, 2.602, 2.829, 3.027, 3.564]),
           st.sampled_from([9.597, 16.16, 17.49, 18.32, 19.01, 24.2, 25.03, 26.43, 27.24, 27.49]),
           st.sampled_from([0.003958, 0.005528, 0.006538, 0.006664, 0.006773, 0.007334, 0.007595, 0.007976, 0.008732, 0.01736]),
           st.sampled_from([0.004899, 0.008432, 0.009362, 0.01047, 0.01174, 0.01432, 0.0182, 0.02337, 0.02417, 0.09586]),
           st.sampled_from([0.002074, 0.003681, 0.003846, 0.007004, 0.01328, 0.02095, 0.03137, 0.03214, 0.05112, 0.0888]),
           st.sampled_from([0.005383, 0.005872, 0.005917, 0.008038, 0.009259, 0.009366, 0.01024, 0.01107, 0.01367, 0.01519]),
           st.sampled_from([0.01414, 0.01487, 0.01609, 0.01865, 0.01894, 0.01943, 0.0256, 0.02734, 0.02769, 0.03675]),
           st.sampled_from([0.001566, 0.001725, 0.001769, 0.002228, 0.002476, 0.002619, 0.002808, 0.002925, 0.005715, 0.007877]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=22.94, max_value=25.669, allow_nan=False),
           st.sampled_from([69.86, 79.82, 86.16, 89.88, 96.09, 99.16, 99.43, 100.2, 101.7, 107.1]),
           st.floats(min_value=810.32, max_value=1499.05, exclude_min=True, allow_nan=False),
           st.sampled_from([0.08774, 0.106, 0.111, 0.119, 0.1227, 0.124, 0.1333, 0.1338, 0.1359, 0.1415]),
           st.sampled_from([0.04327, 0.05445, 0.09358, 0.1202, 0.1257, 0.1357, 0.1457, 0.2231, 0.2506, 0.295]),
           st.sampled_from([0.005518, 0.005579, 0.01824, 0.07153, 0.1346, 0.135, 0.1453, 0.1673, 0.2302, 0.2873]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2376, 0.245, 0.2525, 0.2731, 0.2738, 0.2744, 0.2954, 0.3062, 0.3065, 0.3379]),
           st.sampled_from([0.05974, 0.06769, 0.07062, 0.0732, 0.07875, 0.08052, 0.08175, 0.0896, 0.09464, 0.09585]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.8, 14.68, 14.95, 15.08, 17.46, 17.6, 19.53, 19.59, 20.09, 20.26]),
           st.sampled_from([18.15, 20.31, 20.58, 20.82, 21.24, 21.56, 22.91, 25.74, 26.83, 29.33]),
           st.sampled_from([75.0, 91.56, 97.26, 100.2, 105.7, 108.1, 109.7, 110.2, 123.7, 125.5]),
           st.sampled_from([578.3, 582.7, 698.8, 704.4, 719.5, 766.6, 984.6, 1027.0, 1123.0, 1247.0]),
           st.sampled_from([0.08401, 0.08523, 0.0961, 0.09847, 0.102, 0.106, 0.1092, 0.115, 0.1215, 0.1273]),
           st.floats(min_value=0.120902, max_value=0.165801, exclude_min=True, allow_nan=False),
           st.sampled_from([0.04686, 0.1103, 0.1168, 0.1293, 0.1354, 0.164, 0.1657, 0.195, 0.2283, 0.3368]),
           st.sampled_from([0.05252, 0.05271, 0.05613, 0.0734, 0.08591, 0.08923, 0.09429, 0.1021, 0.1097, 0.1149]),
           st.sampled_from([0.1594, 0.1713, 0.1733, 0.1746, 0.1801, 0.1885, 0.1893, 0.1956, 0.2248, 0.2556]),
           st.sampled_from([0.05674, 0.05697, 0.05892, 0.05916, 0.06303, 0.06343, 0.06382, 0.06673, 0.06937, 0.07331]),
           st.sampled_from([0.2563, 0.2787, 0.4203, 0.4357, 0.4565, 0.4727, 0.4956, 0.5204, 0.5907, 0.6226]),
           st.sampled_from([0.4956, 0.7339, 0.9173, 0.9489, 0.976, 1.041, 1.309, 1.324, 1.849, 2.11]),
           st.sampled_from([1.534, 2.257, 2.362, 2.407, 2.735, 2.972, 3.868, 5.574, 6.487, 8.83]),
           st.sampled_from([31.0, 48.9, 54.22, 63.37, 72.44, 87.17, 94.03, 94.44, 115.2, 153.1]),
           st.sampled_from([0.004626, 0.004989, 0.005033, 0.005753, 0.005776, 0.00609, 0.006429, 0.007162, 0.008109, 0.009197]),
           st.sampled_from([0.01384, 0.01478, 0.01882, 0.02101, 0.02203, 0.02321, 0.02499, 0.03438, 0.03756, 0.07056]),
           st.sampled_from([0.02315, 0.02332, 0.02817, 0.04649, 0.04983, 0.06072, 0.06577, 0.07359, 0.07649, 0.08958]),
           st.sampled_from([0.006719, 0.006881, 0.008637, 0.00928, 0.01093, 0.01384, 0.01458, 0.01597, 0.01883, 0.03024]),
           st.sampled_from([0.01177, 0.01415, 0.01467, 0.01486, 0.01591, 0.01703, 0.01756, 0.02091, 0.02201, 0.0237]),
           st.sampled_from([0.001754, 0.002725, 0.002887, 0.003532, 0.004417, 0.004968, 0.005617, 0.006, 0.006113, 0.01039]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=22.94, max_value=25.669, allow_nan=False),
           st.sampled_from([85.1, 102.8, 115.0, 119.4, 128.5, 145.3, 151.7, 152.0, 157.6, 159.8]),
           st.floats(min_value=810.32, max_value=1499.05, exclude_min=True, allow_nan=False),
           st.sampled_from([0.1124, 0.1207, 0.1365, 0.1416, 0.1498, 0.1503, 0.1512, 0.1515, 0.1737, 0.1791]),
           st.sampled_from([0.2275, 0.2576, 0.2733, 0.3161, 0.3262, 0.3309, 0.3856, 0.4257, 0.4492, 0.4967]),
           st.sampled_from([0.2687, 0.3194, 0.3207, 0.3215, 0.3809, 0.4433, 0.4819, 0.4932, 0.6305, 1.17]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.1978, 0.2463, 0.248, 0.2736, 0.3003, 0.3029, 0.3206, 0.3751, 0.39, 0.4677]),
           st.sampled_from([0.07397, 0.07625, 0.07944, 0.08067, 0.08075, 0.08218, 0.0895, 0.1019, 0.1132, 0.1243]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([9.295, 9.742, 10.25, 10.82, 11.61, 11.84, 13.11, 13.24, 13.54, 15.0]),
           st.floats(min_value=18.305, max_value=20.453, allow_nan=False),
           st.sampled_from([60.21, 65.31, 66.72, 70.47, 77.42, 78.6, 81.89, 82.02, 88.06, 102.8]),
           st.sampled_from([384.6, 384.8, 388.1, 408.2, 502.5, 513.7, 516.4, 537.3, 689.4, 761.7]),
           st.sampled_from([0.07896, 0.07937, 0.08123, 0.08671, 0.0903, 0.0907, 0.09357, 0.09855, 0.1039, 0.1043]),
           st.sampled_from([0.03393, 0.0363, 0.06678, 0.06679, 0.06779, 0.07658, 0.07808, 0.09453, 0.0958, 0.1535]),
           st.sampled_from([0.00309, 0.007173, 0.02172, 0.0236, 0.02367, 0.02948, 0.02956, 0.04187, 0.04944, 0.1065]),
           st.floats(min_value=0.043279, max_value=0.054098, allow_nan=False),
           st.sampled_from([0.1215, 0.1486, 0.1508, 0.1543, 0.1574, 0.1697, 0.1701, 0.172, 0.1959, 0.2548]),
           st.sampled_from([0.05502, 0.05541, 0.06246, 0.06275, 0.06403, 0.0654, 0.06582, 0.07187, 0.07279, 0.08261]),
           st.sampled_from([0.1344, 0.2023, 0.2073, 0.236, 0.2431, 0.2525, 0.256, 0.3342, 0.5115, 0.5196]),
           st.sampled_from([0.4875, 0.7151, 0.8927, 1.003, 1.143, 1.194, 1.28, 1.502, 1.768, 1.786]),
           st.sampled_from([1.046, 1.184, 1.208, 1.564, 2.077, 2.171, 2.455, 2.561, 2.597, 2.635]),
           st.sampled_from([11.6, 12.33, 14.91, 19.14, 19.33, 20.2, 27.24, 27.49, 32.74, 48.29]),
           st.sampled_from([0.004097, 0.0042, 0.004259, 0.004271, 0.005682, 0.006142, 0.006261, 0.00747, 0.01094, 0.01127]),
           st.sampled_from([0.01067, 0.01082, 0.01097, 0.01203, 0.0125, 0.01382, 0.01541, 0.01555, 0.02736, 0.03051]),
           st.sampled_from([0.007508, 0.008268, 0.00941, 0.009959, 0.01123, 0.01256, 0.01349, 0.01451, 0.02221, 0.04275]),
           st.sampled_from([0.004832, 0.004967, 0.006747, 0.007638, 0.00851, 0.008747, 0.009046, 0.01065, 0.01293, 0.01721]),
           st.sampled_from([0.01466, 0.015, 0.01544, 0.0156, 0.01848, 0.0198, 0.0208, 0.02837, 0.03141, 0.03433]),
           st.sampled_from([0.001656, 0.001802, 0.002228, 0.002248, 0.00253, 0.00354, 0.003674, 0.004572, 0.005875, 0.008675]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=25.672, max_value=30.445, exclude_min=True, allow_nan=False),
           st.sampled_from([79.93, 83.12, 86.57, 88.52, 102.3, 102.8, 109.4, 112.0, 114.2, 114.3]),
           st.sampled_from([412.3, 503.0, 516.5, 547.8, 580.6, 602.0, 661.1, 745.5, 829.5, 830.5]),
           st.sampled_from([0.1025, 0.1085, 0.1249, 0.1314, 0.1342, 0.1343, 0.1402, 0.1494, 0.163, 0.178]),
           st.sampled_from([0.08842, 0.09726, 0.112, 0.1477, 0.1513, 0.165, 0.1679, 0.1963, 0.2399, 0.3842]),
           st.sampled_from([0.04043, 0.06648, 0.1412, 0.1514, 0.1564, 0.2, 0.2299, 0.2456, 0.269, 0.366]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2102, 0.2438, 0.2599, 0.2615, 0.274, 0.2767, 0.3057, 0.3075, 0.327, 0.3387]),
           st.sampled_from([0.0641, 0.06878, 0.06969, 0.07182, 0.0738, 0.07538, 0.07764, 0.08492, 0.08718, 0.09359]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.44, 14.86, 14.87, 19.1, 19.4, 20.18, 21.56, 22.27, 23.27, 24.63]),
           st.floats(min_value=20.456, max_value=24.22, exclude_min=True, allow_nan=False),
           st.sampled_from([78.99, 90.63, 93.63, 96.42, 96.85, 98.78, 110.1, 127.7, 129.5, 130.7]),
           st.sampled_from([565.4, 578.3, 578.9, 597.8, 648.2, 963.7, 1024.0, 1167.0, 1299.0, 1384.0]),
           st.sampled_from([0.09009, 0.09488, 0.09491, 0.101, 0.1037, 0.1092, 0.1106, 0.112, 0.1148, 0.116]),
           st.sampled_from([0.111, 0.113, 0.1298, 0.1402, 0.1559, 0.1697, 0.17, 0.2004, 0.2135, 0.3454]),
           st.sampled_from([0.0311, 0.05253, 0.05375, 0.1063, 0.1122, 0.1127, 0.1128, 0.1411, 0.1659, 0.2136]),
           st.floats(min_value=0.043279, max_value=0.054098, allow_nan=False),
           st.sampled_from([0.1538, 0.1648, 0.1669, 0.1752, 0.1802, 0.1832, 0.2092, 0.2095, 0.2106, 0.2521]),
           st.sampled_from([0.05044, 0.05504, 0.05636, 0.05647, 0.06115, 0.06177, 0.06287, 0.06466, 0.07083, 0.07692]),
           st.sampled_from([0.2602, 0.2986, 0.3906, 0.4007, 0.4312, 0.6003, 0.6997, 0.9622, 0.9806, 1.37]),
           st.sampled_from([0.5679, 0.6633, 0.8225, 1.051, 1.069, 1.426, 1.481, 1.506, 1.961, 2.463]),
           st.sampled_from([3.142, 3.301, 3.445, 4.174, 4.293, 4.533, 5.144, 5.353, 8.758, 9.807]),
           st.sampled_from([19.53, 22.18, 27.19, 31.59, 40.09, 68.46, 87.78, 88.25, 112.4, 156.8]),
           st.sampled_from([0.004029, 0.004057, 0.004314, 0.004821, 0.00486, 0.005367, 0.005769, 0.006548, 0.006766, 0.007964]),
           st.sampled_from([0.009181, 0.01106, 0.01384, 0.018, 0.02008, 0.02083, 0.02616, 0.03715, 0.03756, 0.05374]),
           st.sampled_from([0.02647, 0.02813, 0.02975, 0.03185, 0.03342, 0.03909, 0.04257, 0.04345, 0.06389, 0.08958]),
           st.sampled_from([0.009567, 0.01137, 0.01184, 0.01195, 0.01293, 0.01407, 0.01628, 0.01712, 0.02149, 0.02397]),
           st.sampled_from([0.01323, 0.01369, 0.01465, 0.01467, 0.01578, 0.01703, 0.01768, 0.01857, 0.02428, 0.04783]),
           st.sampled_from([0.001589, 0.002299, 0.002608, 0.002658, 0.002881, 0.002887, 0.003288, 0.004108, 0.005838, 0.006193]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=25.672, max_value=30.445, exclude_min=True, allow_nan=False),
           st.sampled_from([111.6, 113.8, 123.8, 124.9, 132.8, 141.3, 153.9, 161.2, 163.2, 171.1]),
           st.sampled_from([975.2, 1229.0, 1410.0, 1436.0, 1813.0, 1821.0, 2081.0, 2615.0, 2944.0, 3143.0]),
           st.sampled_from([0.1072, 0.1368, 0.139, 0.1396, 0.1492, 0.1497, 0.1512, 0.1592, 0.1765, 0.1883]),
           st.sampled_from([0.1516, 0.1807, 0.2101, 0.2311, 0.3206, 0.3309, 0.3749, 0.4967, 0.5775, 0.9379]),
           st.sampled_from([0.2477, 0.3458, 0.3597, 0.3759, 0.3829, 0.4433, 0.4646, 0.5355, 0.5911, 0.7242]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2452, 0.2593, 0.2792, 0.2853, 0.2994, 0.3103, 0.3274, 0.3828, 0.4027, 0.5774]),
           st.sampled_from([0.05865, 0.06828, 0.07849, 0.07953, 0.082, 0.0906, 0.093, 0.09782, 0.1031, 0.1142]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([12.46, 15.3, 15.53, 15.85, 16.25, 18.65, 18.94, 20.13, 20.55, 28.11]),
           st.sampled_from([15.05, 15.7, 16.67, 17.46, 20.31, 20.38, 20.83, 21.08, 21.53, 28.03]),
           st.sampled_from([93.63, 94.48, 97.26, 98.92, 100.2, 100.4, 114.2, 116.0, 138.9, 143.7]),
           st.sampled_from([556.7, 565.4, 728.2, 933.1, 1102.0, 1145.0, 1167.0, 1191.0, 1311.0, 1364.0]),
           st.sampled_from([0.08402, 0.09055, 0.0915, 0.09444, 0.09488, 0.09797, 0.1003, 0.1068, 0.1137, 0.115]),
           st.sampled_from([0.07862, 0.08597, 0.1283, 0.1298, 0.1304, 0.1305, 0.1572, 0.1642, 0.1893, 0.2576]),
           st.sampled_from([0.03649, 0.09061, 0.09271, 0.09447, 0.1128, 0.1457, 0.1508, 0.169, 0.231, 0.2439]),
           st.floats(min_value=0.054101, max_value=0.08352, exclude_min=True, allow_nan=False),
           st.sampled_from([0.159, 0.1594, 0.1647, 0.1726, 0.1776, 0.1794, 0.1867, 0.2157, 0.2235, 0.304]),
           st.sampled_from([0.05294, 0.05325, 0.05443, 0.05667, 0.06213, 0.06464, 0.0654, 0.07077, 0.07325, 0.07682]),
           st.sampled_from([0.231, 0.3414, 0.37, 0.4743, 0.6137, 0.6191, 0.6422, 0.6874, 0.9317, 1.058]),
           st.sampled_from([0.6342, 0.6633, 0.8509, 0.8733, 1.027, 1.147, 1.169, 1.352, 1.56, 3.12]),
           st.sampled_from([2.587, 2.735, 2.974, 3.061, 3.564, 3.999, 4.115, 4.119, 4.206, 7.733]),
           st.sampled_from([22.18, 47.14, 51.22, 66.91, 72.44, 74.08, 80.6, 87.17, 94.03, 95.77]),
           st.sampled_from([0.00335, 0.004428, 0.004989, 0.005345, 0.005769, 0.006001, 0.00677, 0.006985, 0.007026, 0.0119]),
           st.sampled_from([0.01484, 0.02015, 0.02791, 0.03055, 0.03414, 0.03633, 0.03845, 0.04243, 0.04904, 0.04954]),
           st.sampled_from([0.01998, 0.02806, 0.03185, 0.03576, 0.05042, 0.05321, 0.06072, 0.07117, 0.07649, 0.1278]),
           st.sampled_from([0.01241, 0.01352, 0.01392, 0.01448, 0.01851, 0.01864, 0.01883, 0.0206, 0.02127, 0.02234]),
           st.sampled_from([0.01029, 0.0122, 0.01451, 0.0152, 0.01602, 0.01738, 0.02045, 0.02337, 0.03151, 0.03156]),
           st.sampled_from([0.001575, 0.001578, 0.002205, 0.002336, 0.002686, 0.00304, 0.003042, 0.00313, 0.00352, 0.009208]),
           st.floats(min_value=15.022, max_value=16.7949, allow_nan=False),
           st.floats(min_value=25.672, max_value=30.445, exclude_min=True, allow_nan=False),
           st.sampled_from([91.76, 92.04, 97.65, 108.8, 113.2, 114.6, 124.3, 151.6, 155.3, 162.3]),
           st.sampled_from([567.7, 697.7, 876.5, 888.3, 907.2, 1030.0, 1589.0, 1670.0, 1938.0, 2089.0]),
           st.sampled_from([0.128, 0.1281, 0.1342, 0.1426, 0.1495, 0.1498, 0.1515, 0.1517, 0.1665, 0.1878]),
           st.sampled_from([0.1486, 0.1845, 0.2297, 0.2311, 0.2884, 0.3846, 0.4061, 0.5249, 0.5717, 0.8681]),
           st.sampled_from([0.2606, 0.3194, 0.3272, 0.3728, 0.3912, 0.5936, 0.6181, 0.6872, 0.6956, 0.9034]),
           st.floats(min_value=0.135802, max_value=0.166841, exclude_min=True, allow_nan=False),
           st.sampled_from([0.2355, 0.248, 0.2551, 0.2807, 0.2829, 0.2853, 0.2972, 0.3151, 0.359, 0.427]),
           st.sampled_from([0.06818, 0.07427, 0.07873, 0.08465, 0.08666, 0.09061, 0.09618, 0.1013, 0.105, 0.1065]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([9.606, 11.6, 12.05, 12.34, 12.56, 13.45, 13.7, 13.74, 14.8, 16.3]),
           st.sampled_from([12.39, 13.29, 14.71, 15.45, 16.34, 17.26, 19.04, 21.6, 29.29, 29.37]),
           st.sampled_from([65.31, 73.28, 77.88, 78.61, 78.85, 79.47, 81.09, 85.89, 94.15, 94.57]),
           st.sampled_from([273.9, 373.2, 378.4, 402.0, 409.7, 470.9, 491.9, 585.9, 640.7, 658.8]),
           st.sampled_from([0.07937, 0.08043, 0.08814, 0.08915, 0.09492, 0.09578, 0.1028, 0.1088, 0.1092, 0.1255]),
           st.sampled_from([0.06376, 0.06807, 0.06877, 0.08066, 0.08502, 0.09486, 0.09713, 0.1011, 0.1483, 0.1836]),
           st.sampled_from([0.0, 0.01462, 0.01993, 0.02031, 0.03888, 0.05077, 0.1112, 0.145, 0.1975, 0.4108]),
           st.sampled_from([0.005159, 0.01502, 0.01863, 0.01939, 0.02008, 0.02257, 0.02331, 0.02377, 0.034, 0.07798]),
           st.sampled_from([0.106, 0.1305, 0.1422, 0.1464, 0.1466, 0.1598, 0.1739, 0.178, 0.2217, 0.2378]),
           st.sampled_from([0.0566, 0.05667, 0.05708, 0.05865, 0.06009, 0.06113, 0.06148, 0.06401, 0.06493, 0.06766]),
           st.sampled_from([0.1267, 0.2535, 0.272, 0.3163, 0.3446, 0.346, 0.3478, 0.3628, 0.4165, 0.4489]),
           st.sampled_from([0.3981, 0.489, 0.6612, 0.7615, 0.8732, 0.9227, 1.434, 1.478, 1.571, 1.974]),
           st.sampled_from([1.116, 1.204, 1.359, 1.466, 1.567, 1.628, 1.648, 2.183, 2.279, 3.018]),
           st.sampled_from([6.802, 8.955, 13.86, 15.5, 16.04, 18.32, 22.45, 24.62, 30.19, 39.43]),
           st.sampled_from([0.004291, 0.004599, 0.004775, 0.005169, 0.005371, 0.005518, 0.007278, 0.007334, 0.009058, 0.009702]),
           st.sampled_from([0.01171, 0.01491, 0.01755, 0.02042, 0.02196, 0.02305, 0.02456, 0.03932, 0.04653, 0.05156]),
           st.floats(min_value=0.026972, max_value=0.033714, allow_nan=False),
           st.sampled_from([0.002941, 0.007527, 0.00867, 0.008691, 0.01007, 0.01022, 0.01103, 0.01398, 0.0191, 0.05279]),
           st.sampled_from([0.01391, 0.01414, 0.01501, 0.01719, 0.01748, 0.01872, 0.02087, 0.02152, 0.0297, 0.03433]),
           st.sampled_from([0.00136, 0.001952, 0.002273, 0.002379, 0.002425, 0.002668, 0.00322, 0.003299, 0.003479, 0.008133]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.331, max_value=19.908, allow_nan=False),
           st.sampled_from([66.5, 76.53, 79.29, 81.6, 86.54, 87.4, 87.82, 89.27, 101.1, 102.5]),
           st.sampled_from([328.1, 364.2, 515.3, 529.9, 567.9, 611.1, 653.3, 782.1, 828.5, 947.9]),
           st.sampled_from([0.08774, 0.08864, 0.09293, 0.1005, 0.1282, 0.1303, 0.1341, 0.1384, 0.1413, 0.1475]),
           st.sampled_from([0.06477, 0.09726, 0.1232, 0.1257, 0.131, 0.1667, 0.2256, 0.2426, 0.3253, 0.4848]),
           st.sampled_from([0.02533, 0.05524, 0.07987, 0.1256, 0.1456, 0.1546, 0.1956, 0.305, 0.366, 0.4896]),
           st.sampled_from([0.02564, 0.0569, 0.06384, 0.07911, 0.07963, 0.08312, 0.08512, 0.0875, 0.09744, 0.1374]),
           st.sampled_from([0.2191, 0.2267, 0.2376, 0.2475, 0.2488, 0.2762, 0.2973, 0.3151, 0.3196, 0.32]),
           st.sampled_from([0.0612, 0.06623, 0.06783, 0.06956, 0.07087, 0.07582, 0.07623, 0.08181, 0.0896, 0.1017]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.84, 12.68, 15.32, 15.46, 16.03, 16.11, 17.42, 17.68, 18.66, 21.16]),
           st.sampled_from([16.15, 19.65, 20.26, 21.56, 22.15, 23.06, 23.98, 24.59, 25.42, 27.54]),
           st.sampled_from([102.1, 102.7, 120.2, 120.8, 127.9, 130.7, 141.3, 142.0, 153.5, 174.2]),
           st.sampled_from([386.1, 584.1, 610.7, 748.9, 809.8, 869.5, 955.1, 1006.0, 1007.0, 1261.0]),
           st.sampled_from([0.08923, 0.09081, 0.09116, 0.0943, 0.09874, 0.1015, 0.1131, 0.1133, 0.1172, 0.1447]),
           st.sampled_from([0.0629, 0.1248, 0.1262, 0.1279, 0.131, 0.1348, 0.1365, 0.1402, 0.1739, 0.2236]),
           st.sampled_from([0.05285, 0.0755, 0.09769, 0.09789, 0.1044, 0.1114, 0.1218, 0.1659, 0.1784, 0.1811]),
           st.sampled_from([0.02031, 0.05259, 0.05669, 0.06772, 0.08123, 0.08353, 0.08886, 0.09702, 0.1002, 0.141]),
           st.sampled_from([0.1467, 0.1741, 0.1801, 0.1807, 0.1834, 0.1893, 0.1943, 0.1998, 0.231, 0.2569]),
           st.sampled_from([0.05491, 0.05525, 0.05727, 0.06069, 0.06183, 0.06194, 0.06287, 0.06464, 0.06532, 0.07692]),
           st.sampled_from([0.2796, 0.4697, 0.524, 0.5296, 0.5659, 0.5702, 0.6643, 0.7572, 0.8361, 1.111]),
           st.sampled_from([0.8568, 0.8737, 0.9197, 0.976, 1.001, 1.078, 1.216, 1.331, 1.41, 1.506]),
           st.sampled_from([1.534, 2.061, 3.18, 3.528, 3.598, 4.206, 4.782, 4.906, 7.276, 8.589]),
           st.sampled_from([32.52, 45.4, 48.9, 60.41, 67.66, 83.5, 103.6, 106.0, 115.2, 139.9]),
           st.sampled_from([0.004029, 0.004253, 0.004314, 0.005532, 0.005726, 0.005872, 0.006455, 0.007571, 0.008005, 0.00911]),
           st.sampled_from([0.01162, 0.01906, 0.02219, 0.02239, 0.02895, 0.03169, 0.0371, 0.05121, 0.05693, 0.06213]),
           st.floats(min_value=0.033717, max_value=0.106173, exclude_min=True, allow_nan=False),
           st.sampled_from([0.006009, 0.01209, 0.01232, 0.01297, 0.01604, 0.01742, 0.01843, 0.01885, 0.02593, 0.02624]),
           st.sampled_from([0.01275, 0.01454, 0.01498, 0.01522, 0.01964, 0.02045, 0.02175, 0.02193, 0.03056, 0.03756]),
           st.sampled_from([0.001087, 0.001629, 0.002179, 0.002689, 0.003131, 0.003446, 0.003749, 0.003755, 0.005099, 0.005295]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=18.331, max_value=19.908, allow_nan=False),
           st.sampled_from([123.4, 123.5, 129.0, 132.7, 150.1, 150.6, 153.9, 168.2, 170.1, 178.6]),
           st.sampled_from([514.0, 762.4, 876.5, 928.8, 1227.0, 1315.0, 1344.0, 1417.0, 1603.0, 1681.0]),
           st.sampled_from([0.1223, 0.1385, 0.1431, 0.1446, 0.15, 0.1515, 0.1528, 0.1585, 0.1651, 0.1701]),
           st.sampled_from([0.2089, 0.2947, 0.3135, 0.3235, 0.3309, 0.3913, 0.5955, 0.611, 0.6643, 0.709]),
           st.sampled_from([0.2264, 0.2644, 0.2802, 0.3327, 0.3597, 0.3639, 0.3786, 0.4399, 0.583, 0.6956]),
           st.sampled_from([0.1185, 0.1418, 0.1521, 0.1528, 0.1932, 0.1956, 0.2493, 0.2508, 0.2625, 0.2756]),
           st.sampled_from([0.248, 0.2749, 0.275, 0.3019, 0.3068, 0.3215, 0.3444, 0.3672, 0.3792, 0.4824]),
           st.sampled_from([0.07397, 0.07568, 0.07918, 0.08067, 0.08503, 0.0895, 0.08999, 0.09772, 0.09789, 0.1233]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([8.726, 9.042, 9.295, 12.43, 13.11, 13.45, 13.56, 13.62, 14.87, 16.14]),
           st.sampled_from([12.17, 13.04, 14.09, 15.11, 15.39, 16.17, 17.27, 19.29, 19.54, 22.22]),
           st.sampled_from([71.49, 72.17, 73.88, 81.37, 84.1, 85.48, 85.89, 96.22, 96.39, 98.17]),
           st.sampled_from([288.1, 324.9, 389.4, 442.5, 485.6, 541.8, 561.3, 584.8, 644.2, 685.9]),
           st.sampled_from([0.07969, 0.08313, 0.08477, 0.08675, 0.08743, 0.09309, 0.1031, 0.1037, 0.1049, 0.1068]),
           st.sampled_from([0.02344, 0.03515, 0.05139, 0.05361, 0.05824, 0.06373, 0.08549, 0.09752, 0.1209, 0.1223]),
           st.sampled_from([0.00309, 0.01462, 0.01947, 0.02045, 0.0371, 0.06574, 0.0683, 0.07721, 0.09263, 0.09657]),
           st.sampled_from([0.007875, 0.01499, 0.01571, 0.01654, 0.02036, 0.02594, 0.02799, 0.02932, 0.03078, 0.03483]),
           st.sampled_from([0.135, 0.1449, 0.1496, 0.1614, 0.162, 0.1709, 0.178, 0.2013, 0.2082, 0.2116]),
           st.sampled_from([0.05355, 0.05894, 0.06066, 0.06147, 0.06235, 0.06471, 0.06582, 0.06714, 0.06766, 0.07102]),
           st.sampled_from([0.1302, 0.1931, 0.2212, 0.2338, 0.2699, 0.2747, 0.3342, 0.3344, 0.335, 0.4866]),
           st.sampled_from([0.7395, 0.7886, 0.8745, 1.268, 1.28, 1.376, 1.44, 1.601, 1.621, 1.924]),
           st.sampled_from([0.8484, 0.9857, 1.204, 1.354, 1.714, 1.976, 2.275, 2.346, 2.497, 2.561]),
           st.sampled_from([12.33, 15.5, 15.7, 17.81, 18.54, 19.15, 19.53, 20.39, 24.62, 28.62]),
           st.sampled_from([0.003632, 0.00508, 0.005415, 0.00638, 0.007017, 0.00854, 0.01134, 0.01291, 0.0134, 0.0138]),
           st.sampled_from([0.006134, 0.006364, 0.009238, 0.01567, 0.01966, 0.02417, 0.02845, 0.03498, 0.04638, 0.07446]),
           st.sampled_from([0.005812, 0.005832, 0.005949, 0.007276, 0.03047, 0.03214, 0.04017, 0.04156, 0.04763, 0.05915]),
           st.sampled_from([0.002404, 0.005398, 0.006897, 0.008747, 0.009199, 0.009615, 0.01007, 0.01398, 0.01633, 0.03487]),
           st.sampled_from([0.01416, 0.01561, 0.01708, 0.01715, 0.01799, 0.02079, 0.02187, 0.02625, 0.02882, 0.04077]),
           st.sampled_from([0.001463, 0.001588, 0.001773, 0.002128, 0.002133, 0.002496, 0.002668, 0.004672, 0.004738, 0.009627]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.911, max_value=25.836, exclude_min=True, allow_nan=False),
           st.sampled_from([70.88, 71.11, 72.22, 72.42, 76.53, 85.09, 87.16, 88.91, 97.9, 103.1]),
           st.sampled_from([240.1, 314.9, 366.3, 384.9, 410.4, 436.6, 543.9, 611.1, 867.1, 1009.0]),
           st.floats(min_value=0.084617, max_value=0.087978, allow_nan=False),
           st.sampled_from([0.09515, 0.1105, 0.1256, 0.1477, 0.1808, 0.1854, 0.1979, 0.2521, 0.2878, 0.3842]),
           st.sampled_from([0.007977, 0.03619, 0.04384, 0.05285, 0.1226, 0.1449, 0.1521, 0.1544, 0.1804, 0.7727]),
           st.sampled_from([0.008772, 0.02796, 0.05356, 0.0578, 0.05802, 0.05813, 0.07958, 0.09391, 0.1416, 0.1555]),
           st.sampled_from([0.189, 0.2249, 0.2321, 0.2345, 0.2506, 0.2815, 0.2884, 0.3038, 0.3075, 0.3124]),
           st.sampled_from([0.06386, 0.06428, 0.06958, 0.0738, 0.08096, 0.08385, 0.08468, 0.08982, 0.09218, 0.09638]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.81, 15.05, 15.66, 16.26, 17.3, 18.61, 20.47, 20.94, 24.25, 25.73]),
           st.sampled_from([16.67, 17.12, 21.08, 21.72, 21.91, 23.06, 25.56, 25.74, 26.67, 27.06]),
           st.sampled_from([77.58, 78.99, 85.26, 91.12, 102.9, 109.0, 121.1, 127.7, 131.2, 134.8]),
           st.sampled_from([477.1, 682.5, 705.6, 710.6, 803.1, 1068.0, 1167.0, 1308.0, 1682.0, 1841.0]),
           st.floats(min_value=0.083918, max_value=0.091739, allow_nan=False),
           st.sampled_from([0.08424, 0.111, 0.1141, 0.1275, 0.1336, 0.1954, 0.2236, 0.2413, 0.2768, 0.2832]),
           st.sampled_from([0.09789, 0.09875, 0.1657, 0.1676, 0.1692, 0.1695, 0.2071, 0.2197, 0.2712, 0.3189]),
           st.sampled_from([0.03515, 0.04938, 0.07041, 0.08465, 0.08773, 0.09113, 0.09498, 0.1062, 0.1471, 0.1562]),
           st.sampled_from([0.1554, 0.1609, 0.1727, 0.1846, 0.1853, 0.1974, 0.2202, 0.2251, 0.235, 0.2398]),
           st.sampled_from([0.05025, 0.05557, 0.05587, 0.05796, 0.05966, 0.06149, 0.06325, 0.06566, 0.06898, 0.07325]),
           st.sampled_from([0.2298, 0.3197, 0.4204, 0.4332, 0.5079, 0.5835, 0.6242, 0.6643, 0.7456, 0.8601]),
           st.sampled_from([0.8509, 1.017, 1.027, 1.045, 1.169, 1.194, 1.281, 1.885, 2.11, 3.12]),
           st.sampled_from([1.457, 1.719, 2.466, 2.916, 2.927, 3.043, 4.037, 4.293, 4.554, 6.311]),
           st.sampled_from([22.18, 24.19, 31.59, 41.0, 44.91, 58.53, 69.47, 69.65, 97.07, 138.5]),
           st.sampled_from([0.004253, 0.004426, 0.004428, 0.004714, 0.005038, 0.005072, 0.005654, 0.005878, 0.007231, 0.0119]),
           st.sampled_from([0.01174, 0.01906, 0.02648, 0.03203, 0.03252, 0.03482, 0.03889, 0.04006, 0.04759, 0.0547]),
           st.sampled_from([0.0139, 0.02572, 0.02626, 0.02791, 0.02855, 0.03055, 0.03715, 0.04062, 0.04741, 0.04983]),
           st.sampled_from([0.0111, 0.01143, 0.01184, 0.01232, 0.01361, 0.01458, 0.01459, 0.01623, 0.01746, 0.02536]),
           st.sampled_from([0.01069, 0.01223, 0.01263, 0.0152, 0.01602, 0.01789, 0.02008, 0.02105, 0.0371, 0.05113]),
           st.sampled_from([0.001578, 0.001754, 0.001997, 0.002205, 0.002719, 0.003317, 0.004306, 0.005195, 0.009875, 0.01008]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.911, max_value=25.836, exclude_min=True, allow_nan=False),
           st.sampled_from([107.3, 111.8, 113.1, 123.4, 124.1, 136.5, 149.3, 160.2, 162.3, 251.2]),
           st.sampled_from([915.3, 959.5, 1236.0, 1359.0, 1421.0, 1657.0, 2360.0, 2477.0, 2642.0, 3143.0]),
           st.floats(min_value=0.087981, max_value=0.114904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.205, 0.2053, 0.2534, 0.2536, 0.3559, 0.3682, 0.4503, 0.4706, 0.5634, 0.8681]),
           st.floats(min_value=0.1438, max_value=0.179749, allow_nan=False),
           st.sampled_from([0.02899, 0.1374, 0.1515, 0.152, 0.1864, 0.1956, 0.1984, 0.2027, 0.2543, 0.2701]),
           st.sampled_from([0.2818, 0.2928, 0.2929, 0.3013, 0.3032, 0.3187, 0.3271, 0.3651, 0.4245, 0.467]),
           st.sampled_from([0.05525, 0.06836, 0.06846, 0.07127, 0.07425, 0.07944, 0.07999, 0.08019, 0.08067, 0.08574]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([11.61, 12.76, 12.87, 13.17, 13.45, 13.87, 13.94, 14.5, 14.62, 14.8]),
           st.sampled_from([13.98, 14.36, 15.62, 15.92, 16.35, 17.02, 19.07, 24.49, 28.21, 28.92]),
           st.sampled_from([60.21, 70.21, 73.06, 73.81, 80.45, 84.1, 86.49, 88.99, 95.88, 96.39]),
           st.sampled_from([250.5, 289.9, 324.9, 373.2, 378.4, 384.6, 431.9, 480.1, 514.0, 571.1]),
           st.floats(min_value=0.091742, max_value=0.106073, exclude_min=True, allow_nan=False),
           st.sampled_from([0.03766, 0.03834, 0.05055, 0.05251, 0.07079, 0.08333, 0.09009, 0.09228, 0.1013, 0.1225]),
           st.sampled_from([0.02353, 0.02367, 0.03102, 0.03512, 0.03592, 0.038, 0.03987, 0.04279, 0.0905, 0.09293]),
           st.sampled_from([0.004419, 0.005159, 0.0209, 0.02168, 0.02179, 0.03562, 0.03783, 0.04497, 0.04951, 0.07798]),
           st.sampled_from([0.1215, 0.1713, 0.1776, 0.1859, 0.188, 0.1883, 0.2019, 0.233, 0.2403, 0.2595]),
           st.sampled_from([0.05449, 0.05667, 0.05703, 0.05746, 0.05865, 0.06409, 0.06562, 0.06677, 0.06758, 0.08116]),
           st.sampled_from([0.1833, 0.2136, 0.2182, 0.2204, 0.2315, 0.2467, 0.2841, 0.2929, 0.3975, 0.7311]),
           st.sampled_from([0.6221, 0.9861, 1.13, 1.139, 1.326, 1.39, 1.479, 1.647, 1.705, 3.896]),
           st.sampled_from([0.9219, 1.09, 1.146, 1.171, 1.314, 1.359, 1.592, 1.628, 1.67, 1.787]),
           st.sampled_from([8.322, 15.75, 19.15, 20.21, 21.83, 29.06, 29.84, 30.19, 31.16, 39.84]),
           st.sampled_from([0.004352, 0.004928, 0.005217, 0.005682, 0.008146, 0.008902, 0.009006, 0.009845, 0.01291, 0.0138]),
           st.sampled_from([0.01171, 0.0177, 0.02025, 0.02314, 0.02652, 0.02839, 0.03378, 0.04235, 0.04741, 0.07471]),
           st.sampled_from([0.002817, 0.004826, 0.009127, 0.009959, 0.01328, 0.02662, 0.04757, 0.04804, 0.05915, 0.06271]),
           st.sampled_from([0.005077, 0.005917, 0.00624, 0.006897, 0.007016, 0.007408, 0.01368, 0.01626, 0.02292, 0.05279]),
           st.sampled_from([0.01069, 0.01254, 0.01447, 0.01713, 0.01748, 0.01872, 0.02637, 0.02669, 0.02671, 0.03373]),
           st.sampled_from([0.001126, 0.001708, 0.001956, 0.00206, 0.002222, 0.00236, 0.003087, 0.003114, 0.004261, 0.01045]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.911, max_value=25.836, exclude_min=True, allow_nan=False),
           st.sampled_from([56.65, 59.9, 72.62, 86.92, 88.14, 89.02, 98.4, 99.17, 110.2, 127.1]),
           st.sampled_from([362.7, 475.8, 476.5, 527.8, 559.5, 580.6, 600.5, 701.9, 709.0, 773.4]),
           st.floats(min_value=0.087981, max_value=0.114904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.07057, 0.08971, 0.09052, 0.1232, 0.1415, 0.1773, 0.2031, 0.2196, 0.2429, 0.2878]),
           st.floats(min_value=0.1438, max_value=0.179749, allow_nan=False),
           st.sampled_from([0.03264, 0.04464, 0.08211, 0.08235, 0.08263, 0.08958, 0.09391, 0.1092, 0.1318, 0.1414]),
           st.sampled_from([0.2121, 0.2213, 0.2254, 0.2646, 0.2685, 0.2744, 0.2884, 0.2973, 0.3207, 0.4128]),
           st.sampled_from([0.0658, 0.06596, 0.06772, 0.07123, 0.0723, 0.07626, 0.07961, 0.09206, 0.09241, 0.1082]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([13.4, 14.71, 15.75, 17.08, 17.54, 18.03, 18.45, 18.63, 19.21, 23.21]),
           st.sampled_from([15.56, 17.25, 18.87, 19.07, 19.33, 20.26, 22.29, 25.56, 27.06, 27.15]),
           st.sampled_from([81.72, 92.87, 110.1, 111.2, 128.3, 132.5, 133.6, 133.7, 141.3, 144.4]),
           st.sampled_from([565.4, 572.6, 599.5, 644.8, 671.4, 682.5, 813.7, 886.3, 899.3, 981.6]),
           st.sampled_from([0.07497, 0.0909, 0.0926, 0.09742, 0.1008, 0.1018, 0.1024, 0.1069, 0.1165, 0.1286]),
           st.sampled_from([0.08468, 0.1022, 0.1143, 0.1188, 0.1198, 0.1204, 0.1289, 0.1306, 0.131, 0.1739]),
           st.sampled_from([0.05253, 0.1097, 0.1126, 0.1204, 0.1379, 0.1463, 0.169, 0.1859, 0.231, 0.2439]),
           st.sampled_from([0.04938, 0.06018, 0.0795, 0.08624, 0.08653, 0.09176, 0.1121, 0.1286, 0.1377, 0.1474]),
           st.sampled_from([0.1547, 0.155, 0.1669, 0.1812, 0.1823, 0.1829, 0.1956, 0.2092, 0.2132, 0.2678]),
           st.sampled_from([0.05325, 0.05648, 0.05883, 0.05892, 0.06049, 0.0614, 0.06216, 0.06277, 0.06382, 0.06879]),
           st.sampled_from([0.2121, 0.2577, 0.2976, 0.2986, 0.3414, 0.4357, 0.5659, 0.7128, 1.214, 1.291]),
           st.sampled_from([0.5679, 0.828, 0.8509, 1.045, 1.152, 1.457, 1.667, 1.809, 2.91, 3.12]),
           st.sampled_from([2.563, 2.819, 3.061, 3.123, 3.598, 3.705, 4.206, 4.312, 5.203, 6.051]),
           st.sampled_from([23.35, 31.72, 33.27, 71.0, 71.56, 86.22, 102.6, 103.9, 104.9, 155.8]),
           st.sampled_from([0.005288, 0.005596, 0.005627, 0.006292, 0.006494, 0.006717, 0.007026, 0.00794, 0.009327, 0.03113]),
           st.sampled_from([0.01094, 0.01114, 0.01578, 0.01929, 0.02065, 0.02277, 0.02772, 0.02833, 0.03633, 0.04844]),
           st.sampled_from([0.01246, 0.02791, 0.04029, 0.04644, 0.0473, 0.05688, 0.06329, 0.06591, 0.07649, 0.1091]),
           st.sampled_from([0.008522, 0.009863, 0.01267, 0.01276, 0.01342, 0.01559, 0.01587, 0.01604, 0.01843, 0.02624]),
           st.sampled_from([0.01172, 0.01394, 0.0146, 0.01875, 0.01925, 0.0203, 0.02045, 0.02689, 0.02816, 0.03151]),
           st.sampled_from([0.001087, 0.002336, 0.002461, 0.002575, 0.003187, 0.003711, 0.004045, 0.004476, 0.005466, 0.006213]),
           st.floats(min_value=16.7952, max_value=20.6441, exclude_min=True, allow_nan=False),
           st.floats(min_value=19.911, max_value=25.836, exclude_min=True, allow_nan=False),
           st.sampled_from([98.87, 105.5, 113.1, 114.6, 137.9, 146.0, 158.3, 176.5, 184.2, 220.8]),
           st.sampled_from([508.1, 698.8, 827.2, 876.5, 915.0, 967.0, 1138.0, 1417.0, 1660.0, 2010.0]),
           st.floats(min_value=0.087981, max_value=0.114904, exclude_min=True, allow_nan=False),
           st.sampled_from([0.09866, 0.1866, 0.2336, 0.2394, 0.2444, 0.2678, 0.3904, 0.4099, 0.4186, 0.6997]),
           st.floats(min_value=0.179752, max_value=0.394201, exclude_min=True, allow_nan=False),
           st.sampled_from([0.02899, 0.1225, 0.1515, 0.1573, 0.1659, 0.1712, 0.1986, 0.206, 0.255, 0.2688]),
           st.sampled_from([0.2609, 0.2623, 0.2641, 0.2689, 0.2812, 0.2908, 0.3055, 0.3792, 0.427, 0.467]),
           st.sampled_from([0.06915, 0.07425, 0.08368, 0.08633, 0.1005, 0.1019, 0.1109, 0.1189, 0.1191, 0.1405]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
