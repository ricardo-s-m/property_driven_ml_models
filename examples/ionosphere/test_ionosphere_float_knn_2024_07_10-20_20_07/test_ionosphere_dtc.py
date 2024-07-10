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
    request.cls.data['n_test'] = 23
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    experiment_data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'test_ionosphere_dtc_experiment_data.json')
    yield experiment_data_path
    with open(experiment_data_path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestIonosphereProperty:

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([-0.64286, 0.08333, 0.1859, 0.39179, 0.39286, 0.72727, 0.84783, 0.91353, 0.94531, 0.94598]),
           st.sampled_from([-0.86701, -0.45161, -0.16667, -0.12727, -0.05529, -0.00838, 0.0, 0.00705, 0.01437, 0.02461]),
           st.floats(min_value=-1.0, max_value=0.041438, allow_nan=False),
           st.sampled_from([-1.0, -0.36156, 0.00838, 0.04861, 0.08069, 0.1, 0.12159, 0.15033, 0.5125, 0.80521]),
           st.sampled_from([-0.75693, -0.39466, -0.2962, -0.0352, 0.26459, 0.46643, 0.76001, 0.88444, 0.97545, 0.98019]),
           st.sampled_from([-0.39896, -0.11111, -0.09473, -0.04444, -0.02, -0.01639, 0.04348, 0.07143, 0.24125, 0.35639]),
           st.sampled_from([0.07979, 0.10991, 0.1147, 0.23346, 0.30682, 0.61745, 0.62745, 0.66667, 0.80648, 0.99709]),
           st.sampled_from([-1.0, -0.40446, -0.10984, -0.10236, -0.06811, -0.05909, 0.21053, 0.28788, 0.71242, 0.95041]),
           st.sampled_from([-1.0, -0.0485, 0.01676, 0.04517, 0.07595, 0.20743, 0.38359, 0.43, 0.43685, 0.89514]),
           st.sampled_from([-0.94853, -0.89098, -0.41368, -0.3871, -0.25712, -0.03279, 0.02575, 0.12808, 0.25292, 0.7619]),
           st.sampled_from([-1.0, -0.71875, -0.6723, -0.07576, -0.05921, 0.11538, 0.16667, 0.33333, 0.81809, 1.0]),
           st.sampled_from([-0.79313, -0.59043, -0.03103, 0.18227, 0.27153, 0.30729, 0.36965, 0.51894, 0.5364, 0.63966]),
           st.sampled_from([-0.61354, -0.20019, -0.095, -0.04572, 0.0625, 0.14645, 0.68852, 0.7592, 0.88683, 0.91962]),
           st.sampled_from([-0.67708, -0.1554, -0.12, -0.09804, -0.03403, -0.00391, 0.14137, 0.14516, 0.35507, 1.0]),
           st.sampled_from([-0.55711, -0.00343, 0.2, 0.21032, 0.29062, 0.50455, 0.66791, 0.81395, 0.8227, 0.85701]),
           st.sampled_from([-0.83297, -0.62237, -0.35985, -0.29, -0.22727, -0.14754, -0.11351, -0.01818, 0.41168, 0.82813]),
           st.sampled_from([-0.34773, -0.01712, 0.18993, 0.22807, 0.33746, 0.36482, 0.51979, 0.66818, 0.77979, 0.92188]),
           st.sampled_from([-0.60294, -0.54467, -0.54248, -0.43025, -0.42292, -0.11567, 0.0, 0.18135, 0.36353, 0.78036]),
           st.sampled_from([-0.75, -0.5183, -0.31402, -0.06959, 0.04325, 0.04372, 0.06897, 0.58824, 0.65574, 0.68839]),
           st.sampled_from([-0.453, -0.21081, -0.08, -0.04, -0.03393, 0.03669, 0.05078, 0.16371, 0.85268, 1.0]),
           st.sampled_from([-0.84286, -0.63777, -0.2, 0.01834, 0.02793, 0.29908, 0.60138, 0.61818, 0.8667, 0.8913]),
           st.sampled_from([-0.94118, -0.29508, -0.27083, -0.02282, -0.01149, -0.00559, 0.00102, 0.20301, 0.34766, 0.80934]),
           st.sampled_from([-0.71528, -0.18856, -0.02862, -0.02494, 0.03513, 0.15103, 0.24242, 0.29167, 0.32727, 0.5375]),
           st.sampled_from([-1.0, -0.26569, -0.24, -0.17012, -0.01535, -0.01326, 0.06639, 0.17803, 0.5, 0.90695]),
           st.sampled_from([-0.04314, 0.00019, 0.04167, 0.33333, 0.39752, 0.41098, 0.48867, 0.52385, 0.70882, 0.93124]),
           st.sampled_from([-0.57092, -0.23864, -0.0614, -0.03578, -0.00043, 0.1463, 0.15942, 0.63492, 0.75, 0.88248]),
           st.sampled_from([-0.2234, -0.1517, -0.14786, -0.00403, 0.02586, 0.32787, 0.43182, 0.43481, 0.56863, 0.62958]),
           st.sampled_from([-1.0, -0.48485, -0.31373, -0.11593, -0.01947, 0.15436, 0.42677, 0.67708, 0.69401, 1.0]),
           st.sampled_from([-0.66667, -0.09339, -0.01693, -0.00761, 0.02564, 0.18182, 0.20614, 0.48927, 0.69792, 0.87757]),
           st.sampled_from([-0.87354, -0.7817, -0.3254, -0.07464, -0.06522, 0.0, 0.00108, 0.20175, 0.20477, 0.65268]),
           st.sampled_from([-0.23186, -0.19792, 0.00705, 0.16827, 0.25441, 0.2619, 0.40445, 0.47059, 0.47569, 1.0]),
           st.sampled_from([-1.0, -0.29091, -0.27451, -0.09556, -0.08978, 0.00325, 0.19551, 0.37409, 0.38065, 0.45745]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([-1.0, 0.10135, 0.1859, 0.39286, 0.42708, 0.62121, 0.71253, 0.72727, 0.84557, 0.84783]),
           st.sampled_from([-0.86701, -0.5421, -0.5, -0.16667, -0.12727, -0.08459, -0.07843, -0.00838, 0.15625, 0.40332]),
           st.floats(min_value=0.041441, max_value=0.231539, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.2381, -0.23067, -0.11949, 0.0, 0.03462, 0.12889, 0.15033, 0.26131, 0.26708, 0.3913]),
           st.sampled_from([-1.0, -0.32937, -0.17778, -0.07661, -0.0352, -0.01923, 0.08374, 0.33333, 0.50814, 0.57377]),
           st.sampled_from([-0.93597, -0.82456, -0.70984, -0.67321, -0.67273, -0.14375, -0.04444, -0.01997, 0.04348, 0.55735]),
           st.sampled_from([0.00559, 0.25795, 0.34649, 0.45238, 0.46458, 0.60586, 0.61745, 0.62745, 0.65574, 0.8401]),
           st.sampled_from([-0.23393, -0.22222, -0.11765, 0.0, 0.0119, 0.01639, 0.04576, 0.18719, 0.35, 0.95833]),
           st.sampled_from([-0.82143, -0.66886, 0.025, 0.20743, 0.43, 0.47744, 0.48182, 0.54478, 0.58958, 0.97311]),
           st.sampled_from([-0.94853, -0.41368, -0.3617, -0.33333, -0.16818, -0.07342, -0.06302, 0.07981, 0.64, 0.7619]),
           st.sampled_from([-0.71875, -0.26048, -0.02439, 0.11538, 0.23958, 0.34432, 0.61567, 0.88462, 0.95824, 1.0]),
           st.sampled_from([-0.85727, -0.8209, -0.79313, -0.69707, -0.25342, -0.11111, 0.02726, 0.08978, 0.16406, 0.45489]),
           st.sampled_from([-0.61354, -0.54852, -0.33656, -0.095, -0.04572, 0.08949, 0.12292, 0.68852, 0.83726, 0.91962]),
           st.sampled_from([-0.67708, -0.3466, -0.3125, -0.09804, -0.07143, -0.02614, 0.15452, 0.22957, 0.38602, 0.65565]),
           st.sampled_from([-0.04708, -0.03891, 0.0, 0.01108, 0.0197, 0.28301, 0.36585, 0.76929, 0.81395, 0.95452]),
           st.sampled_from([-0.71951, -0.4627, -0.45417, -0.3933, -0.35985, -0.15625, 0.0, 0.19444, 0.235, 0.51163]),
           st.sampled_from([-0.60131, -0.34187, 0.05058, 0.31755, 0.36482, 0.46642, 0.54762, 0.59167, 0.66818, 0.92188]),
           st.sampled_from([-0.80682, -0.42862, -0.375, -0.27461, -0.11567, -0.00423, 0.03406, 0.05172, 0.22115, 0.78036]),
           st.sampled_from([-0.25914, -0.04575, 0.0, 0.00048, 0.00759, 0.01838, 0.0214, 0.04325, 0.45238, 0.65574]),
           st.sampled_from([-0.19608, -0.09375, 0.01478, 0.04762, 0.24582, 0.30745, 0.38803, 0.63496, 0.85268, 1.0]),
           st.sampled_from([-0.73864, -0.6, -0.57576, -0.34615, -0.34128, -0.29825, -0.11976, 0.17021, 0.94118, 1.0]),
           st.sampled_from([-0.94118, -0.35734, -0.29508, -0.02282, 0.00019, 0.20301, 0.24936, 0.37361, 0.45522, 0.62349]),
           st.floats(min_value=-1.0, max_value=0.183739, allow_nan=False),
           st.sampled_from([-0.46875, -0.2549, -0.16667, -0.01326, -0.01269, 2e-05, 0.06639, 0.19405, 0.44828, 0.68]),
           st.sampled_from([0.00019, 0.03443, 0.08467, 0.24889, 0.3219, 0.48867, 0.49309, 0.56982, 0.70882, 0.74385]),
           st.sampled_from([-0.57092, -0.46809, -0.20325, -0.10897, -0.01128, -0.00838, 0.01373, 0.01724, 0.14673, 0.15942]),
           st.sampled_from([-0.27917, -0.22917, -0.2234, -0.1904, -0.14786, -0.12879, -0.05911, 0.03073, 0.125, 0.42273]),
           st.sampled_from([-0.48936, -0.38914, -0.27778, -0.23649, -0.01947, -0.00279, 0.06977, 0.10349, 0.18854, 0.69401]),
           st.sampled_from([-0.125, -0.03125, -0.02568, -0.00761, 0.04447, 0.12727, 0.18301, 0.35088, 0.38352, 0.74468]),
           st.sampled_from([-0.3254, -0.27083, -0.21569, -0.1875, 0.05419, 0.13521, 0.14643, 0.24086, 0.88428, 1.0]),
           st.sampled_from([-0.67553, -0.19792, -0.18494, -0.00526, -0.00039, 0.1167, 0.16827, 0.40445, 0.47059, 0.82895]),
           st.sampled_from([-0.9375, -0.65935, -0.48241, -0.08978, -0.00575, 0.0, 0.02381, 0.19551, 0.33611, 0.38065]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([0.49515, 0.51207, 0.64947, 0.79847, 0.83479, 0.86284, 0.92308, 0.9449, 0.96355, 0.97714]),
           st.sampled_from([-0.92453, -0.57224, -0.16316, -0.0373, -0.02712, -0.01604, 0.04167, 0.20408, 0.28082, 0.36724]),
           st.floats(min_value=0.041441, max_value=0.231539, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.14333, -0.11326, -0.02423, -0.01918, -0.01272, -0.00228, 0.01566, 0.10435, 0.24323, 0.74804]),
           st.sampled_from([-0.22222, 0.71771, 0.77123, 0.83904, 0.85096, 0.85705, 0.88783, 0.92046, 0.92442, 0.95177]),
           st.sampled_from([-0.39175, -0.01214, -0.00509, 0.01214, 0.19498, 0.29292, 0.47619, 0.48866, 0.88576, 0.9346]),
           st.sampled_from([-0.32104, 0.36667, 0.47894, 0.57282, 0.63636, 0.74313, 0.85996, 0.89179, 0.90169, 0.97026]),
           st.sampled_from([-0.09316, 0.0, 0.01456, 0.18261, 0.41252, 0.6452, 0.76527, 0.83548, 0.95193, 0.96534]),
           st.sampled_from([-0.29784, 0.0107, 0.13753, 0.31409, 0.33081, 0.52462, 0.52993, 0.60504, 0.82208, 0.95473]),
           st.sampled_from([-0.0778, -0.03689, -0.02809, -0.00054, 0.06678, 0.31152, 0.34826, 0.47904, 0.79886, 0.8323]),
           st.sampled_from([-0.62403, -0.51763, 0.45652, 0.57576, 0.6289, 0.71598, 0.75474, 0.8425, 0.87945, 0.95735]),
           st.sampled_from([0.00585, 0.04731, 0.08436, 0.10346, 0.38643, 0.40552, 0.42739, 0.46899, 0.62078, 0.77092]),
           st.sampled_from([-0.06827, -0.06333, 0.63636, 0.67849, 0.68798, 0.69737, 0.72322, 0.87161, 0.90782, 0.91714]),
           st.sampled_from([-0.24942, -0.22222, -0.15851, -0.0549, -0.03636, 0.00485, 0.02609, 0.37988, 0.54965, 0.91404]),
           st.sampled_from([-0.66455, -0.51587, 0.32432, 0.4375, 0.63291, 0.80681, 0.83719, 0.88809, 0.91381, 0.95745]),
           st.sampled_from([-0.29441, -0.25324, -0.24598, -0.24482, -0.00788, -0.00085, 0.01852, 0.09189, 0.46059, 0.82193]),
           st.sampled_from([-1.0, -0.90995, -0.0323, 0.48869, 0.5283, 0.58213, 0.79113, 0.86104, 0.86676, 0.87433]),
           st.sampled_from([-0.30942, -0.17813, -0.04662, -0.02845, -0.02358, 0.0339, 0.03939, 0.04511, 0.362, 0.70398]),
           st.sampled_from([-0.81056, -0.51778, -0.50251, -0.46635, 0.31667, 0.35294, 0.47541, 0.52494, 0.90909, 0.95016]),
           st.sampled_from([-0.47308, -0.19095, 0.04901, 0.07211, 0.18888, 0.20489, 0.67135, 0.68104, 0.73696, 0.76969]),
           st.sampled_from([-0.37671, -0.02876, 0.25, 0.51109, 0.5343, 0.68874, 0.7451, 0.75676, 0.79578, 0.82987]),
           st.sampled_from([-0.59661, -0.28656, -0.22786, -0.12062, -0.11111, -0.02685, 0.08247, 0.16667, 0.25316, 0.50428]),
           st.floats(min_value=0.183742, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.85703, -0.81634, -0.33145, -0.20851, -0.15094, -0.08341, 0.02376, 0.09756, 0.11299, 0.15957]),
           st.sampled_from([-1.0, -0.77097, -0.63038, -0.57587, -0.1695, 0.19304, 0.33401, 0.71698, 0.79058, 0.83796]),
           st.sampled_from([-0.60678, -0.05827, -0.02174, -0.01237, 0.04452, 0.06477, 0.08304, 0.13282, 0.14314, 0.69435]),
           st.floats(min_value=-1.0, max_value=0.767748, allow_nan=False),
           st.sampled_from([-0.75353, -0.68794, -0.60858, -0.28123, -0.18725, -0.17365, 0.09599, 0.16137, 0.17539, 0.27338]),
           st.sampled_from([-0.75321, -0.65754, -0.1737, 0.06927, 0.3867, 0.58137, 0.6015, 0.83051, 0.83777, 0.95434]),
           st.sampled_from([-0.42226, -0.30057, -0.01389, 0.04158, 0.13632, 0.1593, 0.36364, 0.43396, 0.54522, 0.76764]),
           st.sampled_from([-0.19312, 0.02249, 0.13994, 0.42223, 0.44767, 0.56167, 0.57418, 0.87403, 0.93617, 0.98674]),
           st.sampled_from([-0.57758, -0.56618, -0.453, -0.44222, -0.07329, -0.05402, -0.00296, 0.01214, 0.03446, 0.44534]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([-0.67935, -0.65625, -0.01864, 0.05866, 0.1859, 0.42708, 0.62121, 0.65909, 0.72727, 0.79157]),
           st.sampled_from([-0.93996, -0.62879, -0.14754, -0.12727, -0.09524, -0.0858, -0.05529, 0.10811, 0.4, 0.40332]),
           st.floats(min_value=0.041441, max_value=0.231539, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76087, -0.36156, -0.04119, 0.0, 0.04861, 0.08069, 0.21908, 0.2228, 0.51724, 0.80521]),
           st.sampled_from([-0.39466, -0.32937, -0.16628, -0.10868, -0.08961, 0.01997, 0.10526, 0.26459, 0.68182, 0.97545]),
           st.sampled_from([-1.0, -0.78502, -0.67321, -0.01639, -0.01563, 0.06874, 0.3175, 0.35639, 0.63317, 1.0]),
           st.sampled_from([0.01128, 0.1147, 0.23346, 0.30682, 0.35821, 0.39468, 0.61745, 0.66364, 0.80893, 0.87719]),
           st.sampled_from([-0.5, -0.11765, -0.10984, -0.10236, 0.04372, 0.1323, 0.18719, 0.21053, 0.31746, 0.47173]),
           st.sampled_from([-0.67285, 0.06404, 0.19455, 0.29091, 0.31081, 0.46053, 0.47744, 0.64394, 0.85246, 0.93243]),
           st.sampled_from([-1.0, -0.41368, -0.41119, -0.33333, -0.10951, -0.07342, -0.06609, -0.05455, 0.03876, 0.36194]),
           st.sampled_from([-0.6723, -0.05921, -0.03158, 0.01975, 0.0925, 0.34432, 0.47368, 0.58965, 0.66667, 0.88462]),
           st.sampled_from([-0.84848, -0.79313, -0.13725, 0.0431, 0.09559, 0.16406, 0.26042, 0.275, 0.29091, 1.0]),
           st.sampled_from([-0.54852, -0.34797, -0.30241, -0.1335, -0.095, -0.04572, 0.08949, 0.35088, 0.7592, 0.83726]),
           st.sampled_from([-0.34265, -0.23476, -0.1548, -0.03403, -0.03281, 0.07027, 0.14516, 0.35507, 0.65565, 0.6791]),
           st.sampled_from([-0.53788, -0.375, -0.0409, 0.16458, 0.18681, 0.37162, 0.47364, 0.66791, 0.72059, 0.95452]),
           st.sampled_from([-0.4627, -0.17863, -0.10196, -0.03947, -0.01818, 0.08046, 0.15189, 0.21818, 0.36965, 0.95122]),
           st.sampled_from([-0.12808, 0.05058, 0.10567, 0.3141, 0.33746, 0.4375, 0.45076, 0.54902, 0.63728, 0.8384]),
           st.sampled_from([-0.93599, -0.42292, -0.18333, -0.17176, -0.11567, 0.0, 0.03566, 0.11628, 0.18729, 0.22115]),
           st.sampled_from([-0.99219, -0.25, -0.19792, -0.13151, -0.04575, 0.01894, 0.02011, 0.24903, 0.58824, 0.6585]),
           st.sampled_from([-0.25, -0.03393, 0.00075, 0.02299, 0.03279, 0.5067, 0.63496, 0.68101, 0.74359, 0.9273]),
           st.sampled_from([-0.93359, -0.84286, -0.6, -0.57576, -0.47651, -3e-05, 0.01834, 0.02793, 0.16667, 1.0]),
           st.sampled_from([-0.35734, -0.29508, -0.2619, -0.23529, 0.00019, 0.00102, 0.16364, 0.23725, 0.26585, 0.37361]),
           st.floats(min_value=0.183742, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.53846, -0.46875, -0.43137, -0.40888, -0.01326, -0.00391, 0.16256, 0.23387, 0.2381, 0.68]),
           st.sampled_from([-0.61538, -0.34146, 0.00862, 0.01834, 0.25758, 0.28079, 0.52385, 0.56982, 0.93124, 1.0]),
           st.sampled_from([-0.76667, -0.25288, -0.18401, -0.08571, -0.0614, 0.0, 0.09176, 0.13758, 0.14673, 0.22]),
           st.floats(min_value=0.767751, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.75, -0.61393, -0.48485, -0.3625, -0.23649, -0.21901, -0.05455, -0.01947, 0.00713, 0.65625]),
           st.sampled_from([-0.27104, -0.21354, -0.02568, -0.01693, -0.00761, 0.04447, 0.06329, 0.09611, 0.18301, 0.74468]),
           st.sampled_from([-0.70076, -0.31228, -0.29915, -0.1875, 0.00015, 0.01149, 0.07064, 0.1286, 0.65268, 0.81007]),
           st.sampled_from([-0.18494, -0.13738, -0.04598, 0.00015, 0.04749, 0.2619, 0.40445, 0.56522, 0.81979, 0.90667]),
           st.sampled_from([-0.9375, -0.59867, -0.41667, -0.34375, -0.27451, 2e-05, 0.00325, 0.01997, 0.04586, 0.19551]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([0.31034, 0.46785, 0.62335, 0.74704, 0.75564, 0.76046, 0.82254, 0.94333, 0.96355, 0.99701]),
           st.sampled_from([-0.12095, -0.04388, -0.02811, -0.02259, -0.00857, 0.04974, 0.1085, 0.18923, 0.27475, 1.0]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.11313, 0.00481, 0.00485, 0.00843, 0.06158, 0.10278, 0.12197, 0.25191, 0.35543, 0.57816]),
           st.sampled_from([0.01961, 0.29921, 0.35916, 0.7, 0.7299, 0.77273, 0.81594, 0.83904, 0.96142, 0.98658]),
           st.floats(min_value=-1.0, max_value=-0.896692, allow_nan=False),
           st.sampled_from([-0.32104, -0.25131, 0.56687, 0.65077, 0.79015, 0.85746, 0.87778, 0.8798, 0.91176, 0.96974]),
           st.sampled_from([-0.35818, -0.33075, -0.14286, -0.01911, 0.00224, 0.01064, 0.47127, 0.72997, 0.76527, 0.92324]),
           st.sampled_from([-0.83152, 0.09831, 0.40476, 0.63621, 0.79271, 0.89391, 0.90023, 0.93844, 0.95031, 0.95455]),
           st.sampled_from([-0.09091, -0.07797, 0.02482, 0.03546, 0.06107, 0.08314, 0.11852, 0.14118, 0.53968, 0.85011]),
           st.sampled_from([-0.86622, -0.63714, -0.60316, -0.46177, -0.4542, 0.10107, 0.70716, 0.91161, 0.94681, 0.96186]),
           st.sampled_from([-0.50764, -0.26161, -0.12738, -0.12527, -0.07287, -0.04754, -0.04719, -0.01033, 0.10188, 0.15152]),
           st.sampled_from([-0.96986, -0.8296, -0.52735, 0.06438, 0.30459, 0.55311, 0.62936, 0.66517, 0.68798, 0.72322]),
           st.sampled_from([-0.49967, -0.24164, -0.06598, -0.02542, -0.00543, -0.00337, 0.42314, 0.47744, 0.5882, 0.66237]),
           st.sampled_from([-0.12611, 0.20269, 0.54237, 0.73242, 0.84628, 0.88809, 0.94217, 0.95465, 0.98305, 0.99528]),
           st.sampled_from([-0.6218, -0.32268, -0.05365, 0.01695, 0.06186, 0.07801, 0.11864, 0.19328, 0.31831, 0.50391]),
           st.sampled_from([-0.81641, -0.52145, -0.23529, 0.51247, 0.5283, 0.6351, 0.64719, 0.76771, 0.82222, 0.90343]),
           st.sampled_from([-0.26087, -0.10032, -0.08848, -0.02845, 0.01393, 0.01657, 0.03261, 0.40368, 0.5, 0.7314]),
           st.sampled_from([-0.73083, -0.50931, -0.32354, -0.28571, -0.10237, 0.39903, 0.75, 0.79193, 0.88065, 0.974]),
           st.sampled_from([-0.47153, -0.35519, -0.19512, -0.12126, -0.03377, -0.00377, 0.18182, 0.49508, 0.65924, 0.82162]),
           st.sampled_from([-0.43087, -0.14815, -0.01734, 0.18932, 0.25616, 0.70951, 0.74635, 0.80259, 0.89647, 0.95601]),
           st.sampled_from([-1.0, -0.74785, -0.57679, -0.39458, -0.37643, -0.18667, -0.08079, 0.07098, 0.10087, 0.34871]),
           st.sampled_from([-0.68851, -0.64658, -0.26577, 0.15552, 0.53321, 0.55677, 0.86467, 0.89307, 0.9105, 1.0]),
           st.sampled_from([-0.69774, -0.18645, -0.15114, -0.12174, -0.07705, -0.07231, -0.06295, -0.03229, 0.08101, 0.19588]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.floats(min_value=-1.0, max_value=-0.481937, allow_nan=False),
           st.sampled_from([0.1518, 0.1903, 0.25435, 0.3783, 0.43117, 0.49664, 0.88119, 0.9033, 0.93469, 0.97032]),
           st.sampled_from([-0.57649, -0.54928, -0.19321, -0.12862, -0.02478, -0.01251, 0.0107, 0.27338, 0.39762, 0.56101]),
           st.sampled_from([-0.52823, 0.37549, 0.53165, 0.59024, 0.61217, 0.70058, 0.70798, 0.72973, 0.83951, 0.97059]),
           st.sampled_from([-0.82127, -0.46048, -0.34686, -0.07029, 0.02956, 0.08126, 0.13296, 0.18897, 0.19849, 0.3962]),
           st.sampled_from([-0.47419, -0.13341, 0.02249, 0.33333, 0.34749, 0.50169, 0.5884, 0.64407, 0.70941, 0.96778]),
           st.sampled_from([-0.41173, -0.23902, -0.18182, -0.16243, -0.00577, 0.14783, 0.19083, 0.27926, 0.75837, 0.79444]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.sampled_from([-0.67935, -0.64286, -0.26667, 0.03852, 0.05866, 0.17188, 0.43636, 0.79157, 0.85271, 1.0]),
           st.sampled_from([-0.63636, -0.5421, -0.08459, -0.00838, 0.00686, 0.01437, 0.02461, 0.2875, 0.5782, 0.81586]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.36156, -0.11949, 0.04861, 0.05911, 0.08069, 0.11342, 0.125, 0.21908, 0.5125]),
           st.sampled_from([-0.75693, -0.36393, -0.32937, -0.17778, -0.10868, 0.08374, 0.2825, 0.5, 0.57377, 0.71216]),
           st.floats(min_value=-1.0, max_value=-0.896692, allow_nan=False),
           st.sampled_from([0.0214, 0.25795, 0.3447, 0.45238, 0.5473, 0.61745, 0.66938, 0.80893, 0.91473, 1.0]),
           st.sampled_from([-0.63427, -0.40446, -0.23404, -0.10984, -0.06879, 0.01172, 0.15299, 0.21053, 0.32899, 0.55147]),
           st.sampled_from([-0.45663, -0.0485, 0.0, 0.03786, 0.20743, 0.47744, 0.82143, 0.93243, 0.93359, 1.0]),
           st.sampled_from([-0.42444, -0.34617, -0.16818, -0.10951, -0.10241, -0.01953, -0.01284, -0.00763, 0.35147, 0.64]),
           st.sampled_from([-0.19091, -0.18494, -0.07576, -0.05921, -0.03158, 0.11538, 0.29091, 0.58965, 0.61567, 0.88462]),
           st.sampled_from([-1.0, -0.84848, -0.2, -0.19071, -0.1875, -0.13725, 0.08978, 0.09559, 0.275, 0.58772]),
           st.sampled_from([-1.0, -0.79085, -0.30241, -0.095, 0.06425, 0.14645, 0.23725, 0.28571, 0.36364, 0.68852]),
           st.sampled_from([-0.68065, -0.24206, -0.22414, -0.09804, -0.04598, 0.22957, 0.27322, 0.31439, 0.38869, 0.73393]),
           st.sampled_from([-0.55711, -0.51535, 0.0, 0.17589, 0.2, 0.29062, 0.59091, 0.76929, 0.81395, 0.8227]),
           st.sampled_from([-1.0, -0.77206, -0.71951, -0.45417, -0.29, -0.11765, -0.11351, 0.15476, 0.59152, 1.0]),
           st.sampled_from([0.12121, 0.18993, 0.31755, 0.33108, 0.33746, 0.46642, 0.51979, 0.65558, 0.66818, 0.8076]),
           st.sampled_from([-0.43025, -0.42862, -0.17176, -0.14773, -0.07979, -0.00423, -0.00153, 0.02381, 0.63147, 0.78036]),
           st.sampled_from([-0.74628, -0.29735, -0.20028, -0.19792, -0.04749, -0.04575, 0.00048, 0.04372, 0.45238, 0.98484]),
           st.sampled_from([-0.69594, -0.3, -0.04, 0.00428, 0.01478, 0.09728, 0.38803, 0.5067, 0.5081, 1.0]),
           st.sampled_from([-0.6, -0.18056, -0.00083, 0.0, 0.01519, 0.06897, 0.66667, 0.83209, 0.94118, 0.98636]),
           st.sampled_from([-0.94118, -0.46801, -0.06571, -0.01149, 0.0, 0.00888, 0.23725, 0.26585, 0.45522, 0.5625]),
           st.sampled_from([-0.91574, -0.02494, -0.00069, 0.05136, 0.07853, 0.21, 0.32143, 0.45098, 0.57273, 0.95703]),
           st.sampled_from([-1.0, -0.46734, -0.43137, -0.40888, -0.03718, 0.1639, 0.19405, 0.23387, 0.8071, 0.92236]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.floats(min_value=-0.481934, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.1517, -0.00403, 0.06061, 0.32787, 0.37681, 0.42273, 0.57143, 0.77884, 0.90234]),
           st.sampled_from([-0.61393, -0.27778, -0.02612, -0.01672, 0.00713, 0.05229, 0.06977, 0.08182, 0.10349, 0.69401]),
           st.sampled_from([-1.0, -0.75, -0.125, 0.0, 0.02399, 0.16288, 0.25682, 0.39118, 0.43137, 1.0]),
           st.sampled_from([-0.63156, -0.25946, -0.02398, -0.01854, 0.0, 0.11778, 0.1286, 0.22464, 0.23188, 0.5]),
           st.sampled_from([-0.08208, -0.01427, -0.00039, 0.0, 0.1167, 0.16827, 0.17045, 0.28919, 0.34411, 0.90667]),
           st.sampled_from([-1.0, -0.18826, -0.08978, -0.06314, -0.02447, 0.01997, 0.04586, 0.19551, 0.34732, 1.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.86701, -0.20685, -0.15271, -0.06343, -0.03516, -0.00592, 0.0, 0.02568, 0.15625, 0.83333]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76087, -0.33203, -0.2381, -0.23067, 0.03462, 0.04598, 0.11342, 0.12159, 0.5125, 0.82857]),
           st.floats(min_value=-1.0, max_value=0.925609, allow_nan=False),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.01128, 0.0214, 0.23346, 0.31889, 0.34649, 0.60586, 0.61745, 0.625, 0.66938, 0.9]),
           st.sampled_from([-0.5, -0.40446, 0.00862, 0.04576, 0.0625, 0.08424, 0.21053, 0.30921, 0.55147, 0.95833]),
           st.sampled_from([-0.41912, -0.0485, 0.1875, 0.23341, 0.36285, 0.43685, 0.48182, 0.58958, 0.83721, 0.97311]),
           st.sampled_from([-0.94853, -0.89098, -0.41368, -0.1517, -0.10241, 0.00141, 0.03876, 0.39394, 0.64, 0.99557]),
           st.sampled_from([-0.81699, -0.18494, -0.07576, -0.03158, 0.01427, 0.22807, 0.24514, 0.29091, 0.33333, 0.58965]),
           st.floats(min_value=-1.0, max_value=0.171028, allow_nan=False),
           st.sampled_from([-0.33656, -0.08333, 0.01401, 0.06425, 0.08949, 0.14645, 0.57836, 0.64706, 0.83726, 1.0]),
           st.sampled_from([-0.34265, -0.23476, -0.1554, -0.09804, 0.07027, 0.13712, 0.14516, 0.55172, 0.88444, 1.0]),
           st.sampled_from([-0.38214, -0.00343, 0.02727, 0.16458, 0.18681, 0.28301, 0.66791, 0.80521, 0.95452, 1.0]),
           st.sampled_from([-0.71951, -0.00883, 0.03623, 0.08046, 0.15018, 0.21818, 0.36146, 0.70982, 0.95122, 1.0]),
           st.sampled_from([-1.0, -0.34187, -0.04023, 0.0, 0.22807, 0.46429, 0.66818, 0.8384, 0.84091, 1.0]),
           st.sampled_from([-1.0, -0.63518, -0.18333, -0.07979, -0.02734, 0.05172, 0.07323, 0.07542, 0.1, 0.12727]),
           st.sampled_from([-0.29735, -0.25914, -0.06959, 0.00759, 0.01894, 0.04325, 0.50464, 0.61831, 0.67213, 0.98484]),
           st.sampled_from([-0.62796, -0.4386, -0.04, 0.00238, 0.01478, 0.04762, 0.09728, 0.49091, 0.63496, 0.68101]),
           st.sampled_from([-0.34128, -0.29825, -0.00083, 0.01519, 0.02793, 0.11213, 0.16667, 0.83209, 0.8913, 0.94118]),
           st.sampled_from([-0.94118, 0.02956, 0.06895, 0.24936, 0.26585, 0.34766, 0.35668, 0.5625, 0.77425, 0.80934]),
           st.sampled_from([-0.02494, 0.03125, 0.15103, 0.32727, 0.47015, 0.65493, 0.78932, 0.82937, 0.95703, 1.0]),
           st.sampled_from([-0.46875, -0.40833, -0.24, -0.17012, -0.00391, 0.1639, 0.4375, 0.5, 0.8071, 0.85357]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.floats(min_value=-1.0, max_value=-0.239392, allow_nan=False),
           st.sampled_from([-0.27917, -0.12879, -0.07859, -0.04779, 0.00564, 0.06061, 0.11213, 0.32463, 0.43481, 0.84496]),
           st.sampled_from([-1.0, -0.93182, -0.75625, -0.66494, -0.31373, -0.21901, -0.03279, 0.03125, 0.18854, 0.53846]),
           st.sampled_from([-0.75, -0.66667, -0.09339, -0.00761, 0.04447, 0.18301, 0.27869, 0.35088, 0.43137, 0.87757]),
           st.sampled_from([-1.0, -0.7817, -0.63156, -0.1875, -0.06522, -0.04808, 0.00015, 0.05419, 0.24086, 0.90426]),
           st.sampled_from([-0.67553, -0.19792, -0.19066, -8e-05, 0.17045, 0.25441, 0.2619, 0.34411, 0.4918, 1.0]),
           st.sampled_from([-0.55837, -0.27451, -0.15676, -0.00575, 0.0305, 0.28794, 0.33611, 0.41913, 0.45745, 0.61458]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.92453, -0.24783, -0.12371, -0.01366, 0.05812, 0.06794, 0.0738, 0.09709, 0.1581, 0.29492]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.17674, -0.07383, -0.03227, 0.00202, 0.02968, 0.22222, 0.33598, 0.41149, 0.52013, 0.67248]),
           st.floats(min_value=-1.0, max_value=0.925609, allow_nan=False),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.03643, 0.21592, 0.59784, 0.75683, 0.79571, 0.84111, 0.843, 0.86555, 0.91176, 0.96974]),
           st.sampled_from([-0.36174, -0.10605, -0.06257, 0.02772, 0.35052, 0.6452, 0.82732, 0.87916, 0.93596, 1.0]),
           st.sampled_from([-0.83152, -0.73939, -0.31605, 0.04636, 0.61706, 0.71029, 0.77174, 0.86258, 0.9651, 0.9982]),
           st.sampled_from([-0.49057, 0.02381, 0.04167, 0.05346, 0.15772, 0.50122, 0.51879, 0.62719, 0.80454, 0.96301]),
           st.sampled_from([-0.46177, 0.14939, 0.4327, 0.59755, 0.73961, 0.85571, 0.90818, 0.96186, 0.97213, 1.0]),
           st.floats(min_value=-1.0, max_value=0.171028, allow_nan=False),
           st.sampled_from([-0.82811, -0.82417, -0.77106, -0.70747, -0.35737, 0.05167, 0.62162, 0.69946, 0.71598, 0.72075]),
           st.sampled_from([-0.13719, -0.06486, 0.0, 0.06211, 0.06488, 0.41471, 0.54965, 0.56476, 0.82163, 0.93682]),
           st.sampled_from([-0.77356, -0.38916, -0.33333, 0.58548, 0.64068, 0.86889, 0.87339, 0.94837, 0.9948, 1.0]),
           st.sampled_from([-0.61189, -0.344, -0.21216, -0.15518, -0.14286, -0.04531, 0.02004, 0.07135, 0.12426, 0.2]),
           st.sampled_from([-1.0, -0.71863, -0.20408, -0.00701, 0.28379, 0.53392, 0.54305, 0.68131, 0.76771, 0.83424]),
           st.sampled_from([-0.24017, -0.02956, -0.01667, -0.01039, 0.0303, 0.03881, 0.03939, 0.07717, 0.11339, 0.47794]),
           st.sampled_from([-1.0, -0.92254, -0.57695, -0.52475, -0.12064, -0.10237, 0.13712, 0.37824, 0.68459, 0.80983]),
           st.sampled_from([-0.47308, -0.41085, -0.39328, -0.08333, 0.04743, 0.14258, 0.14887, 0.44614, 0.54212, 0.92001]),
           st.sampled_from([-0.73969, -0.16178, -0.13013, 0.33333, 0.35289, 0.42268, 0.75081, 0.77778, 0.92155, 0.9893]),
           st.sampled_from([-0.91338, -0.7757, -0.70762, -0.15152, -0.08495, -0.03845, -0.01782, 0.05352, 0.08824, 0.88869]),
           st.sampled_from([-0.74265, -0.36799, -0.29291, -0.07782, 0.06911, 0.44483, 0.47038, 0.67016, 0.81329, 0.89593]),
           st.sampled_from([-0.4022, -0.26115, -0.22083, -0.02589, 0.00392, 0.03591, 0.05929, 0.23724, 0.25747, 0.50489]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.floats(min_value=-0.239389, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.75406, -0.57241, -0.55147, -0.18226, 0.0781, 0.41667, 0.431, 0.74822, 0.75, 0.80901]),
           st.sampled_from([-0.85762, -0.60858, -0.42593, -0.19714, -0.07816, 0.02037, 0.08299, 0.26336, 0.70682, 0.80439]),
           st.sampled_from([-0.71475, 0.16604, 0.19667, 0.26214, 0.33921, 0.58187, 0.59759, 0.83524, 0.8666, 0.94578]),
           st.sampled_from([-0.47977, -0.13722, -0.05938, 0.0, 0.068, 0.16667, 0.18345, 0.36364, 0.38915, 0.53372]),
           st.sampled_from([-0.64056, -0.47419, -0.04608, 0.02249, 0.39394, 0.42189, 0.65258, 0.74323, 0.86733, 0.92553]),
           st.sampled_from([-0.497, -0.44739, -0.39487, -0.21918, 0.02762, 0.09719, 0.29942, 0.49831, 0.57552, 1.0]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.62879, -0.05, -0.00592, 0.01437, 0.02461, 0.12482, 0.14861, 0.15564, 0.4, 1.0]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.80553, -0.23067, -0.11949, -0.04119, 0.04918, 0.1, 0.21908, 0.26131, 0.26708, 0.3913]),
           st.floats(min_value=0.925612, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.10991, 0.14706, 0.30682, 0.35821, 0.46458, 0.60586, 0.61745, 0.77273, 0.80648, 0.8401]),
           st.sampled_from([-0.10984, -0.06879, -0.05909, -0.03911, 0.01639, 0.0625, 0.10702, 0.15299, 0.2973, 0.47173]),
           st.sampled_from([-1.0, 0.0, 0.025, 0.23341, 0.46053, 0.54478, 0.64394, 0.74597, 0.82143, 0.89514]),
           st.sampled_from([-0.62723, -0.2803, -0.16818, -0.11765, -0.06609, -0.03279, -0.01284, 0.35147, 0.36194, 1.0]),
           st.sampled_from([-0.6723, -0.07576, -0.00559, 0.0925, 0.12357, 0.22807, 0.24514, 0.29091, 0.58965, 0.61567]),
           st.floats(min_value=-1.0, max_value=0.171028, allow_nan=False),
           st.sampled_from([-0.1335, -0.095, 0.08949, 0.14645, 0.35088, 0.56347, 0.56818, 0.57836, 0.68852, 0.88683]),
           st.sampled_from([-0.41818, -0.24206, -0.1554, -0.14767, -0.03403, 0.14137, 0.14516, 0.35507, 0.65565, 0.6791]),
           st.sampled_from([-0.51535, -0.09483, -0.04708, 0.15625, 0.17589, 0.28301, 0.46429, 0.72059, 0.72807, 0.95313]),
           st.sampled_from([-0.83007, -0.2671, -0.17863, -0.03947, -0.02956, -0.01818, 0.0, 0.09497, 0.21818, 0.59152]),
           st.sampled_from([-1.0, -0.12808, 0.31755, 0.36482, 0.45076, 0.56098, 0.58424, 0.65558, 0.66818, 0.90698]),
           st.sampled_from([-1.0, -0.93599, -0.60294, -0.28257, -0.17176, -0.11567, 0.05952, 0.22115, 0.24903, 0.84922]),
           st.sampled_from([-0.48, -0.31402, -0.20028, -0.102, 0.0, 0.04325, 0.24903, 0.67213, 0.68839, 0.97068]),
           st.sampled_from([-0.25, -0.21081, -0.02539, 0.0, 0.00075, 0.01478, 0.03669, 0.05078, 0.14792, 0.30745]),
           st.sampled_from([-0.57576, -0.29825, -0.18043, -3e-05, 0.02793, 0.06897, 0.29908, 0.60201, 0.83209, 0.94118]),
           st.sampled_from([-0.80769, -0.27778, -0.2619, -0.23529, 0.00019, 0.05456, 0.16516, 0.23725, 0.35668, 0.80934]),
           st.sampled_from([-0.90302, -0.14803, -0.10625, 0.03513, 0.24242, 0.31148, 0.32143, 0.5375, 0.54446, 0.8062]),
           st.sampled_from([-0.46734, -0.27778, -0.03718, -0.01326, -0.01269, 0.00559, 0.03876, 0.16364, 0.28482, 0.70238]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.46809, -0.40116, -0.28475, -0.08571, -0.04926, -0.00043, 0.0, 0.51613, 0.88248, 1.0]),
           st.sampled_from([-0.27917, -0.22917, 0.0, 0.01997, 0.06061, 0.11213, 0.42273, 0.43182, 0.62958, 0.95982]),
           st.sampled_from([-0.38914, -0.20099, -0.06641, -0.01947, -0.01672, 0.00713, 0.10349, 0.15436, 0.24242, 0.69401]),
           st.sampled_from([-0.66667, 0.02586, 0.04469, 0.06329, 0.16288, 0.25682, 0.27869, 0.48927, 0.69792, 1.0]),
           st.sampled_from([-0.63156, -0.29915, -0.06522, -0.04808, -0.02398, 0.00015, 0.00108, 0.14643, 0.65268, 0.81007]),
           st.sampled_from([-0.67553, -0.18494, -0.06117, -0.04598, -0.04013, -8e-05, 0.07389, 0.16827, 0.34411, 0.52632]),
           st.sampled_from([-0.65935, -0.48241, -0.27292, -0.18826, -0.10837, -0.02447, 0.07895, 0.12011, 0.12755, 0.28794]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_9(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_9']['n_samples'] += 1
        self.data['tests']['test_9']['samples'].append(x_test)
        self.data['tests']['test_9']['y_expected'].append(y_expected[0])
        self.data['tests']['test_9']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.21996, -0.07896, -0.0239, 0.00075, 0.07088, 0.0862, 0.1037, 0.11933, 0.39286, 0.45455]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.0, max_value=0.231989, allow_nan=False),
           st.sampled_from([-0.13129, 0.29921, 0.47564, 0.52751, 0.54916, 0.70103, 0.90996, 0.93438, 0.94616, 0.97165]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.06726, 0.15, 0.36667, 0.53968, 0.58876, 0.66725, 0.74313, 0.85996, 0.86486, 0.94817]),
           st.sampled_from([-0.30566, -0.06971, -0.0625, -0.04528, -0.00273, 0.01282, 0.02448, 0.04317, 0.18261, 0.69865]),
           st.sampled_from([-0.41985, -0.35148, 0.36705, 0.44793, 0.47917, 0.50376, 0.65164, 0.8555, 0.91287, 0.96936]),
           st.sampled_from([-0.31809, -0.24077, -0.06444, -0.01511, -0.00221, 0.01132, 0.03546, 0.1313, 0.51879, 0.54701]),
           st.sampled_from([-0.76051, -0.66616, -0.47644, -0.41729, 0.09467, 0.21951, 0.39699, 0.79343, 0.85221, 0.96674]),
           st.floats(min_value=0.171031, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.96078, -0.8268, -0.57377, 0.25192, 0.60194, 0.62264, 0.62698, 0.72517, 0.79555, 0.87161]),
           st.sampled_from([-0.4817, -0.43796, -0.23433, -0.18182, -0.15616, -0.13719, -0.11856, -0.06486, 0.01151, 0.72526]),
           st.sampled_from([-1.0, -0.6675, -0.06328, 0.29352, 0.38696, 0.47087, 0.53131, 0.66775, 0.83669, 0.92124]),
           st.sampled_from([-0.61894, -0.38606, -0.21231, 0.03458, 0.07135, 0.09146, 0.19328, 0.26721, 0.42014, 0.91624]),
           st.sampled_from([-0.56136, 0.2707, 0.31707, 0.45107, 0.51247, 0.8357, 0.95663, 0.96741, 0.98579, 0.98678]),
           st.sampled_from([-0.42052, -0.24497, -0.17813, -0.14327, -0.08668, -0.04887, -0.04711, -0.00775, 0.59862, 0.70398]),
           st.sampled_from([-0.78973, -0.49488, -0.32354, 0.09513, 0.36364, 0.66233, 0.68459, 0.794, 0.96791, 0.97158]),
           st.sampled_from([-0.85205, -0.04971, -0.03377, 0.00988, 0.06667, 0.06856, 0.07631, 0.2, 0.49508, 0.71699]),
           st.sampled_from([-0.37835, -0.0225, 0.00494, 0.42546, 0.5312, 0.77041, 0.83486, 0.92155, 0.93126, 0.98122]),
           st.floats(min_value=-1.0, max_value=0.052703, allow_nan=False),
           st.sampled_from([-0.16489, -0.07361, -0.00031, 0.22792, 0.48311, 0.77, 0.79503, 0.9452, 0.99695, 0.99899]),
           st.sampled_from([-0.66014, -0.20177, -0.0009, 0.02811, 0.04878, 0.08107, 0.10678, 0.11348, 0.15254, 1.0]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.69556, -0.53127, -0.2321, -0.16837, -0.14162, -0.02731, 0.10039, 0.24618, 0.4604, 0.78541]),
           st.sampled_from([0.37941, 0.40675, 0.52798, 0.61706, 0.67273, 0.78889, 0.88568, 0.90854, 0.9434, 0.95603]),
           st.sampled_from([-0.68794, -0.64428, -0.21129, -0.21053, -0.10306, -0.04448, -0.00024, 0.03377, 0.56101, 0.59071]),
           st.sampled_from([0.22222, 0.35217, 0.37549, 0.45098, 0.53165, 0.54545, 0.55916, 0.61217, 0.91991, 0.97845]),
           st.sampled_from([-0.35222, -0.12452, -0.05525, -0.01868, 0.06217, 0.0641, 0.2459, 0.36364, 0.42488, 0.44195]),
           st.sampled_from([-0.11894, 0.01336, 0.48319, 0.57455, 0.61372, 0.75495, 0.78761, 0.83574, 0.93144, 0.93323]),
           st.sampled_from([-1.0, -0.67226, -0.27218, -0.24324, -0.16827, -0.03175, 0.08616, 0.09947, 0.12556, 0.15114]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_10(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_10']['n_samples'] += 1
        self.data['tests']['test_10']['samples'].append(x_test)
        self.data['tests']['test_10']['y_expected'].append(y_expected[0])
        self.data['tests']['test_10']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.18829, -0.07843, 0.00686, 0.02568, 0.10811, 0.12482, 0.14861, 0.15564, 0.2875, 0.35949]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.0, max_value=0.231989, allow_nan=False),
           st.sampled_from([-0.36393, -0.17778, -0.10868, -0.07661, 0.0, 0.02116, 0.26459, 0.50814, 0.57377, 0.60784]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, 0.01128, 0.05405, 0.10991, 0.34649, 0.35821, 0.60586, 0.66938, 0.76087, 0.87719]),
           st.sampled_from([-0.10984, -0.06879, -0.06811, -0.05909, 0.01172, 0.18719, 0.30921, 0.32899, 0.58304, 0.95041]),
           st.sampled_from([0.01676, 0.01724, 0.025, 0.04461, 0.20743, 0.29091, 0.62195, 0.68627, 0.83721, 0.89514]),
           st.sampled_from([-1.0, -0.3871, -0.15194, -0.10241, -0.07542, 0.03876, 0.07018, 0.64, 0.7619, 1.0]),
           st.sampled_from([-0.71875, -0.18494, -0.05921, -0.03125, -0.00885, 0.0106, 0.12357, 0.12766, 0.58965, 0.70067]),
           st.floats(min_value=0.171031, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.95489, -0.54852, -0.12, 0.0106, 0.01401, 0.06425, 0.08949, 0.28571, 0.56347, 0.82077]),
           st.sampled_from([-0.68065, -0.67708, -0.55, -0.1554, -0.1548, -0.02614, -0.00166, 0.06065, 0.14516, 0.21]),
           st.sampled_from([-0.38214, -0.37133, -0.02661, 0.28301, 0.29062, 0.47364, 0.54094, 0.64662, 0.66791, 0.95313]),
           st.sampled_from([-0.45417, -0.29354, -0.11765, -0.10196, -0.04805, -0.01818, 0.13839, 0.45201, 0.46502, 0.75]),
           st.sampled_from([-0.34773, 0.0, 0.0875, 0.14254, 0.22807, 0.46642, 0.65558, 0.7694, 0.8076, 1.0]),
           st.sampled_from([-0.42113, -0.28257, -0.00153, 0.05172, 0.07542, 0.12727, 0.18729, 0.24903, 0.84922, 0.92949]),
           st.sampled_from([-1.0, -0.74628, -0.31402, -0.25, 0.00048, 0.01894, 0.0214, 0.32675, 0.58824, 0.98484]),
           st.sampled_from([-0.453, -0.3, -0.21875, -0.21081, 0.14792, 0.24582, 0.49091, 0.5067, 0.68101, 1.0]),
           st.sampled_from([-0.18043, -0.03736, -3e-05, 0.3125, 0.53448, 0.83209, 0.8667, 0.94118, 0.98636, 1.0]),
           st.floats(min_value=0.052706, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.91574, -0.16253, 0.0, 0.03513, 0.24242, 0.28604, 0.54446, 0.8062, 0.80921, 1.0]),
           st.sampled_from([-0.53846, -0.40888, -0.27778, -0.10297, 2e-05, 0.16418, 0.19405, 0.28482, 0.36508, 1.0]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.40116, -0.24668, -0.04926, -0.01128, 0.1463, 0.15942, 0.26256, 0.35924, 0.625, 0.75]),
           st.sampled_from([-1.0, -0.47, -0.1517, -0.05911, 0.01997, 0.03073, 0.32463, 0.32787, 0.34545, 0.55405]),
           st.sampled_from([-0.48936, -0.24561, -0.0936, 0.03125, 0.06977, 0.10349, 0.14786, 0.185, 0.69401, 1.0]),
           st.sampled_from([-0.54891, -0.21354, -0.16626, -0.09634, -0.02916, 0.02399, 0.04433, 0.06329, 0.09611, 1.0]),
           st.sampled_from([-1.0, -0.7817, -0.3254, -0.27083, -0.21569, 0.00015, 0.02614, 0.13521, 0.20175, 0.24086]),
           st.sampled_from([-0.19792, -0.19066, 0.07389, 0.1167, 0.34411, 0.47059, 0.47569, 0.4918, 0.52632, 1.0]),
           st.sampled_from([-0.48241, -0.34375, -0.03352, 0.0, 2e-05, 0.0305, 0.09375, 0.12755, 0.41913, 0.68822]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_11(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_11']['n_samples'] += 1
        self.data['tests']['test_11']['samples'].append(x_test)
        self.data['tests']['test_11']['y_expected'].append(y_expected[0])
        self.data['tests']['test_11']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.731249, allow_nan=False),
           st.sampled_from([-0.03365, -0.01531, -0.01179, 0.00526, 0.01403, 0.09771, 0.16195, 0.18198, 0.29167, 0.30634]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231992, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.27273, 0.29303, 0.38889, 0.40816, 0.50649, 0.85096, 0.86635, 0.87873, 0.94982, 0.96683]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.13342, 0.21592, 0.47894, 0.53312, 0.71157, 0.73673, 0.73678, 0.84111, 0.90244, 0.9872]),
           st.sampled_from([-0.2707, -0.06119, -0.01707, -0.01418, -0.00494, -0.00462, 0.01198, 0.69865, 0.83548, 0.93345]),
           st.sampled_from([-0.02116, 0.50847, 0.65164, 0.67314, 0.70508, 0.78424, 0.81121, 0.94717, 0.99026, 0.99448]),
           st.sampled_from([-0.31809, -0.07797, -0.07496, 0.04545, 0.10916, 0.1313, 0.14118, 0.2782, 0.81191, 0.83713]),
           st.sampled_from([-0.7766, -0.64574, 0.28931, 0.45283, 0.57966, 0.70716, 0.77937, 0.79962, 0.8, 0.80668]),
           st.floats(min_value=0.171031, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.94053, -0.69767, 0.59361, 0.67849, 0.7, 0.87534, 0.92048, 0.93369, 0.9526, 0.95732]),
           st.sampled_from([-0.43796, -0.18182, -0.0453, 0.00899, 0.05097, 0.12785, 0.14662, 0.27273, 0.74079, 0.84804]),
           st.sampled_from([-0.7894, -0.375, -0.28269, -0.06122, 0.19128, 0.47087, 0.60101, 0.81664, 0.81939, 0.93929]),
           st.sampled_from([-0.31412, -0.2952, -0.2178, -0.20924, -0.13525, -0.1068, -0.00642, 0.03356, 0.07801, 0.48311]),
           st.sampled_from([-0.73564, -0.65426, 0.18702, 0.36087, 0.65091, 0.82222, 0.83584, 0.84537, 0.93784, 0.97032]),
           st.sampled_from([-0.73641, -0.32963, 0.0, 0.01393, 0.01557, 0.06038, 0.1, 0.29304, 0.55102, 1.0]),
           st.sampled_from([-0.81262, -0.52504, -0.49488, 0.47541, 0.47807, 0.47894, 0.63279, 0.8617, 0.91693, 1.0]),
           st.sampled_from([-0.93596, -0.66278, -0.27502, -0.15317, -0.07905, -0.02778, -0.01891, 0.00988, 0.54462, 0.92001]),
           st.sampled_from([-0.86029, -0.02876, 0.00494, 0.28626, 0.71002, 0.75071, 0.87492, 0.88907, 0.92815, 0.95278]),
           st.sampled_from([-0.37643, -0.26795, -0.12062, -0.03641, -0.00166, -0.00112, 0.05506, 0.07697, 0.1875, 0.22625]),
           st.sampled_from([-0.75273, -0.46534, 0.27942, 0.35714, 0.40615, 0.40678, 0.41366, 0.5676, 0.60656, 0.8408]),
           st.sampled_from([-0.81634, -0.66421, -0.59954, -0.26356, -0.19507, -0.03176, 0.0, 0.22449, 0.34658, 0.74689]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.85669, -0.68593, -0.12041, -0.01358, 0.01611, 0.02542, 0.02897, 0.13282, 0.25553, 0.66335]),
           st.sampled_from([0.46275, 0.52613, 0.5355, 0.53952, 0.61706, 0.66792, 0.74046, 0.81988, 0.85611, 0.94599]),
           st.sampled_from([-0.57649, -0.11409, -0.10429, -0.07816, -0.07036, 0.00818, 0.02222, 0.03377, 0.13333, 0.59071]),
           st.sampled_from([-0.1737, -0.02962, 0.19667, 0.47029, 0.53659, 0.61733, 0.64348, 0.72973, 0.78479, 0.90409]),
           st.sampled_from([-0.69613, -0.60454, -0.21542, -0.14053, -0.1087, -0.09461, -0.0422, 0.02956, 0.29074, 0.86835]),
           st.sampled_from([-0.81383, -0.42912, -0.26531, -0.04023, 0.22101, 0.34749, 0.56045, 0.56361, 0.84258, 0.9842]),
           st.sampled_from([-0.66667, -0.58086, -0.53319, -0.44222, -0.33588, -0.22034, -0.10309, -0.01832, -0.01425, 0.14783]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_12(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_12']['n_samples'] += 1
        self.data['tests']['test_12']['samples'].append(x_test)
        self.data['tests']['test_12']['y_expected'].append(y_expected[0])
        self.data['tests']['test_12']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=0.731252, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.86701, -0.59677, -0.5421, -0.05, 0.10811, 0.15564, 0.2875, 0.35949, 0.38889, 0.83333]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76087, -0.36156, -0.33203, -0.23067, -0.14545, 0.1, 0.125, 0.3913, 0.51724, 0.82857]),
           st.sampled_from([-0.32937, -0.00763, 0.02116, 0.10526, 0.18182, 0.23182, 0.5, 0.88444, 0.97545, 0.98019]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.87097, -0.30719, -0.07759, 0.12586, 0.14706, 0.3447, 0.69258, 0.71875, 0.8401, 1.0]),
           st.floats(min_value=-1.0, max_value=-0.755856, allow_nan=False),
           st.sampled_from([-1.0, -0.67285, -0.18599, 0.025, 0.04461, 0.50874, 0.64394, 0.74597, 0.91036, 0.93243]),
           st.sampled_from([-1.0, -0.62723, -0.2803, -0.25712, -0.00763, 0.02575, 0.07018, 0.25292, 0.27038, 0.64]),
           st.sampled_from([-1.0, -0.81699, -0.00885, 0.0925, 0.11538, 0.22807, 0.23958, 0.47368, 0.58965, 0.95824]),
           st.sampled_from([-0.59212, -0.1875, -0.13725, -0.11111, -0.02282, -0.0055, 0.09559, 0.18227, 0.27153, 0.44523]),
           st.sampled_from([-0.79085, -0.51685, -0.20019, -0.095, -0.08333, -0.01975, 0.01754, 0.06425, 0.20635, 0.36364]),
           st.sampled_from([-1.0, -0.67708, -0.24206, -0.23476, -0.02614, -0.00166, 0.07027, 0.1286, 0.28571, 0.65565]),
           st.sampled_from([-0.53788, -0.02661, 0.05499, 0.19672, 0.37162, 0.59091, 0.67, 0.72059, 0.72807, 0.95452]),
           st.sampled_from([-0.81556, -0.4627, -0.03947, -0.03516, -0.02853, 0.03623, 0.04651, 0.16391, 0.36965, 0.40455]),
           st.sampled_from([-0.12808, -0.06425, 0.0, 0.51979, 0.54902, 0.63728, 0.70736, 0.82979, 0.84, 0.90698]),
           st.sampled_from([-0.85583, -0.54248, -0.07979, 0.02381, 0.03406, 0.12727, 0.14375, 0.24903, 0.78175, 0.92949]),
           st.sampled_from([-0.99219, -0.75, -0.14343, 0.00138, 0.04325, 0.12727, 0.24903, 0.32675, 0.50464, 0.68839]),
           st.sampled_from([-0.83007, -0.19608, -0.04, -0.02539, 0.0, 0.00075, 0.24582, 0.30745, 0.68101, 0.74359]),
           st.sampled_from([-0.84286, -0.73864, -0.6, -0.29825, -0.2, -0.18636, 0.05136, 0.8667, 0.94118, 1.0]),
           st.sampled_from([-0.47594, -0.27083, -0.18667, 0.02956, 0.06895, 0.23725, 0.33671, 0.45522, 0.62349, 0.77425]),
           st.sampled_from([-1.0, -0.84792, -0.42568, -0.20332, -0.14003, -0.02862, 0.29167, 0.54446, 0.78932, 0.82937]),
           st.sampled_from([-0.53846, -0.40833, -0.27273, -0.01535, 0.16364, 0.16418, 0.19405, 0.5, 0.68, 0.85714]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.40116, -0.01128, 0.01373, 0.01724, 0.10976, 0.16019, 0.22667, 0.5, 0.63492, 1.0]),
           st.sampled_from([-0.2234, 0.00564, 0.03073, 0.32463, 0.37681, 0.42273, 0.43182, 0.56863, 0.62958, 0.95982]),
           st.sampled_from([-0.79141, -0.66494, -0.31373, -0.0936, -0.06641, -0.03279, -0.00279, 0.01149, 0.15436, 0.18854]),
           st.sampled_from([-0.09634, -0.04585, -0.03125, -0.00761, 0.02564, 0.04447, 0.06329, 0.12727, 0.16595, 0.20614]),
           st.sampled_from([-0.7817, -0.27083, -0.25946, -0.06288, 0.00108, 0.02614, 0.05419, 0.07064, 0.31128, 0.5582]),
           st.sampled_from([-1.0, -0.67553, -0.32382, -0.13738, 0.00705, 0.1167, 0.14939, 0.16827, 0.17045, 0.56522]),
           st.sampled_from([-0.34375, -0.15676, 2e-05, 0.01997, 0.02381, 0.04586, 0.19551, 0.33611, 0.34732, 0.68822]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_13(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_13']['n_samples'] += 1
        self.data['tests']['test_13']['samples'].append(x_test)
        self.data['tests']['test_13']['y_expected'].append(y_expected[0])
        self.data['tests']['test_13']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([0.0, 1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=0.731252, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.0, max_value=-0.146967, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.33203, -0.2381, -0.22703, -0.14545, -0.00846, 0.04598, 0.12889, 0.26131, 0.3913, 0.80521]),
           st.sampled_from([-1.0, -0.36393, -0.32937, -0.17778, -0.07661, 0.02116, 0.08374, 0.5, 0.77941, 0.98019]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.12586, 0.3447, 0.34545, 0.34649, 0.35821, 0.5473, 0.62745, 0.66364, 0.66938, 0.76087]),
           st.floats(min_value=-0.755853, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.67285, -0.41912, -0.0485, 0.03786, 0.1875, 0.19455, 0.23341, 0.54478, 0.62195, 0.64394]),
           st.sampled_from([-0.62723, -0.41119, -0.11765, -0.10241, -0.06302, -0.00763, 0.02575, 0.03125, 0.36194, 0.40476]),
           st.sampled_from([-0.6723, -0.51504, -0.26048, -0.19091, -0.05921, -0.03125, 0.0925, 0.12766, 0.34432, 0.61567]),
           st.sampled_from([-0.8209, -0.79313, -0.37679, -0.0184, 0.0, 0.16406, 0.18227, 0.19556, 0.30729, 0.51894]),
           st.sampled_from([-1.0, -0.33656, -0.30241, -0.12, -0.04572, 0.0106, 0.05911, 0.14645, 0.35088, 0.64706]),
           st.sampled_from([-0.41818, -0.16393, -0.1554, -0.1548, -0.09804, -0.03281, 0.14137, 0.21, 0.23288, 0.38602]),
           st.floats(min_value=-1.0, max_value=0.260518, allow_nan=False),
           st.sampled_from([-0.45417, -0.11344, -0.05206, -0.02956, -0.01818, 0.13839, 0.41168, 0.70982, 0.75, 0.95122]),
           st.sampled_from([-0.60131, -0.34187, -0.11575, 0.00342, 0.12121, 0.18993, 0.3141, 0.51979, 0.8384, 1.0]),
           st.sampled_from([-0.42862, -0.2069, 0.02381, 0.03566, 0.05952, 0.14375, 0.18135, 0.24903, 0.31408, 0.78175]),
           st.sampled_from([-1.0, -0.99219, -0.5183, -0.20028, -0.102, 0.04325, 0.32675, 0.61831, 0.97068, 0.98484]),
           st.sampled_from([-0.69594, -0.25, -0.20188, -0.04, -0.03393, 0.00238, 0.00428, 0.03279, 0.24582, 0.38803]),
           st.sampled_from([-1.0, -0.73864, -0.34615, -0.18043, -0.16667, 0.11213, 0.51933, 0.6203, 0.8913, 0.98636]),
           st.sampled_from([-0.80769, -0.70745, -0.47594, -0.46801, -0.27778, -0.2619, -0.16228, 0.23725, 0.35668, 0.77425]),
           st.sampled_from([-0.90302, -0.71528, -0.42568, -0.10625, -0.00069, 0.0, 0.00838, 0.05136, 0.24242, 0.32143]),
           st.sampled_from([-0.40833, -0.27273, -0.10234, -0.03718, 2e-05, 0.16418, 0.27381, 0.65635, 0.68, 0.92236]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.41307, -0.28475, -0.24668, -0.18401, -0.10897, -0.03578, 0.09176, 0.10976, 0.26256, 0.88248]),
           st.sampled_from([-0.22917, -0.14786, -0.00403, 0.42273, 0.43182, 0.57143, 0.62958, 0.63811, 0.84496, 0.90234]),
           st.sampled_from([-0.75625, -0.48485, -0.24561, -0.01672, -0.00279, 0.00713, 0.39286, 0.42677, 0.67708, 0.81573]),
           st.sampled_from([-1.0, -0.27104, -0.09339, -0.04585, -0.03125, 0.04447, 0.16595, 0.20614, 0.25682, 1.0]),
           st.sampled_from([-0.25946, -0.21569, -0.1875, -0.07464, 0.0, 0.1286, 0.14677, 0.31128, 0.61881, 0.81007]),
           st.sampled_from([-1.0, -0.23186, -8e-05, 0.14939, 0.16827, 0.25441, 0.28919, 0.4918, 0.81979, 1.0]),
           st.sampled_from([-0.9375, -0.48241, -0.27292, -0.10837, -0.06865, 0.09375, 0.33611, 0.34732, 0.38065, 0.41913]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_14(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_14']['n_samples'] += 1
        self.data['tests']['test_14']['samples'].append(x_test)
        self.data['tests']['test_14']['y_expected'].append(y_expected[0])
        self.data['tests']['test_14']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=0.731252, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-1.0, max_value=-0.146967, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.13009, -0.11313, 0.0, 0.09792, 0.14211, 0.18281, 0.47702, 0.5381, 0.76131, 0.7616]),
           st.sampled_from([0.19672, 0.34816, 0.51783, 0.80244, 0.84341, 0.84349, 0.92314, 0.93438, 0.94994, 0.98658]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.03643, 0.52632, 0.74706, 0.76966, 0.77152, 0.85746, 0.91759, 0.92871, 0.92908, 0.99374]),
           st.floats(min_value=-0.755853, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.45725, -0.31527, 0.36364, 0.52993, 0.54003, 0.65164, 0.73584, 0.78735, 0.81434, 0.98762]),
           st.sampled_from([-0.14286, -0.03315, -0.0099, 0.08728, 0.23529, 0.30889, 0.48485, 0.58824, 0.59565, 0.94749]),
           st.sampled_from([-0.79135, -0.56535, 0.01707, 0.40349, 0.71588, 0.78225, 0.79343, 0.93902, 0.97213, 0.99173]),
           st.sampled_from([-0.11111, -0.06124, -0.01767, 0.04688, 0.05029, 0.10768, 0.15858, 0.37639, 0.46899, 0.89335]),
           st.sampled_from([-1.0, -0.78658, -0.77106, -0.68421, -0.20588, 0.2514, 0.60194, 0.87235, 0.91118, 0.9917]),
           st.sampled_from([-0.38223, -0.35924, -0.06374, -0.03846, -0.00337, 0.00854, 0.05756, 0.07797, 0.10976, 0.30356]),
           st.floats(min_value=0.260521, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.22917, -0.21622, -0.11583, 0.0028, 0.04167, 0.04813, 0.32407, 0.55102, 0.60756, 0.85475]),
           st.sampled_from([-0.76305, -0.71863, -0.69397, -0.69318, 0.21137, 0.42174, 0.63636, 0.69876, 0.90137, 0.95035]),
           st.sampled_from([-0.82881, -0.63544, -0.48404, -0.26087, -0.17813, 0.12727, 0.12967, 0.39056, 0.5, 0.63548]),
           st.sampled_from([-0.36591, -3e-05, 0.26957, 0.47807, 0.48485, 0.54473, 0.70245, 0.76979, 0.87605, 0.88125]),
           st.sampled_from([-0.85597, -0.55605, -0.31525, -0.23774, -0.23503, -0.01891, -0.0028, 0.07122, 0.12608, 0.71699]),
           st.sampled_from([-0.86029, -0.57135, -0.39766, 0.5312, 0.5343, 0.6615, 0.68183, 0.71525, 0.8843, 0.93815]),
           st.sampled_from([-1.0, -0.74896, -0.67464, -0.59661, -0.16031, -0.08914, 0.03129, 0.05517, 0.07577, 0.35534]),
           st.sampled_from([-0.38483, 0.21645, 0.36318, 0.60656, 0.66643, 0.67077, 0.71253, 0.84033, 0.85054, 1.0]),
           st.sampled_from([-0.66421, -0.63601, -0.22978, -0.1875, -0.15114, -0.00663, 0.18065, 0.22727, 0.37381, 0.74689]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.32792, -0.14256, -0.11876, -0.04597, -0.02302, 0.01375, 0.04356, 0.07124, 0.08718, 0.26388]),
           st.sampled_from([-0.46422, 0.1518, 0.4, 0.52731, 0.6, 0.62764, 0.72478, 0.74089, 0.75784, 0.86747]),
           st.sampled_from([-0.23728, -0.22293, -0.11129, -0.07816, 0.00198, 0.01547, 0.02037, 0.03377, 0.11598, 0.20833]),
           st.sampled_from([-0.6571, -0.34131, -0.02962, 0.36924, 0.58187, 0.70798, 0.88424, 0.90409, 0.94066, 0.99057]),
           st.sampled_from([-0.64537, -0.54487, -0.25706, -0.1875, 0.16465, 0.18345, 0.32492, 0.38915, 0.55061, 0.63636]),
           st.sampled_from([-0.66932, -0.59943, 0.28985, 0.44767, 0.51351, 0.52361, 0.63333, 0.64207, 0.72727, 0.95838]),
           st.sampled_from([-0.75597, -0.7, -0.44222, -0.09186, -0.06668, -0.02532, 0.02441, 0.02823, 0.08616, 0.50903]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_15(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_15']['n_samples'] += 1
        self.data['tests']['test_15']['samples'].append(x_test)
        self.data['tests']['test_15']['y_expected'].append(y_expected[0])
        self.data['tests']['test_15']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([1.0]),
           st.sampled_from([0.0]),
           st.floats(min_value=0.731252, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.146964, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.20253, -0.14706, -0.07383, 0.01305, 0.10827, 0.16996, 0.29308, 0.29757, 0.4703, 0.62233]),
           st.sampled_from([0.08858, 0.09916, 0.43898, 0.47564, 0.8, 0.84349, 0.90071, 0.92314, 0.97811, 0.99601]),
           st.floats(min_value=-0.896689, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.4902, -0.06726, 0.1203, 0.16851, 0.29796, 0.68421, 0.6932, 0.88928, 0.95135, 0.9872]),
           st.floats(min_value=-0.755853, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.68253, -0.45455, 0.56509, 0.67314, 0.73082, 0.78735, 0.81504, 0.95455, 0.98955, 0.99274]),
           st.sampled_from([-0.47929, -0.23229, -0.14286, -0.11073, -0.04292, 0.04102, 0.36719, 0.37685, 0.52408, 0.88971]),
           st.sampled_from([-0.76051, -0.46177, -0.02367, 0.39699, 0.54911, 0.56937, 0.6742, 0.92766, 0.97059, 0.99745]),
           st.sampled_from([-0.50764, -0.06714, -0.02187, -0.00469, 0.04783, 0.08436, 0.0916, 0.10417, 0.50949, 0.75]),
           st.sampled_from([-0.92819, -0.52735, 0.29268, 0.59361, 0.62698, 0.71492, 0.81815, 0.8481, 0.86303, 0.91118]),
           st.sampled_from([-0.11856, -0.05338, -0.03087, 0.02941, 0.05269, 0.06317, 0.19868, 0.32925, 0.42314, 0.79523]),
           st.sampled_from([-0.93001, -0.52742, 0.15215, 0.32432, 0.49259, 0.51067, 0.81051, 0.81664, 0.95465, 0.9963]),
           st.sampled_from([-0.24482, -0.22411, -0.17531, -0.05365, -0.03191, -0.00788, 0.01569, 0.02744, 0.07135, 0.1]),
           st.sampled_from([-0.7218, -0.69318, 0.28379, 0.60317, 0.66667, 0.7103, 0.81, 0.89716, 0.90782, 0.95035]),
           st.sampled_from([-0.66667, -0.09856, -0.04656, -0.01667, 0.03617, 0.04992, 0.11538, 0.12967, 0.15756, 0.5]),
           st.sampled_from([-0.52475, -0.50931, 0.23292, 0.37824, 0.58965, 0.66233, 0.78698, 0.81221, 0.88065, 0.99695]),
           st.sampled_from([-0.7632, -0.59839, -0.44072, -0.00377, 0.00191, 0.02842, 0.18864, 0.27815, 0.292, 0.89383]),
           st.sampled_from([-0.79147, -0.27377, -0.16178, 0.25, 0.33333, 0.51183, 0.63077, 0.77273, 0.92599, 0.93124]),
           st.sampled_from([-0.93788, -0.66878, -0.08453, -0.05882, -0.03641, -0.02591, 0.02099, 0.08141, 0.30804, 0.33537]),
           st.sampled_from([-0.68159, -0.37199, -0.12897, 0.03004, 0.21581, 0.25, 0.31068, 0.67194, 0.743, 0.93988]),
           st.sampled_from([-0.40747, -0.29852, -0.26883, -0.19507, -0.01942, -0.01478, 0.0, 0.0222, 0.05348, 0.7978]),
           st.floats(min_value=-1.0, max_value=0.999944, allow_nan=False),
           st.sampled_from([-0.54629, -0.11808, -0.01673, -0.01647, -0.01475, -0.00128, 0.04452, 0.40972, 0.53235, 0.69435]),
           st.sampled_from([-0.75406, -0.59184, -0.55147, 0.3783, 0.431, 0.51384, 0.53952, 0.74846, 0.88125, 0.90854]),
           st.sampled_from([-0.39143, -0.38059, -0.37088, -0.32298, -0.16713, -0.07479, -0.06423, 0.01547, 0.02242, 0.55805]),
           st.sampled_from([-0.05707, 0.16997, 0.55916, 0.61733, 0.66316, 0.78479, 0.80253, 0.91991, 0.97845, 0.9819]),
           st.sampled_from([-0.39061, -0.25217, -0.17921, -0.15755, -0.09461, -0.08637, -0.01389, 0.00246, 0.1593, 0.77072]),
           st.sampled_from([-0.66932, -0.42912, 0.40821, 0.42152, 0.60294, 0.64662, 0.70833, 0.75115, 0.80355, 0.98934]),
           st.sampled_from([-0.39487, -0.21918, -0.0625, -0.00291, 0.04167, 0.05972, 0.07173, 0.10067, 0.13297, 0.14783]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_16(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_16']['n_samples'] += 1
        self.data['tests']['test_16']['samples'].append(x_test)
        self.data['tests']['test_16']['y_expected'].append(y_expected[0])
        self.data['tests']['test_16']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.0, max_value=0.49, allow_nan=False),
           st.sampled_from([0.0]),
           st.sampled_from([-0.5418, -0.26667, 0.38521, 0.42, 0.5984, 0.71253, 0.72727, 0.84783, 0.91353, 0.94598]),
           st.sampled_from([-1.0, -0.5, -0.14754, -0.06343, -0.00838, 0.02568, 0.2875, 0.35949, 0.52381, 0.83333]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.80553, -0.76087, -0.11949, -0.00846, 0.04918, 0.11342, 0.125, 0.12889, 1.0]),
           st.sampled_from([-0.16628, 0.0, 0.01437, 0.18919, 0.5, 0.52012, 0.71216, 0.77941, 0.98019, 1.0]),
           st.sampled_from([-0.93597, -0.82456, -0.78502, -0.13932, -0.11824, -0.01997, 0.06874, 0.63317, 0.69841, 1.0]),
           st.sampled_from([0.01128, 0.0214, 0.06192, 0.10991, 0.36111, 0.61745, 0.62745, 0.66364, 0.66938, 1.0]),
           st.sampled_from([-0.40446, -0.23404, -0.1209, -0.06879, -0.06178, 0.0119, 0.01639, 0.06637, 0.35, 0.58304]),
           st.sampled_from([-0.41912, -0.0485, 0.00282, 0.025, 0.03786, 0.1875, 0.19455, 0.23341, 0.38359, 0.93359]),
           st.sampled_from([-0.2803, -0.25712, -0.07542, -0.03353, -0.03279, 0.00141, 0.02083, 0.03125, 0.07018, 0.07297]),
           st.sampled_from([-0.6723, -0.07576, 0.01427, 0.12315, 0.12357, 0.16667, 0.34432, 0.61567, 0.66667, 1.0]),
           st.sampled_from([-0.59212, -0.1875, -0.11111, -0.0897, 0.18227, 0.26042, 0.30729, 0.33636, 0.51894, 0.5364]),
           st.sampled_from([-0.95489, -0.79085, -0.51685, -0.1335, 0.0106, 0.0625, 0.06425, 0.57836, 0.7592, 0.82077]),
           st.sampled_from([-0.97515, -0.3466, -0.34265, -0.3125, -0.22414, 0.1286, 0.14516, 0.15452, 0.38869, 0.73393]),
           st.sampled_from([-1.0, -0.04708, 0.01108, 0.18681, 0.19672, 0.21032, 0.22813, 0.48684, 0.50455, 0.64662]),
           st.sampled_from([-0.35985, -0.14754, -0.11344, -0.05206, -0.03516, -0.02956, 0.15018, 0.16391, 0.40455, 0.70982]),
           st.sampled_from([-0.60131, -0.34773, 0.00342, 0.10567, 0.3141, 0.36482, 0.4375, 0.63728, 0.7694, 1.0]),
           st.sampled_from([-0.85583, -0.63518, -0.54248, -0.42113, -0.28257, 0.00097, 0.05172, 0.05455, 0.18135, 0.78036]),
           st.sampled_from([-0.99219, -0.5183, -0.19792, -0.102, -0.06959, 0.0, 0.00759, 0.01894, 0.04325, 0.98484]),
           st.sampled_from([0.00238, 0.02514, 0.03669, 0.04762, 0.09728, 0.24582, 0.44961, 0.63496, 0.74359, 0.9273]),
           st.sampled_from([-0.73864, -0.47651, -0.05128, 0.01519, 0.01834, 0.16667, 0.51933, 0.61818, 0.83209, 0.8667]),
           st.sampled_from([-0.35734, -0.06571, -0.02282, -0.00559, 0.16516, 0.17803, 0.20301, 0.26585, 0.29961, 0.5625]),
           st.sampled_from([0.00838, 0.03161, 0.28604, 0.31148, 0.5375, 0.54446, 0.57273, 0.65493, 0.95703, 1.0]),
           st.sampled_from([-0.2549, -0.16667, -0.10234, -0.04802, -0.01326, -0.00391, -0.0005, 0.16418, 0.35696, 0.92236]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.62745, -0.51282, -0.51079, -0.17568, -0.07027, -0.02294, -0.00838, 0.01569, 0.14481]),
           st.sampled_from([0.0, 0.00026, 0.32463, 0.34545, 0.43182, 0.43481, 0.57143, 0.62958, 0.77884, 0.95982]),
           st.sampled_from([-0.75, -0.48485, -0.24561, -0.21901, -0.02612, 0.06977, 0.09615, 0.11146, 0.24242, 1.0]),
           st.sampled_from([-0.66667, -0.02568, -0.00761, 0.02399, 0.09611, 0.16288, 0.25682, 0.38352, 0.43137, 0.87757]),
           st.sampled_from([-0.29915, -0.06522, -0.02398, -0.02273, 0.02614, 0.05419, 0.13521, 0.14643, 0.22464, 0.90426]),
           st.sampled_from([-0.67553, -0.23186, -0.18494, 0.0, 0.00705, 0.07389, 0.14939, 0.25441, 0.34411, 0.47569]),
           st.sampled_from([-0.41667, -0.29091, -0.27451, -0.18826, 0.00325, 0.01997, 0.07895, 0.34732, 0.41913, 0.72831]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_17(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_17']['n_samples'] += 1
        self.data['tests']['test_17']['samples'].append(x_test)
        self.data['tests']['test_17']['y_expected'].append(y_expected[0])
        self.data['tests']['test_17']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.sampled_from([-1.0, -0.67935, -0.01864, 0.01667, 0.17188, 0.28409, 0.39286, 0.50932, 0.63816, 0.84557]),
           st.sampled_from([-1.0, -0.93996, -0.63636, -0.31818, -0.14754, -0.09524, 0.15564, 0.38889, 0.4, 0.81586]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.80553, -0.62766, -0.14545, -0.04119, -0.00846, 0.0, 0.04328, 0.15033, 0.15244, 0.80521]),
           st.sampled_from([-0.17778, -0.01923, 0.28354, 0.46643, 0.52012, 0.68182, 0.76001, 0.77941, 0.88444, 1.0]),
           st.sampled_from([-0.99265, -0.78502, -0.76378, -0.14286, -0.13932, -0.07843, 0.0, 0.06874, 0.27955, 1.0]),
           st.sampled_from([-0.30719, 0.00559, 0.25795, 0.35821, 0.36111, 0.625, 0.71875, 0.76087, 0.87719, 0.91473]),
           st.sampled_from([-0.63427, -0.10236, -0.03636, 0.00862, 0.04372, 0.04576, 0.0625, 0.1323, 0.71242, 0.95833]),
           st.sampled_from([0.23341, 0.29091, 0.38359, 0.43685, 0.47744, 0.86789, 0.89514, 0.93243, 0.97311, 1.0]),
           st.sampled_from([-0.62723, -0.3617, -0.16818, -0.15194, -0.01284, -0.01144, 0.0, 0.03125, 0.31385, 0.99557]),
           st.sampled_from([-1.0, -0.71875, -0.03158, -0.03125, -0.00885, 0.01975, 0.11538, 0.34848, 0.72131, 1.0]),
           st.sampled_from([-1.0, -0.2, -0.13725, -0.11111, -0.09852, -0.02282, 0.0, 0.19556, 0.58772, 0.63966]),
           st.sampled_from([-0.12, -0.09677, -0.01975, 0.0106, 0.01754, 0.06425, 0.08949, 0.56818, 0.57836, 0.68852]),
           st.floats(min_value=-1.0, max_value=-0.212072, allow_nan=False),
           st.sampled_from([-1.0, -0.55711, -0.53788, -0.37133, 0.17589, 0.21032, 0.46429, 0.72059, 0.95313, 0.95452]),
           st.sampled_from([-0.77206, -0.4627, -0.35985, -0.25084, -0.03516, -0.02956, -0.00537, 0.0, 0.75, 1.0]),
           st.sampled_from([-1.0, 0.0875, 0.22807, 0.33109, 0.46642, 0.54902, 0.63728, 0.84091, 0.88673, 0.92188]),
           st.sampled_from([-0.60294, -0.54467, -0.50424, -0.42113, -0.375, -0.14773, -0.00423, 0.0, 0.24903, 0.84922]),
           st.sampled_from([-1.0, -0.99219, -0.14343, -0.04575, 0.00048, 0.01894, 0.1167, 0.12727, 0.97068, 0.98484]),
           st.sampled_from([-0.83007, -0.53409, -0.453, -0.21081, -0.19608, 0.01478, 0.14792, 0.30745, 0.5081, 0.9273]),
           st.sampled_from([-0.93359, -0.73864, -0.47651, -0.34615, -0.29825, -0.18636, 0.01519, 0.02793, 0.11213, 0.29908]),
           st.sampled_from([-0.47594, -0.29508, -0.27778, -0.2619, -0.06571, -0.00229, 0.00888, 0.02956, 0.06895, 0.34766]),
           st.sampled_from([-0.91574, -0.90302, -0.71528, -0.58, -0.18856, -0.14803, -0.02862, 0.03513, 0.07853, 0.15103]),
           st.sampled_from([-0.59609, -0.04802, -0.01269, -0.00391, 0.00559, 0.16418, 0.19405, 0.35696, 0.85357, 0.90695]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.93939, -0.62745, -0.46809, -0.25288, -0.23864, -0.03578, -0.01186, 0.01569, 0.10976, 0.38848]),
           st.sampled_from([-0.71875, -0.27917, 0.03073, 0.06061, 0.125, 0.34545, 0.43182, 0.43481, 0.56863, 0.84496]),
           st.sampled_from([-0.48936, -0.31373, -0.24561, -0.06641, 0.03125, 0.09615, 0.42677, 0.50794, 0.69401, 0.81573]),
           st.sampled_from([-1.0, -0.75, -0.66667, 0.02586, 0.04433, 0.09611, 0.12727, 0.27869, 0.38352, 0.43137]),
           st.sampled_from([-1.0, -0.87354, -0.7817, -0.63156, -0.32317, -0.06288, 0.05419, 0.14643, 0.22464, 0.24086]),
           st.sampled_from([-0.67553, -0.19792, -0.19066, -0.04013, -0.00039, 0.1167, 0.17045, 0.2619, 0.81979, 1.0]),
           st.sampled_from([-1.0, -0.34375, -0.27292, -0.06314, 0.01997, 0.02381, 0.04586, 0.12755, 0.19551, 0.37409]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_18(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_18']['n_samples'] += 1
        self.data['tests']['test_18']['samples'].append(x_test)
        self.data['tests']['test_18']['y_expected'].append(y_expected[0])
        self.data['tests']['test_18']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.sampled_from([-0.65625, -0.5418, -0.205, -0.00641, 0.06404, 0.50932, 0.62121, 0.71253, 0.84783, 0.85271]),
           st.sampled_from([-0.86701, -0.31818, -0.09524, -0.00838, 0.01437, 0.05426, 0.15625, 0.4, 0.5782, 0.81586]),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.36156, -0.2381, -0.14545, 0.0, 0.04918, 0.12159, 0.12889, 0.15244, 0.85388]),
           st.floats(min_value=-1.0, max_value=0.789094, allow_nan=False),
           st.sampled_from([-1.0, -0.93597, -0.82456, -0.13932, -0.11824, -0.08238, -0.07843, -0.01117, 0.0, 0.64516]),
           st.sampled_from([0.06192, 0.25795, 0.30682, 0.34545, 0.34649, 0.35821, 0.60586, 0.71875, 0.76087, 0.8401]),
           st.sampled_from([-0.375, -0.2681, -0.23393, -0.06811, 0.00862, 0.1323, 0.31746, 0.35, 0.53819, 0.95041]),
           st.sampled_from([-1.0, 0.04517, 0.07595, 0.19455, 0.23341, 0.36285, 0.38359, 0.42803, 0.54478, 0.97078]),
           st.sampled_from([-1.0, -0.62723, -0.3617, -0.33333, -0.1517, -0.03353, -0.03279, -0.01953, 0.0, 0.00141]),
           st.sampled_from([-0.71875, -0.2902, -0.19091, -0.00885, 0.0106, 0.12357, 0.47368, 0.70067, 0.95824, 1.0]),
           st.sampled_from([-0.89375, -0.59043, -0.0184, 0.0, 0.16406, 0.275, 0.29091, 0.37333, 0.45489, 0.58772]),
           st.sampled_from([-0.61354, -0.51685, -0.1335, -0.09677, -0.08333, 0.17917, 0.23725, 0.28571, 0.56347, 0.7592]),
           st.floats(min_value=-0.212069, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.55711, -0.53788, -0.0875, -0.02753, 0.01108, 0.36585, 0.64662, 0.76929, 0.85701, 0.91921]),
           st.sampled_from([-0.83297, -0.35985, -0.2671, -0.02853, -0.00883, 0.0268, 0.08046, 0.15189, 0.40455, 0.70982]),
           st.sampled_from([-0.34773, -0.04023, 0.12121, 0.2381, 0.3141, 0.36482, 0.46429, 0.46642, 0.77979, 0.84]),
           st.sampled_from([-0.54467, -0.42292, -0.27461, -0.2069, -0.03289, 0.00097, 0.02381, 0.05455, 0.14375, 0.22115]),
           st.sampled_from([-1.0, -0.75, -0.74628, -0.25914, -0.20028, 0.01838, 0.06897, 0.24903, 0.6585, 0.98484]),
           st.sampled_from([-1.0, -0.21875, 0.00238, 0.02299, 0.04762, 0.14792, 0.38803, 0.5067, 0.68101, 0.9273]),
           st.sampled_from([-0.93359, -0.73864, -0.57576, -0.18636, 0.17021, 0.51933, 0.53448, 0.6203, 0.66667, 1.0]),
           st.sampled_from([-0.35734, -0.27778, -0.27083, 0.02956, 0.05456, 0.06895, 0.26585, 0.33176, 0.35668, 0.62349]),
           st.sampled_from([-0.18856, -0.10625, -0.1038, -0.02494, 0.03125, 0.24242, 0.45098, 0.5375, 0.65493, 0.8062]),
           st.sampled_from([-0.30063, -0.27273, -0.0005, 0.08974, 0.16418, 0.32727, 0.44828, 0.65635, 0.85714, 0.9625]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76667, -0.5813, -0.51282, -0.03578, 0.0, 0.01373, 0.01724, 0.15942, 0.22216, 0.35924]),
           st.sampled_from([-0.71875, -0.2234, -0.12879, 0.125, 0.32787, 0.43182, 0.43481, 0.56863, 0.77884, 1.0]),
           st.sampled_from([-0.66494, -0.61393, -0.31373, -0.21901, -0.06641, 0.00713, 0.18854, 0.67708, 0.69401, 0.81573]),
           st.sampled_from([-0.94375, -0.75, -0.66667, -0.09634, -0.02916, 0.0, 0.02399, 0.02586, 0.12727, 0.86136]),
           st.sampled_from([-1.0, -0.3254, -0.25946, -0.02273, -0.01854, 0.14643, 0.20477, 0.24086, 0.31128, 0.87317]),
           st.sampled_from([-0.18494, -0.13738, 0.07389, 0.1167, 0.17045, 0.25441, 0.2619, 0.40445, 0.90667, 1.0]),
           st.sampled_from([-0.9375, -0.59867, -0.55837, -0.15676, -0.09556, 0.0, 0.01997, 0.12755, 0.28794, 0.34732]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_19(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_19']['n_samples'] += 1
        self.data['tests']['test_19']['samples'].append(x_test)
        self.data['tests']['test_19']['y_expected'].append(y_expected[0])
        self.data['tests']['test_19']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.sampled_from([-0.64286, -0.205, -0.00641, 0.17188, 0.36876, 0.38521, 0.42708, 0.5984, 0.7381, 0.84783]),
           st.floats(min_value=-1.0, max_value=-0.680061, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.80553, -0.36156, -0.2381, -0.23067, 0.0, 0.15033, 0.26131, 0.3913, 0.82857, 1.0]),
           st.floats(min_value=0.789097, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.14286, -0.11111, -0.09473, -0.07843, -0.04444, -0.01997, 0.04598, 0.06874, 0.27955, 0.55735]),
           st.sampled_from([-1.0, -0.87097, 0.07979, 0.12586, 0.14706, 0.31889, 0.625, 0.68869, 0.80893, 0.97266]),
           st.sampled_from([-1.0, -0.63427, -0.40446, -0.10984, -0.06879, 0.0119, 0.01639, 0.31746, 0.47173, 0.55147]),
           st.sampled_from([-0.82143, -0.45663, -0.41912, -0.18599, 0.025, 0.04517, 0.42803, 0.50874, 0.89514, 0.93243]),
           st.sampled_from([-1.0, -0.62723, -0.16818, -0.10951, -0.05455, -0.03279, -0.00763, 0.02575, 0.07018, 0.1306]),
           st.sampled_from([-0.6723, -0.03736, -0.03125, -0.02439, 0.0106, 0.01427, 0.16667, 0.33333, 0.72131, 0.95824]),
           st.sampled_from([-0.1875, -0.13725, -0.11111, -0.0184, 0.07772, 0.16406, 0.19556, 0.29091, 0.58772, 1.0]),
           st.sampled_from([-0.51685, -0.12, -0.095, 0.01401, 0.08949, 0.56347, 0.68852, 0.88683, 0.91962, 1.0]),
           st.floats(min_value=-0.212069, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.55711, -0.53788, -0.51535, -0.0875, -0.03891, 0.37162, 0.59091, 0.64662, 0.67, 0.80521]),
           st.sampled_from([-0.81556, -0.11351, 0.02672, 0.03623, 0.08046, 0.16391, 0.21818, 0.46502, 0.59152, 0.70982]),
           st.sampled_from([-0.34773, -0.34187, 0.00342, 0.0047, 0.18993, 0.45076, 0.46642, 0.59167, 0.66818, 0.92188]),
           st.sampled_from([-1.0, -0.50424, -0.43025, -0.14773, -0.00423, 0.14375, 0.18135, 0.24903, 0.31408, 0.92949]),
           st.sampled_from([-1.0, -0.48, -0.35256, -0.102, -0.04749, 0.00048, 0.01838, 0.04325, 0.65574, 0.97068]),
           st.sampled_from([-0.31454, -0.19608, -0.09375, -0.03393, 0.01478, 0.02299, 0.07121, 0.24582, 0.38803, 0.85268]),
           st.sampled_from([-0.93359, -0.47651, -0.34128, -0.2, -0.18043, -0.00083, 0.02793, 0.29908, 0.61818, 0.94118]),
           st.sampled_from([-0.94118, -0.47594, -0.46801, -0.35734, 0.0, 0.05456, 0.20301, 0.24936, 0.26585, 1.0]),
           st.sampled_from([-0.90302, -0.84792, -0.71528, -0.58, -0.42568, -0.14803, -0.10625, 0.03021, 0.31148, 0.57273]),
           st.sampled_from([-0.53846, -0.43137, -0.16667, 0.0, 0.01854, 0.03876, 0.19405, 0.23387, 0.4375, 1.0]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-1.0, -0.40116, -0.24668, -0.0614, -0.01186, -0.00838, 0.01569, 0.14673, 0.22216, 0.625]),
           st.sampled_from([-0.47, -0.27917, -0.1904, -0.1517, -0.12879, 0.0, 0.00026, 0.02586, 0.32787, 0.55405]),
           st.sampled_from([-0.75625, -0.75, -0.05455, -0.01947, -0.01551, 0.06977, 0.185, 0.18854, 0.53846, 0.67708]),
           st.sampled_from([-0.94375, -0.66667, -0.09339, 0.02564, 0.06329, 0.18301, 0.39118, 0.69792, 0.86136, 0.87757]),
           st.sampled_from([-0.29915, -0.25946, 0.00015, 0.00108, 0.01149, 0.11778, 0.1286, 0.22464, 0.81007, 0.88428]),
           st.sampled_from([-0.40201, -0.18494, -0.13738, -0.08208, -0.04598, 0.0, 0.25441, 0.29091, 0.4918, 0.82895]),
           st.sampled_from([-0.27451, -0.09556, -0.06557, -0.02447, 0.0, 0.04586, 0.07895, 0.23913, 0.34732, 0.72831]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_20(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_20']['n_samples'] += 1
        self.data['tests']['test_20']['samples'].append(x_test)
        self.data['tests']['test_20']['y_expected'].append(y_expected[0])
        self.data['tests']['test_20']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.860798, allow_nan=False),
           st.floats(min_value=-0.680058, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.1438, -0.11326, -0.11057, -0.04494, 0.02306, 0.04296, 0.16996, 0.31543, 0.52312, 0.55756]),
           st.floats(min_value=0.789097, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.01604, 0.0826, 0.14756, 0.19152, 0.19498, 0.22108, 0.35222, 0.43388, 0.48561, 0.79784]),
           st.sampled_from([-0.03643, 0.2381, 0.56687, 0.74273, 0.77152, 0.8, 0.89009, 0.92908, 0.95584, 0.95947]),
           st.sampled_from([-0.49962, -0.36174, -0.13969, -0.125, -0.07643, -0.06561, -0.03913, 0.08091, 0.83548, 0.88744]),
           st.sampled_from([-0.52236, -0.45455, 0.03649, 0.21212, 0.50847, 0.90103, 0.91667, 0.92867, 0.93844, 0.99274]),
           st.sampled_from([0.0213, 0.04545, 0.05882, 0.06678, 0.08865, 0.09921, 0.21225, 0.26957, 0.59942, 0.75334]),
           st.sampled_from([-1.0, 0.30952, 0.48698, 0.57576, 0.71598, 0.85193, 0.85443, 0.85788, 0.9304, 0.93902]),
           st.sampled_from([-0.11111, -0.05202, 0.02966, 0.12252, 0.25283, 0.27515, 0.31217, 0.5156, 0.71996, 0.93778]),
           st.sampled_from([-1.0, -0.96986, -0.94053, 0.4313, 0.4921, 0.83121, 0.87937, 0.93094, 0.95947, 0.9917]),
           st.floats(min_value=-0.212069, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.66667, -0.6603, -0.65114, -0.38916, -0.38391, -0.12611, 0.70681, 0.88899, 0.94277, 0.95853]),
           st.sampled_from([-0.58019, -0.32268, -0.2952, -0.17531, -0.06109, -0.03191, 0.0028, 0.00773, 0.02744, 0.05713]),
           st.sampled_from([-0.00701, 0.51247, 0.55556, 0.7103, 0.71254, 0.75535, 0.78552, 0.82222, 0.83744, 0.90137]),
           st.sampled_from([-0.86063, -0.57579, -0.34275, -0.15633, 0.06618, 0.12727, 0.14439, 0.16265, 0.2194, 0.85952]),
           st.sampled_from([-0.81056, -0.46637, 0.33333, 0.37824, 0.39903, 0.47222, 0.49736, 0.52494, 0.91428, 0.99589]),
           st.sampled_from([-0.77778, -0.59839, -0.44072, -0.34691, -0.19095, -0.09091, -0.08333, -0.00299, 0.04901, 0.11053]),
           st.sampled_from([-0.74102, -0.43087, -0.14815, 0.28626, 0.5343, 0.59127, 0.80496, 0.91463, 0.94815, 0.99933]),
           st.sampled_from([-0.73855, -0.59691, -0.5743, -0.5122, -0.08079, 0.0, 0.00978, 0.16667, 0.21683, 0.82492]),
           st.sampled_from([-0.40552, 0.21645, 0.22792, 0.33333, 0.63333, 0.67016, 0.83333, 0.84932, 0.87234, 0.90071]),
           st.sampled_from([-1.0, -0.75533, -0.26356, -0.15675, -0.10474, -0.04451, 0.03968, 0.14079, 0.27226, 0.37381]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.43122, -0.10571, -0.0776, -0.05827, -0.01358, -0.01237, 0.00824, 0.04891, 0.12331, 0.13608]),
           st.sampled_from([-0.41223, -0.07162, 0.01033, 0.08425, 0.1903, 0.38228, 0.4764, 0.58644, 0.75638, 0.94118]),
           st.sampled_from([-0.45911, -0.29897, -0.27387, -0.26289, -0.05025, 0.01307, 0.05182, 0.05614, 0.32448, 0.80439]),
           st.sampled_from([0.14312, 0.15602, 0.36586, 0.45098, 0.53389, 0.58983, 0.64348, 0.74389, 0.83142, 0.92774]),
           st.floats(min_value=-1.0, max_value=-0.034862, allow_nan=False),
           st.sampled_from([0.08333, 0.13376, 0.39394, 0.41667, 0.65817, 0.87403, 0.91176, 0.93144, 0.98674, 0.98934]),
           st.sampled_from([-0.94823, -0.80975, -0.31954, -0.11717, -0.08571, -0.07027, 0.29942, 0.37465, 0.49831, 0.62658]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_21(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_21']['n_samples'] += 1
        self.data['tests']['test_21']['samples'].append(x_test)
        self.data['tests']['test_21']['y_expected'].append(y_expected[0])
        self.data['tests']['test_21']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.floats(min_value=-1.0, max_value=0.860798, allow_nan=False),
           st.floats(min_value=-0.680058, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.80553, -0.76087, -0.22703, -0.04119, -0.00846, 0.1, 0.15244, 0.21908, 0.5125, 0.82857]),
           st.floats(min_value=0.789097, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.76378, -0.70984, -0.67321, -0.11824, -0.11111, -0.01639, 0.01128, 0.04348, 0.07143, 0.69841]),
           st.sampled_from([-1.0, -0.03636, 0.01128, 0.05405, 0.06192, 0.23346, 0.34649, 0.36111, 0.9, 1.0]),
           st.sampled_from([-0.2681, -0.22222, -0.11765, -0.06879, -0.06178, 0.04576, 0.30921, 0.32899, 0.51042, 0.53819]),
           st.sampled_from([-0.18599, 0.03786, 0.07595, 0.31081, 0.54478, 0.81349, 0.82143, 0.85246, 0.86789, 0.91036]),
           st.sampled_from([-0.67743, -0.3871, -0.3617, -0.06609, -0.01284, 0.02083, 0.02575, 0.06439, 0.36194, 0.64]),
           st.sampled_from([-0.51504, -0.18494, 0.12357, 0.12766, 0.22807, 0.24514, 0.29091, 0.47368, 0.61567, 0.66667]),
           st.sampled_from([-1.0, -0.25342, -0.09852, 0.05307, 0.18227, 0.26042, 0.29091, 0.30729, 0.5364, 1.0]),
           st.sampled_from([-0.95489, -0.33656, -0.09677, 0.08949, 0.12292, 0.14645, 0.20635, 0.28571, 0.56347, 0.91962]),
           st.floats(min_value=-0.212069, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.98039, -0.55711, -0.09483, 0.72807, 0.76929, 0.80521, 0.81395, 0.91921, 0.95313, 0.95452]),
           st.sampled_from([-0.15625, -0.11765, -0.05206, -0.03516, -0.02956, -0.01818, 0.15476, 0.16391, 0.19444, 0.46502]),
           st.sampled_from([-0.50998, 0.0047, 0.12121, 0.18993, 0.2381, 0.3141, 0.33108, 0.4375, 0.73529, 0.7694]),
           st.sampled_from([-0.85583, -0.63518, -0.54467, -0.50424, -0.43025, -0.375, -0.27461, -0.14773, 0.07323, 0.12727]),
           st.sampled_from([-0.75, -0.74628, -0.5183, -0.25, -0.06959, -0.04575, 0.01894, 0.04372, 0.06897, 0.97068]),
           st.sampled_from([-1.0, -0.83007, -0.21875, 0.00075, 0.01478, 0.44961, 0.63496, 0.68101, 0.74359, 0.85268]),
           st.sampled_from([-1.0, -0.42572, -0.34128, -0.18043, 0.0, 0.11213, 0.61818, 0.8667, 0.8913, 1.0]),
           st.sampled_from([-0.46801, -0.29508, -0.27083, -0.06571, 0.00019, 0.35668, 0.45522, 0.62349, 0.90625, 1.0]),
           st.sampled_from([-0.91574, -0.18856, -0.1038, -0.02862, 0.00838, 0.05136, 0.15103, 0.31148, 0.65493, 1.0]),
           st.sampled_from([-0.53846, -0.46734, -0.27778, -0.10297, -0.0005, 0.00559, 0.06639, 0.32727, 0.4375, 0.5]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.93939, -0.57092, -0.51079, -0.40116, -0.07114, 0.10976, 0.17076, 0.22, 0.26256, 0.5]),
           st.sampled_from([-0.71875, -0.12879, -0.04779, 0.00026, 0.00564, 0.34545, 0.43481, 0.62958, 0.63811, 0.84496]),
           st.sampled_from([-0.38914, -0.21901, -0.06636, -0.02612, 0.0, 5e-05, 0.03125, 0.15436, 0.65625, 0.69401]),
           st.sampled_from([-1.0, -0.04585, -0.02568, 0.04447, 0.25682, 0.38352, 0.39118, 0.48927, 0.74468, 0.87757]),
           st.floats(min_value=-0.034859, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.19792, -0.19066, -0.13738, -0.08208, -8e-05, 0.0, 0.14939, 0.28919, 0.29091, 0.47569]),
           st.sampled_from([-1.0, -0.29091, -0.27292, -0.10837, -0.08978, -0.06865, 0.00325, 0.04586, 0.23913, 0.33611]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_22(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_22']['n_samples'] += 1
        self.data['tests']['test_22']['samples'].append(x_test)
        self.data['tests']['test_22']['y_expected'].append(y_expected[0])
        self.data['tests']['test_22']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=0.51, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([0.0]),
           st.floats(min_value=0.860801, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=-0.680058, max_value=1.0, exclude_min=True, allow_nan=False),
           st.floats(min_value=0.231542, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.13244, -0.03818, -0.02494, -0.00343, -0.0026, 0.00982, 0.02306, 0.341, 0.62233, 1.0]),
           st.floats(min_value=0.789097, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.58388, -0.21313, -0.11039, -0.08484, -0.06486, 0.01678, 0.0218, 0.03466, 0.24618, 0.3679]),
           st.sampled_from([-0.03643, 0.05832, 0.14177, 0.36667, 0.5013, 0.57282, 0.7176, 0.83675, 0.87947, 0.98602]),
           st.sampled_from([-0.07398, -0.07014, -0.0106, 0.01266, 0.02446, 0.0376, 0.08091, 0.08764, 0.5683, 0.77941]),
           st.sampled_from([0.45556, 0.56667, 0.62455, 0.66026, 0.88889, 0.90244, 0.94054, 0.94124, 0.94717, 0.98762]),
           st.sampled_from([-0.07092, -0.03243, 0.04102, 0.12131, 0.28778, 0.37685, 0.59565, 0.6132, 0.80441, 0.9173]),
           st.sampled_from([-0.87192, -0.67274, -0.20015, 0.47368, 0.6907, 0.70302, 0.81818, 0.83333, 0.97173, 0.97872]),
           st.sampled_from([-0.51874, -0.10598, -0.07287, -0.06544, -0.05202, 0.04545, 0.0722, 0.0733, 0.14706, 0.50949]),
           st.sampled_from([-0.96631, -0.4382, -0.3973, -0.16412, -0.06333, 0.69737, 0.78342, 0.90392, 0.93094, 0.99828]),
           st.floats(min_value=-0.212069, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.87474, -0.7035, -0.64706, 0.17959, 0.55621, 0.7855, 0.84211, 0.88899, 0.97173, 0.97613]),
           st.sampled_from([-0.55262, -0.31884, -0.28106, -0.14329, -0.04064, -0.00893, 0.06186, 0.06878, 0.51877, 1.0]),
           st.sampled_from([0.15196, 0.2707, 0.33333, 0.43204, 0.59615, 0.73827, 0.78324, 0.86473, 0.94207, 0.97901]),
           st.sampled_from([-0.86275, -0.32192, -0.12928, -0.10191, -0.08935, -0.03515, -0.01479, 0.04278, 0.04992, 0.62999]),
           st.sampled_from([-0.92254, -0.71225, 0.09709, 0.26957, 0.37824, 0.48571, 0.70674, 0.79016, 0.86043, 0.86763]),
           st.sampled_from([-0.0949, -0.08771, -0.08039, -0.04971, -0.02612, 0.11355, 0.1183, 0.14157, 0.62563, 0.68104]),
           st.sampled_from([-0.52838, -0.47412, 0.31898, 0.39159, 0.52778, 0.70331, 0.71002, 0.77041, 0.9519, 1.0]),
           st.sampled_from([-0.88591, -0.38095, -0.3523, -0.02778, 0.0224, 0.05882, 0.08041, 0.25728, 0.39597, 0.41176]),
           st.sampled_from([-0.75273, -0.68159, -0.50757, -0.15144, -0.11147, 0.21951, 0.26436, 0.3334, 0.61946, 0.9105]),
           st.sampled_from([-0.85703, -0.59954, -0.58042, -0.15049, -0.1284, -0.08341, -0.05484, 0.09756, 0.11348, 0.44199]),
           st.floats(min_value=0.999947, max_value=1.0, exclude_min=True, allow_nan=False),
           st.sampled_from([-0.31565, -0.09038, -0.04597, -0.02908, 0.026, 0.04452, 0.10168, 0.13608, 0.40753, 0.69435]),
           st.sampled_from([-1.0, -0.75406, 0.09694, 0.34635, 0.8292, 0.85536, 0.85611, 0.87568, 0.94681, 1.0]),
           st.sampled_from([-1.0, -0.61488, -0.57649, -0.43383, -0.21154, -0.21129, 0.09385, 0.12815, 0.27338, 0.42827]),
           st.sampled_from([-0.65754, 0.36364, 0.4397, 0.45114, 0.66624, 0.85198, 0.87492, 0.87893, 0.88424, 0.98564]),
           st.sampled_from([-0.73145, -0.47977, -0.31857, -0.31262, -0.15609, -0.09091, 0.01484, 0.35294, 0.47033, 0.66315]),
           st.sampled_from([-0.67699, -0.19609, 0.16655, 0.18353, 0.28985, 0.39394, 0.45361, 0.56286, 0.59407, 0.83867]),
           st.sampled_from([-1.0, -0.80975, -0.71273, -0.41573, -0.27201, -0.04307, 0.0, 0.01525, 0.27273, 0.42467]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_23(self, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34):
        x_test = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_23']['n_samples'] += 1
        self.data['tests']['test_23']['samples'].append(x_test)
        self.data['tests']['test_23']['y_expected'].append(y_expected[0])
        self.data['tests']['test_23']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
