class GenericHypothesisGenerationStrategy:
    pass


class SimpleTypeGenerationStrategy:

    def __init__(self, input_data, criteria):
        self.input_data = input_data
        self.criteria = criteria

    def get_method_decorators(self, classification_path, property_id):
        method_decorators = '\n'
        method_decorators += '    @given('

        for feature_values_specifier in classification_path.sample_values:
            if feature_values_specifier.min_is_derived_from_tree == False and feature_values_specifier.max_is_derived_from_tree == False:
                feature_id = feature_values_specifier.id
                class_id = classification_path.class_type
                samples = self.input_data.get_feature_values_by_class(feature_id, class_id, property_id)

                method_decorators += 'st.sampled_from(' + str(samples) + ')'

            elif feature_values_specifier.min_is_derived_from_tree == True and feature_values_specifier.max_is_derived_from_tree == False:
                feature_id = feature_values_specifier.id
                max_value = self.input_data.get_max_feature_value(feature_id)
                feature_values_specifier.max_value = max_value

                # set max_bva restrictions
                if feature_values_specifier.max_bva_value > feature_values_specifier.max_value:
                    feature_values_specifier.max_bva_is_derived_from_max_value = True
                    feature_values_specifier.max_bva_is_derived_from_training_data = True

                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            elif feature_values_specifier.min_is_derived_from_tree == False and feature_values_specifier.max_is_derived_from_tree == True:
                feature_id = feature_values_specifier.id
                min_value = self.input_data.get_min_feature_value(feature_id)
                feature_values_specifier.min_value = min_value

                # set min_bva restrictions
                if feature_values_specifier.min_bva_value < feature_values_specifier.min_value:
                    feature_values_specifier.min_bva_is_derived_from_min_value = True
                    feature_values_specifier.min_bva_is_derived_from_training_data = True

                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            else:
                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            if feature_values_specifier != classification_path.sample_values[-1]:
                method_decorators += ',\n           '

        method_decorators += ')'

        return method_decorators

    def get_method_signature(self, property_id, n_features):
        text_test_impl = '\n'
        text_test_impl += '    def test_' + str(property_id + 1) + '(self, '

        list_of_feature_id = [n for n in range(n_features)]

        for feature_id in list_of_feature_id:
            feature_name = 'feature_' + str(feature_id + 1)
            text_test_impl += feature_name

            if feature_id != list_of_feature_id[-1]:
                text_test_impl += ','
            if feature_id >= 0 and feature_id < list_of_feature_id[-1]:
                text_test_impl += ' '

        text_test_impl += '):\n'

        return text_test_impl

    def get_test_data(self, n_features, treat_flaot_as_decimal):
        FLOAT_AS_DECIMAL = treat_flaot_as_decimal
        list_of_feature_id = [n for n in range(n_features)]

        # list of feature values
        text_test_impl = '        x_test = ['

        for feature_id in list_of_feature_id:
            feature_name = 'feature_' + str(feature_id + 1)

            if FLOAT_AS_DECIMAL:
                text_test_impl += 'float(' + feature_name + ')'
            else:
                text_test_impl += feature_name

            if feature_id != list_of_feature_id[-1]:
                text_test_impl += ','
            if feature_id >= 0 and feature_id < list_of_feature_id[-1]:
                text_test_impl += ' '

        text_test_impl += ']\n'

        return text_test_impl

    def get_test_assertions(self, class_type, n_samples, property_id, is_for_experimentation):
        # y_expected
        test_assertions = '        y_expected = [' + str(class_type) + ']\n'

        # y_predicted
        test_assertions += '        y_predicted = self.model.predict([x_test]).tolist()\n'

        if is_for_experimentation:
            test_assertions += '\n'

            test_id = 'test_' + str(property_id + 1)

            test_assertions += '        self.data[\'tests\'][\'' + test_id + '\'][\'n_samples\'] += 1\n'
            test_assertions += '        self.data[\'tests\'][\'' + test_id + '\'][\'samples\'].append(x_test)\n'
            test_assertions += '        self.data[\'tests\'][\'' + test_id + '\'][\'y_expected\'].append(y_expected[0])\n'
            test_assertions += '        self.data[\'tests\'][\'' + test_id + '\'][\'y_predicted\'].append(y_predicted[0])\n'
            test_assertions += '\n'

        test_assertions += '        assert y_expected == y_predicted\n'

        return test_assertions


class TupleTypeGenerationStrategy:

    def __init__(self, input_data, criteria):
        self.input_data = input_data
        self.criteria = criteria

    def get_method_decorators(self, classification_path, property_id):
        method_decorators = '\n'
        method_decorators += '    @given(st.tuples('

        for feature_values_specifier in classification_path.sample_values:
            if feature_values_specifier.min_is_derived_from_tree == False and feature_values_specifier.max_is_derived_from_tree == False:
                feature_id = feature_values_specifier.id
                class_id = classification_path.class_type
                samples = self.input_data.get_feature_values_by_class(feature_id, class_id, property_id)

                method_decorators += 'st.sampled_from(' + str(samples) + ')'

            elif feature_values_specifier.min_is_derived_from_tree == True and feature_values_specifier.max_is_derived_from_tree == False:
                feature_id = feature_values_specifier.id
                max_value = self.input_data.get_max_feature_value(feature_id)
                feature_values_specifier.max_value = max_value

                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            elif feature_values_specifier.min_is_derived_from_tree == False and feature_values_specifier.max_is_derived_from_tree == True:
                feature_id = feature_values_specifier.id
                min_value = self.input_data.get_min_feature_value(feature_id)
                feature_values_specifier.min_value = min_value

                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            else:
                method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator(self.criteria)

            if feature_values_specifier != classification_path.sample_values[-1]:
                method_decorators += ', '

        method_decorators += '))'

        return method_decorators

    def get_method_signature(self, property_id, n_features):
        text_test_impl = '\n'
        text_test_impl += '    def test_' + str(property_id + 1) + '(self, generated_sample):\n'

        return text_test_impl

    def get_test_data(self, n_features, treat_flaot_as_decimal):
        FLOAT_AS_DECIMAL = treat_flaot_as_decimal

        text_test_impl = '        x_test = ['

        if FLOAT_AS_DECIMAL:
            text_test_impl += '[float(x) for x in generated_sample]\n'
        else:
            text_test_impl += 'x for x in generated_sample]\n'

        return text_test_impl

    def get_test_assertions(self, class_type, n_samples, property_id, is_for_experimentation):
        # y_expected
        test_assertions = '        y_expected = [' + str(class_type) + ']\n'

        # y_predicted
        test_assertions += '        y_predicted = self.model.predict([x_test]).tolist()\n'

        if is_for_experimentation:
            test_id = 'test_' + str(property_id + 1)
            test_assertions += '\n'
            test_assertions += '        self.executions[\'' + test_id + '\'][\'y_expected\'] = y_expected[0]\n'
            test_assertions += '        self.executions[\'' + test_id + '\'][\'n_samples\'] = self.executions[\'' + test_id + '\'][\'n_samples\'] + 1\n'
            test_assertions += '        self.executions[\'' + test_id + '\'][\'samples\'].append({\'sample\': x_test, \'y_predicted\': y_predicted[0]})\n'
            test_assertions += '\n'

        test_assertions += '        assert y_expected == y_predicted\n'

        return test_assertions


class ListGenerationStrategy:
    def __init__(self):
        pass

    def get_method_decorators(self, classification_path, property_id):
        method_decorators = '\n'
        method_decorators += '    @given(st.lists(st.tuples('

        for feature_values_specifier in classification_path.sample_values:
            method_decorators += 'st.' + feature_values_specifier.get_hypothesis_decorator()

            if feature_values_specifier != classification_path.sample_values[-1]:
                method_decorators += ', '

        method_decorators += '), min_size=10, max_size=10)'

        method_decorators += ')'

        return method_decorators

    def get_method_signature(self, property_id, n_features):
        text_test_impl = '\n'
        text_test_impl += '    def test_' + str(property_id + 1) + '(self, x_generated):\n'

        return text_test_impl

    def get_test_data(self, n_features, treat_flaot_as_decimal):
        list_of_feature_id = [n for n in range(n_features)]

        test_data = '        x_test = [[feature for feature in sample] for sample in x_generated]\n\n'

        return test_data

    def get_test_assertions(self, class_type, n_samples):
        # y_expected
        str_y_expected = ''
        for _ in range(n_samples):
            str_y_expected += str(class_type) + ','
        str_y_expected = str_y_expected[:-1]

        test_assertions = '        y_expected = [' + str_y_expected + ']\n'

        # y_predicted
        test_assertions += '        y_predicted = self.model.predict(x_test).tolist()'
        test_assertions += '\n'

        test_assertions += '        assert y_expected == y_predicted\n'

        return test_assertions
