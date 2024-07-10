from data import SampleValuesSpecifier, InputData
import util
import numpy as np


class Node:

    def __init__(self, id, feature, threshold, left_child_id, right_child_id, is_leaf_node, value):
        self.id = id
        self.parent = None
        self.feature = feature
        self.threshold = threshold
        self.left_child_id = left_child_id
        self.left_child = None
        self.right_child_id = right_child_id
        self.right_child = None
        self.is_left_child = None
        self.is_leaf_node = is_leaf_node
        self.class_type_id = None
        self.value = value

    def print_node(self):
        print(f'Id: {self.id}')
        if (self.parent != None):
            print(f'Parent Id: {self.parent.id}')
        else:
            print(f'Parent Id: {self.parent}')
        print(f'Feature: {self.feature}')
        print(f'Threshold: {self.threshold}')
        print(f'Left Child Id: {self.left_child_id}')
        print(f'Right Child Id: {self.right_child_id}')
        print(f'Is Left Child: {self.is_left_child}')
        print(f'Is leaf Node: {self.is_leaf_node}')
        print('--------------------')

    def print_node_and_childs(self):
        print(f'Id: {self.id}')
        if (self.parent != None):
            print(f'Parent Id: {self.parent.id}')
        else:
            print(f'Parent Id: {self.parent}')
        print(f'Feature: {self.feature}')
        print(f'Threshold: {self.threshold}')
        if (self.left_child_id != None):
            print(f'Left Child id: {self.left_child.id}')
        if (self.right_child_id != None):
            print(f'Right Child Id: {self.right_child.id}')
        print(f'Is Left Child: {self.is_left_child}')
        print(f'Is leaf Node: {self.is_leaf_node}')
        print(f'Class Type Id: {self.class_type_id}')
        print('--------------------')


class Tree:

    def __init__(self, decision_tree):
        self.__decision_tree = decision_tree
        self.__nodes = []
        self.__paths_of_nodes = []
        self.classification_paths = []

        self.leaf_nodes = None
        self.leaf_nodes = None
        self.n_leaf_nodes = None
        self.undefined_leaf_nodes = None
        self.n_undefined_leaf_nodes = None

        self.__process_nodes()
        self.__process_node_childs()

        self.leaf_nodes = self.__get_leaf_nodes()
        self.n_leaf_nodes = len(self.leaf_nodes)
        self.undefined_leaf_nodes = self.__get_undefined_leaf_nodes()
        self.n_undefined_leaf_nodes = len(self.undefined_leaf_nodes)

        self.__paths_of_nodes_temp = self.__process_paths2(self.__nodes[0])
        self.__paths_of_nodes = list()

        for temp_path_id in range(len(self.__paths_of_nodes_temp) - 1, -1, -1):
            self.__paths_of_nodes.append(self.__paths_of_nodes_temp[temp_path_id])


    @property
    def nodes(self):
        return self.__nodes

    @property
    def paths(self):
        return self.__paths_of_nodes

    @property
    def n_features(self):
        return self.__decision_tree.tree_.n_features

    @property
    def n_nodes(self):
        return len(self.__nodes)

    @property
    def classes(self):
        return self.__decision_tree.classes_

    @property
    def max_depth(self):
        return self.__decision_tree.max_depth

    def __get_leaf_nodes(self):
        leaf_nodes = []
        for node in (self.__nodes):
            if node.is_leaf_node:
                leaf_nodes.append(node)

        return leaf_nodes

    def __get_undefined_leaf_nodes(self):
        undefined_leaf_nodes = []

        for leaf_node in (self.leaf_nodes):
            value = leaf_node.value
            i_max_value = np.argmax(value)
            max_value = value[i_max_value]

            n_max_values = np.count_nonzero(value == max_value)

            if n_max_values > 1:
                undefined_leaf_nodes.append(leaf_node)

        return undefined_leaf_nodes

    def __get_class_type_id(self, node):
        class_type_id = None

        if node.is_leaf_node:
            class_type_id = self.__find_class_type_id(node)

        return class_type_id

    def __find_class_type_id(self, leaf_node):
        leaf_node_id = leaf_node.id
        tree = self.__decision_tree
        values = tree.tree_.value[leaf_node_id]
        one_dimension_values = values[0]

        return tree.classes_[np.argmax(one_dimension_values)]

    def get_list_of_class_type(self):
        class_types = [None for _ in range(len(self.__nodes))]
        print(class_types)
        leaf_nodes = self.__get_leaf_nodes()

        for node in self.__nodes:
            if node.is_leaf_node:
                class_type = self.__find_class_type_id(node)
                node_id = node.id
                class_types[node_id] = class_type
                print(f'Node: {node_id} Class-Type: {class_type}')

        return class_types

    def get_feature(self, i_node):
        tree = self.__decision_tree
        feature = None

        if tree.tree_.children_left[i_node] != tree.tree_.children_right[i_node]:
            feature = tree.tree_.feature[i_node]

        return feature

    def get_threshold(self, i_node):
        tree = self.__decision_tree
        threshold = None

        if tree.tree_.children_left[i_node] != tree.tree_.children_right[i_node]:
            threshold = tree.tree_.threshold[i_node]

        return threshold

    def get_left_child(self, i_node):
        tree = self.__decision_tree
        left_child = None

        if tree.tree_.children_left[i_node] != tree.tree_.children_right[i_node]:
            left_child = tree.tree_.children_left[i_node]

        return left_child

    def get_right_child(self, i_node):
        tree = self.__decision_tree
        right_child = None

        if tree.tree_.children_left[i_node] != tree.tree_.children_right[i_node]:
            right_child = tree.tree_.children_right[i_node]

        return right_child

    def get_value(self, i_node):
        tree = self.__decision_tree
        value = tree.tree_.value[i_node][0]

        return value

    def verify_leaf_node(self, left_child_id, right_child_id):
        if left_child_id is None and right_child_id is None:
            return True
        else:
            return False


    def __process_nodes(self):
        n_nodes = self.__decision_tree.tree_.node_count
        print(f"N_nodes: {n_nodes}")

        for i_node in range(n_nodes):
            print(f'Processando node: {i_node}')

            id = i_node
            feature = self.get_feature(id)
            threshold = self.get_threshold(id)
            left_child_id = self.get_left_child(id)
            right_child_id = self.get_right_child(id)
            is_leaf_node = self.verify_leaf_node(left_child_id, right_child_id)
            value = self.get_value(id)

            node = Node(id, feature, threshold, left_child_id, right_child_id, is_leaf_node, value)

            node_class_type_id = self.__get_class_type_id(node)
            node.class_type_id = node_class_type_id

            self.__nodes.append(node)

    def __process_node_childs(self):
        n_nodes = self.__decision_tree.tree_.node_count

        for i_node in range(n_nodes):
            print(f'Processando filhos do node: {i_node}')

            node = self.__nodes[i_node]

            left_child_id = node.left_child_id
            right_child_id = node.right_child_id

            if left_child_id is not None:
                node.left_child = self.__nodes[left_child_id]
                node.left_child.is_left_child = True
                node.left_child.parent = node

            if right_child_id is not None:
                node.right_child = self.__nodes[right_child_id]
                node.right_child.is_left_child = False
                node.right_child.parent = node

            # node.print_node_and_childs()

    def __process_paths(self):
        # list to store path
        root = self.__nodes[0]
        path = []
        self.__process_paths_recursive(root, path, 0)

    def __process_paths_recursive(self, root, path, path_len):

        if root is None:
            return

        if (len(path) > path_len):
            path[path_len] = root
        else:
            path.append(root)

        path_len = path_len + 1

        if root.left_child is None and root.right_child is None:
            self.__paths_of_nodes.append(path[0:path_len])
        else:
            # try for left and right subtree
            self.__process_paths_recursive(root.left_child, path, path_len)
            self.__process_paths_recursive(root.right_child, path, path_len)

    def __process_paths2(self, node, paths=None, current_path=None):

        if paths is None:
            paths = []
        if current_path is None:
            current_path = []

        current_path.append(node)

        if node.is_leaf_node:
            paths.append(current_path)
        else:
            self.__process_paths2(node.right_child, paths, list(current_path))
            self.__process_paths2(node.left_child, paths, list(current_path))
        return paths

    def print_paths(self):
        for path in self.__paths_of_nodes:
            print([node.id for node in path])

    def process_classification_paths(self, treat_flaot_as_decimal, boundary_value):

        for node_path in self.__paths_of_nodes:
            classification_path = ClassificationPath(node_path, self.n_features, treat_flaot_as_decimal)
            classification_path.calc()
            self.classification_paths.append(classification_path)

    def process_classification_paths_bva(self, boundary_value, random_state):
        # set boundary value analysis
        for classification_path in self.classification_paths:
            sample_specifier = classification_path.sample_values
            sample_specifier.set_boundary_values(boundary_value, random_state)
            # sample_specifier.print()
            # print('____________________________')

    def update_classification_paths(self, input_data: InputData):
        for property_id in range(len(self.classification_paths)):
            classification_path = self.classification_paths[property_id]
            for feature_values_specifier in classification_path.sample_values:
                if feature_values_specifier.min_is_derived_from_tree == True and feature_values_specifier.max_is_derived_from_tree == False:
                    feature_id = feature_values_specifier.id
                    max_value = input_data.get_max_feature_value(feature_id)
                    feature_values_specifier.max_value = max_value

                if feature_values_specifier.min_is_derived_from_tree == False and feature_values_specifier.max_is_derived_from_tree == True:
                    feature_id = feature_values_specifier.id
                    min_value = input_data.get_min_feature_value(feature_id)
                    feature_values_specifier.min_value = min_value

                feature_id = feature_values_specifier.id
                feature_type = input_data.get_feature_type(feature_id)
                feature_values_specifier.update_feature_type(feature_type)

    def update_classification_paths_old(self, input_data: InputData):
        for classification_path in self.classification_paths:
            # print('--------------')
            samples, samples_classification = input_data.get_samples(classification_path.sample_values)

            classification_path.sample_values.print()

            x_train_sample_values = input_data.create_x_train_sample_values(samples, self.n_features)
            x_train_sample_values.print()

            classification_path.sample_values.merge_sample_values(x_train_sample_values)
            # classification_path.sample_values.print()

    def update_feature_values_precision(self, input_data: InputData, float_as_decimal):
        decimal_places_in_each_feature = input_data.count_max_decimal_places_in_each_feature()

        for classification_path in self.classification_paths:
            classification_path.sample_values.update_decimal_places(decimal_places_in_each_feature, float_as_decimal)

    def update_feature_values_precision_bva(self, input_data: InputData, float_as_decimal):
        decimal_places_in_each_feature = input_data.count_max_decimal_places_in_each_feature()

        for classification_path in self.classification_paths:
            classification_path.sample_values.update_decimal_places_bva(decimal_places_in_each_feature, float_as_decimal)

    def get_report_info(self):
        report_info = dict()

        report_info['Nodes'] = self.n_nodes
        report_info['Leaf Nodes'] = self.n_leaf_nodes
        report_info['Undefined Leaf Nodes'] = self.n_undefined_leaf_nodes
        report_info['Class Labels'] = str(self.classes.tolist())
        report_info['Decision Paths'] = len(self.classification_paths)
        report_info['Max Depth'] = self.max_depth

        return report_info



class Path:

    def __init__(self, path):
        self.path = path
        self.class_type = None
        self.value = None
        self.leaf_node = path[-1]
        self.x_range = []
        self.x_range_complete = []

        # Set leaf node
        self.leaf_node.is_leaf_node = True

    def calc(self, n_features):
        self.x_range = [(float('-inf'), float('inf')) for _ in range(n_features)]

        for i in range(len(self.path) - 1, 0, -1):
            node = self.path[i]
            parent = node.parent
            feature = parent.feature
            threshold = parent.threshold

            lower_bound, upper_bound = self.x_range[feature]

            if (node.is_left_child == True):
                self.x_range[feature] = (lower_bound, min(upper_bound, threshold))
            else:
                self.x_range[feature] = (max(lower_bound, threshold), upper_bound)

    def print_path(self):
        print([node.id for node in self.path])

    def print_x_range(self):
        print(self.x_range)


class ClassificationPath:

    def __init__(self, path, n_features, treat_flaot_as_decimal):
        self.path = path
        self.value = None
        self.leaf_node = path[-1]
        self.class_type = self.leaf_node.class_type_id
        self.sample_values = SampleValuesSpecifier(n_features, treat_flaot_as_decimal)

        # Set leaf node
        self.leaf_node.is_leaf_node = True

    def calc(self):
        for i in range(len(self.path) - 1, 0, -1):
            node = self.path[i]
            parent = node.parent
            feature = parent.feature
            threshold = parent.threshold

            feature_values = self.sample_values.get_feature_values(feature)
            lower_bound, upper_bound = feature_values.get_tuple_values()

            if (node.is_left_child == True):
                feature_values.update_values(lower_bound, threshold)
                feature_values.max_is_exclusive = False
                feature_values.max_is_derived_from_tree = True
            else:
                feature_values.update_values(threshold, upper_bound)
                feature_values.min_is_exclusive = True
                feature_values.min_is_derived_from_tree = True

    def update_sample_values_precision(self):
        pass

    def print_path(self):
        print([node.id for node in self.path])
