import pandas as pd
import file_io


class Report:

    def __init__(self):
        self.data_set_name = None
        self.n_samples = None
        self.n_attributes = None
        self.n_features = None
        self.n_classes = None
        self.n_samples_per_class = None
        self.class_labels = None
        self.n_nodes = None
        self.n_decision_nodes = None
        self.n_leaf_nodes = None
        self.n_undefined_leaf_nodes = None
        self.decision_paths = None
        self.max_depth = None

        self.as_dict = {
            'Samples': None,
            'Attributes': None,
            'Features': None,
            'Classes': None,
            'Samples per Class': None,
            'Class Labels': None,
            'Nodes': None,
            'Decision Nodes': None,
            'Leaf Nodes': None,
            'Undefined Leaf Nodes': None,
            'Decision Paths': None,
        }

    def add_information(self, report_info: dict):

        for key, value in report_info.items():
            # print(key)
            if key == 'Data Set':
                self.data_set_name = report_info[key]
            if key == 'Samples':
                self.n_samples = report_info[key]
            if key == 'Attributes':
                self.n_attributes = report_info[key]
            if key == 'Features':
                self.n_features = report_info[key]
            if key == 'Classes':
                self.n_classes = report_info[key]
            if key == 'Samples per Class':
                self.n_samples_per_class = report_info[key]
            if key == 'Class Labels':
                self.class_labels = report_info[key]
            if key == 'Nodes':
                self.n_nodes = report_info[key]
            if key == 'Decision Nodes':
                self.n_decision_nodes = report_info[key]
            if key == 'Leaf Nodes':
                self.n_leaf_nodes = report_info[key]
            if key == 'Undefined Leaf Nodes':
                self.n_undefined_leaf_nodes = report_info[key]
            if key == 'Decision Paths':
                self.decision_paths = report_info[key]
            if key == 'Max Depth':
                self.max_depth = report_info[key]

    def to_dict(self):

        self.as_dict['Samples'] = self.n_samples
        self.as_dict['Attributes'] = self.n_attributes
        self.as_dict['Features'] = self.n_features
        self.as_dict['Classes'] = self.n_classes
        self.as_dict['Samples per Class'] = self.n_samples_per_class
        self.as_dict['Class Labels'] = self.class_labels
        self.as_dict['Nodes'] = self.n_nodes
        self.as_dict['Decision Nodes'] = self.n_nodes - self.n_leaf_nodes
        self.as_dict['Leaf Nodes'] = self.n_leaf_nodes
        self.as_dict['Undefined Leaf Nodes'] = self.n_undefined_leaf_nodes
        self.as_dict['Decision Paths'] = self.decision_paths
        
        return self.as_dict

    def export_csv(self, export_directory):
        report_dict = self.to_dict()

        file_io.create_report_file('GeneralReport.csv', report_dict, export_directory)
