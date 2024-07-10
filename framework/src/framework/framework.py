import sys

from ml_model import DecisionTreeModel, get_training_and_test_data
from domain_model import Tree
from data import InputData, load_built_in_data
from class_generator import PytestClassGenerator
from cli import CLI, Settings
from report import Report


def main(args):
    cli = CLI()
    settings = cli.process_args()

    dt_model = DecisionTreeModel(settings.random_state)
    x_train, x_test, y_train, y_test = get_training_and_test_data(settings.data.data,
                                                                  settings.data.target,
                                                                  1.0,
                                                                  settings.random_state)
    
    input_data = InputData(x_train, x_test, y_train, y_test, settings.random_state)
    
    decision_tree = dt_model.get_model(x_train, y_train)
    
    if x_test is not None:
        y_pred = dt_model.predict(x_test)
        classificator = dt_model.classificator

        # validacao da arvore de decisao
        dt_model.validate(y_test, y_pred)

    # criacao da figura da arvore de decisao
    dt_model.create_image(settings.image_name, settings.export_directory)

    tree = Tree(decision_tree)
    tree.process_classification_paths(settings.float_as_decimal, settings.boundary_value_rate)
    tree.update_classification_paths(input_data)
    tree.update_feature_values_precision(input_data, settings.float_as_decimal)

    tree.process_classification_paths_bva(settings.boundary_value_rate, settings.random_state)
    tree.update_feature_values_precision_bva(input_data, settings.float_as_decimal)


    print(f"NUMERO PATHS: {len(tree.paths)}")
    print(f"NUMERO DE CLASSIFICATIONS PATHS: {len(tree.classification_paths)}")
    print(f"Classes: {tree.classes}")

    if settings.criteria == 'DTC':
        settings.conf_generation_strategy(input_data, settings.criteria)
        pytest_class_generator = PytestClassGenerator(settings.class_name,
                                                      settings.module_name,
                                                      tree.n_features,
                                                      tree.classification_paths,
                                                      input_data,
                                                      settings.target_ml_model,
                                                      settings.generation_strategy,
                                                      settings.experimentation,
                                                      settings.export_directory,
                                                      settings.float_as_decimal,
                                                      settings.export_target_model,
                                                      settings.n_samples_per_test,
                                                      settings.criteria)
        pytest_class_generator.generate_test_class()
    elif settings.criteria == 'BVA':
        settings.conf_generation_strategy(input_data, settings.criteria)
        pytest_class_generator = PytestClassGenerator(settings.class_name,
                                                      settings.module_name,
                                                      tree.n_features,
                                                      tree.classification_paths,
                                                      input_data,
                                                      settings.target_ml_model,
                                                      settings.generation_strategy,
                                                      settings.experimentation,
                                                      settings.export_directory,
                                                      settings.float_as_decimal,
                                                      settings.export_target_model,
                                                      settings.n_samples_per_test,
                                                      settings.criteria)
        pytest_class_generator.generate_test_class()
    else:
        criteria = 'dtc'
        settings.conf_generation_strategy(input_data, criteria)
        pytest_class_generator = PytestClassGenerator(settings.class_name,
                                                      settings.module_name,
                                                      tree.n_features,
                                                      tree.classification_paths,
                                                      input_data,
                                                      settings.target_ml_model,
                                                      settings.generation_strategy,
                                                      settings.experimentation,
                                                      settings.export_directory,
                                                      settings.float_as_decimal,
                                                      settings.export_target_model,
                                                      settings.n_samples_per_test,
                                                      criteria)
        pytest_class_generator.generate_test_class()

        criteria = 'bva'
        settings.conf_generation_strategy(input_data, criteria)
        pytest_class_generator = PytestClassGenerator(settings.class_name,
                                                      settings.module_name,
                                                      tree.n_features,
                                                      tree.classification_paths,
                                                      input_data,
                                                      settings.target_ml_model,
                                                      settings.generation_strategy,
                                                      settings.experimentation,
                                                      settings.export_directory,
                                                      settings.float_as_decimal,
                                                      settings.export_target_model,
                                                      settings.n_samples_per_test,
                                                      criteria)
        pytest_class_generator.generate_test_class()

    # Target Model Under Test
    x_train, x_test, y_train, y_test = get_training_and_test_data(settings.data.data,
                                                                  settings.data.target,
                                                                  settings.training_size_rate,
                                                                  settings.random_state)
    target_ml_model = dt_model.get_target_model(settings.target_ml_model, x_train, y_train)
    dt_model.dump_target_model(settings.target_ml_model.name(), target_ml_model, settings.export_directory)

    # Report Generation
    report = Report()

    data_report_info = settings.data.get_report_info()
    tree_report_info = tree.get_report_info()

    report.add_information(data_report_info)
    report.add_information(tree_report_info)

    report.export_csv(settings.export_directory)

    return


if __name__ == '__main__':
    sys.exit(main(sys.argv))
