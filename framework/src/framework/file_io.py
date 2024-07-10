import pandas as pd
from datetime import datetime
import os
import json


def read_file_as_df(source):
    df = pd.read_csv(source)

    return df


def read_file_as_np(source):
    df = pd.read_csv(source)
    df_as_np = df.to_numpy()

    return df_as_np


def create_export_directory(destination_directory, module_name):
    str_date_and_hour = __get_str_date_and_hour()

    destination_subdirectory = module_name + '_' + str_date_and_hour

    export_directory = os.path.join(destination_directory, destination_subdirectory)
    print(f"export_directory in creation: {export_directory}")
    print(f"export_directory in creation string: {str(export_directory)}")
    os.mkdir(export_directory)

    if os.path.isdir(export_directory):
        return export_directory
    else:
        return None


def create_complete_export_directory(destination_directory, module_name, float_as_decimal, taget_model_name):
    if float_as_decimal:
        generated_type = 'decimal'
    else:
        generated_type = 'float'

    str_date_and_hour = __get_str_date_and_hour()

    destination_subdirectory = module_name + '_' + generated_type + '_' + taget_model_name + '_' + str_date_and_hour

    export_directory = os.path.join(destination_directory, destination_subdirectory)
    print(f"export_directory in creation: {export_directory}")
    print(f"export_directory in creation string: {str(export_directory)}")
    os.mkdir(export_directory)

    if os.path.isdir(export_directory):
        return export_directory
    else:
        return None


def __get_str_date_and_hour():
    date_and_hour = datetime.now()
    str_date_and_hour = str(date_and_hour)

    str_date_and_hour = str_date_and_hour.replace('-', '_')
    str_date_and_hour = str_date_and_hour.replace(':', '_')
    str_date_and_hour = str_date_and_hour.replace('.', '_')
    str_date_and_hour = str_date_and_hour.replace(' ', '-')
    str_date_and_hour = str_date_and_hour[0:19]

    return str_date_and_hour


def get_fig_directory(image_name, export_directory):
    return os.path.join(export_directory, image_name)


def create_python_file(text_class, module_name, export_directory):
    python_file_name = module_name + '.py'
    file_path = os.path.join(export_directory, python_file_name)
    file = open(file_path, 'w')
    file.write(text_class)
    file.close()


def create_json_file(data_json, module_name, export_directory):
    json_file_name = module_name + '_training_data.json'
    file_path = os.path.join(export_directory, json_file_name)

    with open(file_path, mode='w') as json_file:
        json.dump(data_json, json_file)


def create_report_file(report_name, report_data, export_directory):
    file_path = os.path.join(export_directory, report_name)

    df = pd.DataFrame(data=report_data, dtype=str, index=[1])
    df.to_csv(file_path, index=False)


def get_train_data_file_path(module_name, export_directory):
    file_name = module_name + '_training_data.json'
    file_path = os.path.join(export_directory, file_name)

    str_file_path = str(file_path)
    str_file_path = str_file_path.replace('\\\\', '\\')
    str_file_path = str_file_path.replace('\\', '/')

    return str_file_path


def get_target_model_path(model_name, export_directory):
    file_name = model_name.lower() + '_model.joblib'
    file_path = os.path.join(export_directory, file_name)

    str_file_path = str(file_path)
    str_file_path = str_file_path.replace('\\\\', '\\')
    str_file_path = str_file_path.replace('\\', '/')

    return str_file_path


def get_target_model_name(model_name):
    file_name = model_name.lower() + '_model.joblib'
    return str(file_name)


def get_experiment_data_file_path(file_suffix, export_directory):
    file_name = file_suffix + 'experiment_data.json'
    file_path = os.path.join(export_directory, file_name)

    str_file_path = str(file_path)
    str_file_path = str_file_path.replace('\\\\', '\\')
    str_file_path = str_file_path.replace('\\', '/')

    return str_file_path


def get_experiment_data_file_name(file_suffix):
    file_name = file_suffix + 'experiment_data.json'
    return str(file_name)

def get_experiment_dt_image_path(export_directory):
    file_name = 'experiment_dt_image'
    file_path = os.path.join(export_directory, file_name)

    str_file_path = str(file_path)
    str_file_path = str_file_path.replace('\\\\', '\\')
    str_file_path = str_file_path.replace('\\', '/')

    return str_file_path
