from decimal import *
import numpy


def format_float(num):
    return numpy.format_float_positional(num, trim='-')


def count_decimal_places(float_value):
    if isinstance(float_value, (int, numpy.int64)):
        return 0

    str_float_value = str(float_value)

    if 'e' in str_float_value:
        float_value = format_float(float_value)
        str_float_value = str(float_value)

    substrings_float_value = str_float_value.split(".")
    n_decimal_places = len(substrings_float_value[1])

    return n_decimal_places


def max_decimal_places(float_value_1, float_value_2):
    n_decimal_places_value_1 = count_decimal_places(float_value_1)
    n_decimal_places_value_2 = count_decimal_places(float_value_2)

    if n_decimal_places_value_1 > n_decimal_places_value_2:
        return n_decimal_places_value_1
    else:
        return n_decimal_places_value_2


def min_decimal_places(float_value_1, float_value_2):
    n_decimal_places_value_1 = count_decimal_places(float_value_1)
    n_decimal_places_value_2 = count_decimal_places(float_value_2)

    if n_decimal_places_value_1 < n_decimal_places_value_2:
        return n_decimal_places_value_1
    else:
        return n_decimal_places_value_2


def max_decimal_places_in_list(float_values):
    max_decimal_places = 0

    for float_value in float_values:
        n_decimal_places = count_decimal_places(float_value)

        if n_decimal_places > max_decimal_places:
            max_decimal_places = n_decimal_places

    return max_decimal_places


def identify_type(min_value, max_value):
    if type(min_value) == type(max_value):
        if isinstance(min_value, int):
            return 'int'
        if isinstance(min_value, float):
            return 'float'
    else:
        return None


def create_decimal_with(n_decimal_places: int):
    if n_decimal_places == 0:
        return Decimal(0.0)
    if n_decimal_places >= 1:
        str_decimal = '.'
        for i in range(n_decimal_places - 1):
            str_decimal += '0'
        str_decimal += '1'

        num_decimal = Decimal(str_decimal)

        return num_decimal
    else:
        return None


def create_decimal(value):
    np_int_types = (numpy.int0, numpy.int8, numpy.int16, numpy.int32, numpy.int64)
    np_float_types = (numpy.float16, numpy.float32, numpy.float64)

    if isinstance(value, np_int_types):
        return Decimal(value.item())

    return Decimal(value)
