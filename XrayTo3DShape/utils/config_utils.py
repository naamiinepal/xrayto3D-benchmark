"""utils for working with dicts"""
from collections import ChainMap
import copy


def substitute_value_in_nested_dict(key, template_dict, substitute_val):
    """use recursion to find a key in a nested dict and update value"""
    if hasattr(template_dict, "items"):  # dict-like object
        for k, v in template_dict.items():
            if k == key:
                template_dict[k] = substitute_val
            if isinstance(v, dict):
                substitute_value_in_nested_dict(key, v, substitute_val)


def update_multiple_key_values_in_nested_dict(target_dict: dict, source_dict: dict):
    """update nested dict"""
    updated_dict = copy.deepcopy(target_dict)
    for k, v in source_dict.items():
        substitute_value_in_nested_dict(k, updated_dict, v)
    return updated_dict


def merge_dicts(dict1, dict2):
    """simpler naming"""
    return ChainMap(dict1, dict2)
