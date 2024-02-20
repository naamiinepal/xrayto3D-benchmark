# code from https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/utils/registry.py

import importlib

__all__ = [
    'ARCHITECTURES'
]

class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f'Value of a Registry must be a callable!\nValue:{value}')
        if key is None:
            key = value.__name__
        if key in self._dict:
            print(f'Key {key} already in registry {self._name}')
        self._dict[key] = value
    
    def register(self, target):
        """Decorator to register a function or class"""
        def add(key,value):
            self[key] = value
            return value
        if callable(target):
            # @reg.register
            return add(None, target)
        return lambda x: add(target, x)
    
    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()
    
ARCHITECTURES = Register('architectures')

def _handle_errors(errors):
    "log error and reraise errors during import"
    if not errors:
        return
    for name, err in errors:
        print(f'Module {name} import failed: {err}')

ALL_MODULES = [
('XrayTo3DShape.architectures',['onedconcat','twodpermuteconcat','twodpermuteconcatmultiscale','autoencoder'])
]

def import_all_modules_for_register():
    all_modules = ALL_MODULES
    errors = []
    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != '':
                    full_name = base_dir+'.'+name
                else:
                    full_name = name
                importlib.import_module(full_name)
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)