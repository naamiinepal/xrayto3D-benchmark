import unittest
from XrayTo3DShape import get_model, name2arch

class TestArchitectureRegistry(unittest.TestCase):

    def test_off_the_shelf(self):
        models = name2arch.keys()
        for model_name in models:
            try:
                model = get_model(model_name,image_size=128,dropout=True)
                print(f'Succesfully loaded {model_name} from registry')
            except Exception as e:
                self.fail(f'Error:{e} {model_name} could not be loaded from registry')

if __name__ == '__main__':
    unittest.main()