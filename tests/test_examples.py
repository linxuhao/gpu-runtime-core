"""The example configuration is one the runtime accepts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from gpu_runtime import load_config

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'


class ExampleTests(unittest.TestCase):
    def test_the_example_config_loads(self):
        config=load_config(EXAMPLES/'gpu-config.json')
        self.assertEqual(sorted(config['tenants']),['chat','embed'])
        self.assertEqual(config['tenants']['chat']['models'],['qwen3-8b'])


if __name__=='__main__':unittest.main()
