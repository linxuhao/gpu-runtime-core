"""What is published stands alone: the runtime and the facade import only the standard library,
each other and the facade's declared requirements, so no deployment's modules leak in."""
import ast
from pathlib import Path
import sys
import unittest

CORE=Path(__file__).resolve().parents[1]
FACADE_REQUIREMENTS={'starlette','uvicorn','httpx'}


def imports(path):
    found=set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Import):found|={alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node,ast.ImportFrom) and node.level==0:found.add(node.module.split('.')[0])
    return found


class BoundaryTests(unittest.TestCase):
    def check(self, folder, allowed):
        own={p.stem for p in folder.glob('*.py')}
        for path in sorted(folder.glob('*.py')):
            with self.subTest(module=path.name):
                self.assertEqual(imports(path)-set(sys.stdlib_module_names)-own-allowed,set())

    def test_the_runtime_needs_nothing_but_the_standard_library(self):self.check(CORE/'runtime',set())

    def test_the_facade_needs_only_its_requirements(self):
        declared={line.split('==')[0].strip() for line in (CORE/'facade'/'requirements.txt').read_text().splitlines() if line.strip()}
        self.assertEqual(declared,FACADE_REQUIREMENTS)
        self.check(CORE/'facade',FACADE_REQUIREMENTS)


if __name__=='__main__':unittest.main()
