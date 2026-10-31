import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StopTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/stop.py'
        self.assertTrue(path.exists(), 'A process-scoped stop command is required')
        spec = importlib.util.spec_from_file_location('stop', path)
        self.stop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.stop)
        self.command = str(ROOT / '.venv/bin/python').encode() + b'\0-m\0sglang.launch_server\0'

    def test_accepts_only_the_project_server_process(self):
        self.stop.validate_target(ROOT, self.command, os.getuid())

    def test_rejects_another_user(self):
        with self.assertRaises(PermissionError):
            self.stop.validate_target(ROOT, self.command, os.getuid() + 1)

    def test_rejects_unrelated_process_or_another_project(self):
        for command in (b'/usr/bin/python3\0-m\0http.server\0',
                        b'/elsewhere/.venv/bin/python\0-m\0sglang.launch_server\0'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.stop.validate_target(ROOT, command, os.getuid())


if __name__ == '__main__':
    unittest.main()
