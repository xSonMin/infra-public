import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        script = ROOT / 'scripts' / 'launch.py'
        self.assertTrue(script.is_file(), 'GPU-scoped launch script must exist')
        spec = importlib.util.spec_from_file_location('launch', script)
        self.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launch)

    def test_rejects_gpu_outside_authorized_range(self):
        for values in ([4], [-1], [0, 4], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.launch.build_launch(ROOT, values, 'none', 30000)

    def test_rejects_duplicate_or_unsupported_parallelism(self):
        for values in ([0, 0], [0, 1, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.launch.build_launch(ROOT, values, 'none', 30000)

    def test_uuid_isolation_and_loopback_binding(self):
        command, env = self.launch.build_launch(ROOT, [2, 3], 'none', 30001)
        expected = ','.join(self.launch.ALLOWED_GPUS[i] for i in (2, 3))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], expected)
        self.assertEqual(command[command.index('--tp') + 1], '2')
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--port') + 1], '30001')
        self.assertNotIn('--speculative-algorithm', command)

    def test_virtual_environment_tools_are_on_path(self):
        _, env = self.launch.build_launch(ROOT, [0, 1], 'none', 30000)
        self.assertEqual(env['PATH'].split(__import__('os').pathsep)[0], str(ROOT / '.venv/bin'))

    def test_ordinal_mode_keeps_the_same_physical_gpu_allowlist(self):
        command, env = self.launch.build_launch(ROOT, [0, 1, 2, 3], 'none', 30000, 'ordinal')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(command[command.index('--tp') + 1], '4')

    def test_inventory_mismatch_prevents_ordinal_launch(self):
        expected = self.launch.ALLOWED_GPUS
        rows = '\n'.join(f'{i}, {expected[i]}' for i in (0, 1))
        self.launch.validate_inventory(rows, [0, 1])
        with self.assertRaises(ValueError):
            self.launch.validate_inventory(rows.replace(expected[1], 'GPU-different'), [0, 1])
        with self.assertRaises(ValueError):
            self.launch.validate_inventory(rows, [0, 1, 2, 3])

    def test_cuda_visibility_must_match_the_requested_physical_devices(self):
        self.assertTrue(hasattr(self.launch, 'validate_visible_uuids'))
        expected = [self.launch.ALLOWED_GPUS[i] for i in (2, 3)]
        self.launch.validate_visible_uuids(expected, [2, 3])
        for actual in (expected[::-1], expected[:1], expected + ['GPU-not-authorized']):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.launch.validate_visible_uuids(actual, [2, 3])

    def test_mtp_is_an_explicit_experiment(self):
        command, _ = self.launch.build_launch(ROOT, [0, 1, 2, 3], 'mtp', 30000)
        self.assertEqual(command[command.index('--speculative-algorithm') + 1], 'EAGLE')
        self.assertIn('--speculative-num-draft-tokens', command)

    def test_rejects_undefined_modes_and_invalid_ports(self):
        for spec, port in [('unknown', 30000), ('none', 0), ('none', 70000)]:
            with self.subTest(spec=spec, port=port), self.assertRaises(ValueError):
                self.launch.build_launch(ROOT, [0, 1], spec, port)

    def test_draft_capacity_tracks_topk_one_depth(self):
        for steps in (1, 2, 3):
            command, env = self.launch.build_launch(
                ROOT, [0, 1, 2, 3], 'mtp', 30000, speculative_steps=steps)
            self.assertEqual(command[command.index('--speculative-num-steps') + 1], str(steps))
            self.assertEqual(command[command.index('--speculative-num-draft-tokens') + 1], str(steps + 1))

    def test_rejects_unsupported_depth_or_plain_decode_speculation(self):
        for steps, mode in ((0, 'mtp'), (4, 'mtp'), (1, 'none')):
            with self.subTest(steps=steps, mode=mode), self.assertRaises(ValueError):
                self.launch.build_launch(ROOT, [0, 1, 2, 3], mode, 30000, speculative_steps=steps)

    def test_adaptive_config_limits_candidate_states_and_path(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = root / 'policy.json'
            config.write_text(json.dumps({'1': {'candidate_steps': [1, 2, 3]}}))
            command, _ = self.launch.build_launch(root, [0, 1, 2, 3], 'mtp', 30000, adaptive_config=config)
            self.assertIn('--speculative-adaptive', command)
            self.assertEqual(command[command.index('--speculative-adaptive-config') + 1], str(config))
            config.write_text(json.dumps({'1': {'candidate_steps': [0, 7]}}))
            with self.assertRaises(ValueError):
                self.launch.build_launch(root, [0, 1, 2, 3], 'mtp', 30000, adaptive_config=config)
            with self.assertRaises(ValueError):
                self.launch.build_launch(root, [0, 1, 2, 3], 'mtp', 30000, adaptive_config=root.parent / 'outside.json')


if __name__ == '__main__':
    unittest.main()
