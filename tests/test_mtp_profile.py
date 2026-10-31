import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]

class MtpProfileTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/mtp_profile.py'
        self.assertTrue(path.exists(), 'Trace validation helper is required')
        spec = importlib.util.spec_from_file_location('mtp_profile', path)
        self.profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.profile)

    def test_gpu_events_and_stage_markers_are_distinct(self):
        events = [dict(name='verify', cat='user_annotation', ph='X', ts=0, dur=10, tid=1),
                  dict(name='compute', cat='kernel', ph='X', ts=3, dur=4, args={'stream': 1}),
                  dict(name='ncclKernel_AllReduce', cat='kernel', ph='X', ts=5, dur=4, args={'stream': 2})]
        row = self.profile.trace_summary(events)
        self.assertEqual(row['gpu_busy_union_us'], 6)
        self.assertEqual(row['communication_labeled_union_us'], 4)
        self.assertEqual(row['stage_cpu_wall_us']['verify'], 10)
        self.assertEqual(row['kernel_count'], 2)
        self.assertTrue(row['attribution_limited'])

    def test_cpu_only_trace_cannot_pass_gpu_validation(self):
        with self.assertRaises(ValueError):
            self.profile.trace_summary([dict(name='draft', cat='user_annotation', ph='X', ts=0, dur=10)])

    def test_profiled_load_has_no_second_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            calls = []
            class FakeStudy:
                def __init__(self): self.run = run
                def bench(self, phase, *args, **kwargs):
                    calls.append((phase, kwargs))
                    if phase == 'profile-load':
                        folder = run / 'profiles/short-c1-mtp'
                        (folder / 'capture-start.json').write_text(json.dumps({'status':200}))
                        for rank in range(4):
                            events = [dict(name='verify', cat='user_annotation', ph='X', ts=0, dur=10),
                                      dict(name='compute', cat='kernel', ph='X', ts=1, dur=5)]
                            (folder / f'short-TP-{rank}.trace.json').write_text(json.dumps({'traceEvents':events}))
                def event(self, *args, **kwargs): pass
            with patch.object(self.profile, 'post', return_value={'status':200}):
                first = self.profile.capture(FakeStudy(), 'short', 512, 128, 1, 'mtp')
                count_before_reuse = len(calls)
                second = self.profile.capture(FakeStudy(), 'short', 512, 128, 1, 'mtp')
                self.assertEqual(len(calls), count_before_reuse)
                self.assertEqual(first, second)
            profile_call = next(kwargs for phase, kwargs in calls if phase == 'profile-load')
            self.assertEqual(profile_call.get('warmups'), 0)

    def test_gpu_launch_correlation_maps_asynchronous_stage_work(self):
        events = [
            dict(name='draft', cat='user_annotation', ph='X', ts=0, dur=10, pid=1, tid=1),
            dict(name='verify', cat='user_annotation', ph='X', ts=10, dur=10, pid=1, tid=1),
            dict(name='cudaLaunchKernel', cat='cuda_runtime', ph='X', ts=2, dur=1,
                 pid=1, tid=1, args={'correlation':11}),
            dict(name='cudaLaunchKernel', cat='cuda_runtime', ph='X', ts=12, dur=1,
                 pid=1, tid=1, args={'correlation':12}),
            dict(name='draft_kernel', cat='kernel', ph='X', ts=25, dur=5,
                 args={'correlation':11, 'stream':1}),
            dict(name='ncclKernel_AllReduce', cat='kernel', ph='X', ts=30, dur=4,
                 args={'correlation':12, 'stream':1}),
            dict(name='unlinked', cat='kernel', ph='X', ts=34, dur=1, args={'stream':1}),
        ]
        row = self.profile.trace_summary(events)
        self.assertIn('gpu_stage_union_us', row)
        self.assertEqual(row['gpu_stage_union_us']['draft'], 5)
        self.assertEqual(row['gpu_stage_union_us']['verify'], 4)
        self.assertIn('communication_stage_union_us', row)
        self.assertEqual(row['communication_stage_union_us']['verify'], 4)
        self.assertEqual(row['stage_annotation_count']['draft'], 1)
        self.assertEqual(row['gpu_unknown_union_us'], 1)
        self.assertEqual(row['mapping_method'], 'cpu-launch-correlation')

    def test_existing_model_runner_marker_is_verify_forward_only(self):
        events = [
            dict(name='step[VERIFY bs=8 forward_id=123]', cat='user_annotation',
                 ph='X', ts=0, dur=10, pid=1, tid=1),
            dict(name='cudaGraphLaunch', cat='cuda_runtime', ph='X', ts=2, dur=1,
                 pid=1, tid=1, args={'correlation':1}),
            dict(name='compute', cat='kernel', ph='X', ts=20, dur=6,
                 args={'correlation':1}),
        ]
        row = self.profile.trace_summary(events)
        self.assertIn('verify_forward', row['gpu_stage_union_us'])
        self.assertEqual(row['gpu_stage_union_us']['verify_forward'], 6)
        self.assertEqual(row['forward_steps'][0]['batch_size'], 8)

    def test_cuda_graph_gpu_annotations_are_not_cpu_forward_calls(self):
        events = [
            dict(name='step[VERIFY bs=1 forward_id=5]', cat='user_annotation',
                 ph='X', ts=0, dur=10, pid=100, tid=100),
            dict(name='step[VERIFY bs=1 forward_id=5]', cat='gpu_user_annotation',
                 ph='X', ts=20, dur=10, pid=0, tid=7),
            dict(name='compute', cat='kernel', ph='X', ts=20, dur=5),
        ]
        row = self.profile.trace_summary(events)
        self.assertEqual(len(row['forward_steps']), 1)
        self.assertEqual(row['stage_annotation_count']['verify_forward'], 1)
        self.assertEqual(row['stage_cpu_wall_us']['verify_forward'], 10)

if __name__ == '__main__':
    unittest.main()
