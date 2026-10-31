import sys,unittest,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

class TimelineMetricTests(unittest.TestCase):
 def test_payload_is_frozen_and_trace_output_stays_under_run(self):
  from timeline_metrics import profile_payload
  run=ROOT/'results/timeline-study-2026-10-31/run-fixture'
  p=profile_payload(run,'profile-sync8')
  self.assertEqual(p['num_steps'],48);self.assertEqual(p['activities'],['CPU','GPU'])
  self.assertFalse(p['with_stack']);self.assertFalse(p['record_shapes']);self.assertFalse(p['merge_profiles'])
  self.assertTrue(p['detailed_annotations'])
  self.assertEqual(Path(p['output_dir']).parent,run/'profiles')
  with self.assertRaises(ValueError):profile_payload(run,'../../escape')
 def fixture(self):
  return [
   dict(name='step[EXTEND bs=2 toks=4096 c_sq=2048 forward_id=1]',cat='user_annotation',ph='X',ts=0,dur=10,pid=1,tid=1),
   dict(name='step[DRAFT bs=2 toks=4096]',cat='user_annotation',ph='X',ts=1,dur=2,pid=1,tid=1),
   dict(name='cudaGraphLaunch',cat='cuda_runtime',ph='X',ts=4,dur=1,pid=1,tid=1,args={'correlation':11}),
   dict(name='step[VERIFY bs=8 forward_id=2]',cat='user_annotation',ph='X',ts=20,dur=10,pid=1,tid=1),
   dict(name='cudaGraphLaunch',cat='cuda_runtime',ph='X',ts=22,dur=1,pid=1,tid=1,args={'correlation':12}),
   dict(name='step[VERIFY bs=7 forward_id=3]',cat='user_annotation',ph='X',ts=50,dur=10,pid=1,tid=1),
   dict(name='cudaGraphLaunch',cat='cuda_runtime',ph='X',ts=52,dur=1,pid=1,tid=1,args={'correlation':13}),
   dict(name='step[EXTEND bs=2 toks=4096]',cat='gpu_user_annotation',ph='X',ts=80,dur=7),
   dict(name='prefill_kernel',cat='kernel',ph='X',ts=80,dur=7,args={'correlation':11}),
   dict(name='verify_kernel',cat='kernel',ph='X',ts=100,dur=5,args={'correlation':12}),
   dict(name='ncclKernel_AllReduce',cat='kernel',ph='X',ts=110,dur=4,args={'correlation':13})]
 def test_target_counts_exclude_draft_and_gpu_duplicate_annotations(self):
  from timeline_metrics import summarize_trace
  r=summarize_trace(self.fixture())
  self.assertEqual(r['target_prefill_calls'],1);self.assertEqual(r['target_verify_calls'],2)
  self.assertEqual(r['prefill_batch_histogram'],{'2':1})
  self.assertEqual(r['verify_batch_histogram'],{'7':1,'8':1})
  self.assertEqual(r['target_prefill_input_tokens'],4096)
  self.assertEqual(r['verify_start_gap_median_us'],30)
 def test_gpu_activity_uses_launch_correlation_not_cpu_timestamp_overlap(self):
  from timeline_metrics import summarize_trace
  r=summarize_trace(self.fixture())
  self.assertEqual(r['gpu_stage_union_us']['prefill_forward'],7)
  self.assertEqual(r['gpu_stage_union_us']['verify_forward'],9)
  self.assertEqual(r['communication_stage_union_us']['verify_forward'],4)
 def test_missing_gpu_or_missing_target_stage_cannot_pass(self):
  from timeline_metrics import summarize_trace
  with self.assertRaises(ValueError):summarize_trace([dict(name='step[VERIFY bs=1]',cat='user_annotation',ph='X',ts=0,dur=1)])
  with self.assertRaises(ValueError):summarize_trace([dict(name='k',cat='kernel',ph='X',ts=0,dur=1)])
 def test_diagnostic_load_is_fixed_and_not_added_to_old_formal_protocol(self):
  from timeline_metrics import validate_load
  validate_load(2048,128,4,120,'periodic');validate_load(512,128,2,4,'periodic')
  for args in [(2048,128,7,120,'periodic'),(512,128,4,120,'periodic'),(2048,512,4,120,'periodic')]:
   with self.assertRaises(ValueError):validate_load(*args)
 def test_client_origin_trigger_and_trace_limit_are_finite(self):
  from timeline_metrics import trigger_due,validate_trace_size
  self.assertFalse(trigger_due(100,139.99));self.assertTrue(trigger_due(100,140))
  validate_trace_size([10,20])
  with self.assertRaises(ValueError):validate_trace_size([1024**3+1])

 def test_sampling_wait_consumes_the_same_client_deadline(self):
  from timeline_metrics import remaining_case_seconds
  self.assertEqual(remaining_case_seconds(100,100,120),300)
  self.assertEqual(remaining_case_seconds(100,250,120),150)
  self.assertEqual(remaining_case_seconds(100,450,120),0)
  with self.assertRaises(ValueError):remaining_case_seconds(100,90,120)
