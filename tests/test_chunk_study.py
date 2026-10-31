import sys,unittest,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

class ChunkStudyTests(unittest.TestCase):
 def runpath(self):return ROOT/'results/chunk-study-2026-10-31/run-fixture'
 def test_only_native_chunk_size_differs_between_two_server_commands(self):
  from chunk_study import command_for
  a,ae=command_for(ROOT,self.runpath(),'chunk2048');b,be=command_for(ROOT,self.runpath(),'chunk1024')
  i=a.index('--chunked-prefill-size')+1
  self.assertEqual(a[i],'2048');self.assertEqual(b[i],'1024')
  aa=a[:];bb=b[:];aa[i]=bb[i]='CHUNK';self.assertEqual(aa,bb)
  self.assertEqual(a[a.index('--speculative-num-steps')+1],'1')
  self.assertEqual(a[a.index('--prefill-decode-interval')+1],'8')
  self.assertIn('INFRA_WAIT_BUDGET_CONFIG',ae);self.assertIn('INFRA_WAIT_BUDGET_CONFIG',be)
 def test_restore_is_exact_original_without_experiment_environment(self):
  from chunk_study import command_for,clean_environment
  import launch
  cmd,env=command_for(ROOT,self.runpath(),'stable3',restore=True)
  self.assertEqual(cmd,launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)[0])
  self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',env)
  dirty={'INFRA_CHUNK_CASE':'x','INFRA_WAIT_BUDGET_CONFIG':'x','PYTHONPATH':str(self.runpath()/'runtime')+':'+str(ROOT/'results/timeline-study-2026-10-31/run-old/runtime')+':/keep'}
  self.assertEqual(clean_environment(ROOT,dirty),{'PYTHONPATH':'/keep'})
 def test_unknown_label_and_run_fail_before_server_actions(self):
  from chunk_study import command_for,study_directory
  with self.assertRaises(ValueError):command_for(ROOT,self.runpath(),'chunk512')
  with self.assertRaises(ValueError):study_directory(ROOT,ROOT/'results/timeline-study-2026-10-31/run-fixture')
 def test_initialization_creates_required_directories(self):
  from chunk_study import ChunkStudy
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run=root/'results/chunk-study-2026-10-31/run-fixture';s=ChunkStudy(root,run)
   for f in ['cases','case-configs','profiles','client-logs','server-logs','patch-receipts']:
    self.assertTrue((s.run/f).is_dir())
 def test_automatic_profile_completion_requires_exact_four_nonempty_gzip_traces(self):
  from chunk_metrics import completed_profile_files
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)
   self.assertFalse(completed_profile_files(folder))
   for rank in range(4):(folder/f'fixture-TP-{rank}.trace.json.gz').write_bytes(b'fixture')
   self.assertTrue(completed_profile_files(folder))
   (folder/'extra-TP-0.trace.json.gz').write_bytes(b'fixture')
   self.assertFalse(completed_profile_files(folder))
 def test_target_chunk_validation_uses_actual_prefill_token_annotations(self):
  from chunk_metrics import validate_chunk_trace
  validate_chunk_trace({'prefill_token_histogram':{'1024':20},'target_prefill_calls':20},1024)
  validate_chunk_trace({'prefill_token_histogram':{'2048':12},'target_prefill_calls':12},2048)
  with self.assertRaises(ValueError):validate_chunk_trace({'prefill_token_histogram':{'2048':12},'target_prefill_calls':12},1024)
  with self.assertRaises(ValueError):validate_chunk_trace({'prefill_token_histogram':{},'target_prefill_calls':0},1024)
 def test_worker_records_body_hash_without_changing_native_stream_parser(self):
  p=ROOT/'scripts/chunk_worker.py';s=p.read_text() if p.exists() else ''
  self.assertIn('client-origin.json',s);self.assertIn('timeout=45',s)
  self.assertNotIn('start_profile',s)
