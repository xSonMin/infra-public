import sys,unittest,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

class TimelineStudyTests(unittest.TestCase):
 def runpath(self):return ROOT/'results/timeline-study-2026-10-31/run-fixture'
 def test_all_diagnostic_configs_have_same_mtp_depth_and_cpu_sync(self):
  from timeline_study import command_for
  for label,interval in [('sync0',0),('sync8',8),('budget100',8)]:
   command,env=command_for(ROOT,self.runpath(),label)
   self.assertEqual(command[command.index('--speculative-num-steps')+1],'1')
   self.assertIn('INFRA_WAIT_BUDGET_CONFIG',env)
   self.assertEqual(int(command[command.index('--prefill-decode-interval')+1]) if '--prefill-decode-interval' in command else 0,interval)
 def test_sync8_and_budget100_use_identical_model_argv(self):
  from timeline_study import command_for
  a,_=command_for(ROOT,self.runpath(),'sync8');b,_=command_for(ROOT,self.runpath(),'budget100')
  self.assertEqual(a,b)
 def test_restore_is_original_and_removes_both_experiment_overrides(self):
  from timeline_study import command_for,clean_environment
  import launch
  cmd,env=command_for(ROOT,self.runpath(),'stable3',restore=True)
  self.assertEqual(cmd,launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)[0])
  self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',env)
  dirty={'INFRA_WAIT_BUDGET_CONFIG':'fixture','PYTHONPATH':str(self.runpath()/'runtime')+':'+str(ROOT/'results/wait-budget-study-2026-10-31/run-old/runtime')+':/keep'}
  cleaned=clean_environment(ROOT,dirty)
  self.assertEqual(cleaned['PYTHONPATH'],'/keep');self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',cleaned)
 def test_unsupported_label_and_directory_fail_before_actions(self):
  from timeline_study import command_for,study_directory
  with self.assertRaises(ValueError):command_for(ROOT,self.runpath(),'budget200')
  with self.assertRaises(ValueError):study_directory(ROOT,ROOT/'results/wait-budget-study-2026-10-31/run-fixture')
 def test_new_run_has_output_and_profile_directories(self):
  from timeline_study import TimelineStudy
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run=root/'results/timeline-study-2026-10-31/run-fixture'
   s=TimelineStudy(root,run)
   for name in ['cases','client-logs','server-logs','profiles','patch-receipts']:
    self.assertTrue((s.run/name).is_dir())
 def test_profile_worker_has_independent_origin_and_uniform_45_second_timeout(self):
  source=(ROOT/'scripts/timeline_worker.py').read_text() if (ROOT/'scripts/timeline_worker.py').exists() else ''
  self.assertIn('client-origin.json',source)
  self.assertIn('timeout=45',source)
  self.assertNotIn('start_profile',source)
