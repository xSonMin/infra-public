import os,sys,unittest,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class WaitBudgetStudyTests(unittest.TestCase):
 def runpath(self):return ROOT/'results/wait-budget-study-2026-10-31/run-fixture'
 def test_budget_and_sync_control_have_identical_model_commands(self):
  from wait_budget_study import command_for
  sync,env=command_for(ROOT,self.runpath(),'sync8')
  cap,cenv=command_for(ROOT,self.runpath(),'budget100')
  self.assertEqual(sync,cap)
  self.assertEqual(sync[sync.index('--speculative-num-steps')+1],'1')
  self.assertEqual(sync[sync.index('--prefill-decode-interval')+1],'8')
  self.assertTrue(env['INFRA_WAIT_BUDGET_CONFIG'].endswith('/sync8.json'))
  self.assertTrue(cenv['INFRA_WAIT_BUDGET_CONFIG'].endswith('/budget100.json'))
 def test_plain_control_avoids_runtime_override(self):
  from wait_budget_study import command_for
  cmd,env=command_for(ROOT,self.runpath(),'interval8')
  self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',env)
  self.assertNotIn(str(self.runpath()/'runtime'),env.get('PYTHONPATH',''))
 def test_original_restoration_has_no_runtime_override(self):
  from wait_budget_study import command_for
  import launch
  cmd,env=command_for(ROOT,self.runpath(),'stable3',restore=True)
  original,_=launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)
  self.assertEqual(cmd,original)
  self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',env)
  for label in ['interval8','sync8','budget100']:
   with self.assertRaises(ValueError):command_for(ROOT,self.runpath(),label,restore=True)
 def test_environment_cleanup_drops_only_experiment_override(self):
  from wait_budget_study import clean_environment
  env={'INFRA_WAIT_BUDGET_CONFIG':'fixture','PATH':'/usr/bin','PYTHONPATH':str(self.runpath()/'runtime')+':/keep-me'}
  cleaned=clean_environment(ROOT,env)
  self.assertEqual(cleaned['PYTHONPATH'],'/keep-me')
  self.assertNotIn('INFRA_WAIT_BUDGET_CONFIG',cleaned)
  self.assertEqual(cleaned['PATH'],'/usr/bin')
 def test_invalid_label_or_run_is_rejected(self):
  from wait_budget_study import command_for,study_directory
  with self.assertRaises(ValueError):command_for(ROOT,self.runpath(),'budget50')
  with self.assertRaises(ValueError):study_directory(ROOT,ROOT/'results/factorial-study-2026-10-31/run-fixture')
 def test_new_run_creates_client_output_directories(self):
  from wait_budget_study import BudgetStudy
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);run=root/'results/wait-budget-study-2026-10-31/run-fixture'
   s=BudgetStudy(root,run)
   for name in ['cases','case-configs','client-logs','server-logs','patch-receipts']:
    self.assertTrue((s.run/name).is_dir())
