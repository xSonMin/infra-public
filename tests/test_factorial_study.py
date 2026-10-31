import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class FactorialStudyTests(unittest.TestCase):
 def test_two_factors_cover_exactly_four_cells(self):
  from factorial_metrics import CONFIGS
  self.assertEqual(set(CONFIGS.values()),{(1,0),(3,0),(1,8),(3,8)})
  self.assertEqual(len(CONFIGS),4)
 def test_only_frozen_120_second_loads_and_smoke_are_accepted(self):
  from factorial_metrics import validate_workload
  for args in [(512,128,7,120,'periodic'),(2048,128,4,120,'periodic'),(512,128,2,4,'periodic')]:validate_workload(*args)
  for args in [(512,128,7,300,'periodic'),(2048,128,7,120,'periodic'),(512,512,7,120,'periodic'),(512,128,7,120,'poisson')]:
   with self.assertRaises(ValueError):validate_workload(*args)
 def test_fixed_schedules_are_finite_and_old_protocol_stays_frozen(self):
  from factorial_metrics import make_schedule
  self.assertEqual(len(make_schedule('periodic',7,120,1)),840)
  self.assertEqual(len(make_schedule('periodic',4,120,1)),480)
  self.assertEqual(make_schedule('periodic',2,4,1),[i/2 for i in range(8)])
  from overnight_metrics import validate_workload
  with self.assertRaises(ValueError):validate_workload(512,128,7,120,'periodic')
 def test_factorial_commands_vary_only_the_two_factors(self):
  from factorial_study import command_for
  one,_=command_for(ROOT,'s1i0');three,_=command_for(ROOT,'s3i0')
  eight1,_=command_for(ROOT,'s1i8');eight3,_=command_for(ROOT,'s3i8')
  self.assertEqual(eight1,one+['--prefill-decode-interval','8'])
  self.assertEqual(eight3,three+['--prefill-decode-interval','8'])
  self.assertNotIn('--enable-metrics',eight3)
  self.assertNotIn('--enable-request-time-stats-logging',eight3)
  self.assertEqual(eight3[eight3.index('--host')+1],'127.0.0.1')
  self.assertEqual(eight3[eight3.index('--speculative-num-steps')+1],'3')
 def test_restore_is_byte_for_byte_original_launch_argv(self):
  from factorial_study import command_for
  import launch
  restored,_=command_for(ROOT,'s3i0',restore=True)
  expected,_=launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)
  self.assertEqual(restored,expected)
  for label in ['s1i0','s1i8','s3i8']:
   with self.assertRaises(ValueError):command_for(ROOT,label,restore=True)
 def test_client_is_local_bounded_and_has_no_implicit_client_queue(self):
  from factorial_study import client_command
  run=ROOT/'results/factorial-study-2026-10-31/run-test'
  load={'input_len':2048,'output_len':128,'rate':4,'window_s':120,'pattern':'periodic','offsets':[i/4 for i in range(480)]}
  cmd=client_command(ROOT,run,'F-s3i8-r1-context',load)
  self.assertEqual(cmd[cmd.index('--num-prompts')+1],'480')
  self.assertEqual(cmd[cmd.index('--host')+1],'127.0.0.1')
  self.assertNotIn('--max-concurrency',cmd)
 def test_cutoff_keeps_time_for_drain_and_original_service_restore(self):
  from factorial_metrics import can_start
  self.assertTrue(can_start(1500,0,120))
  self.assertFalse(can_start(1000,0,120))
 def test_bad_labels_and_paths_are_rejected_before_remote_actions(self):
  from factorial_study import command_for,study_directory
  with self.assertRaises(ValueError):command_for(ROOT,'s3i80')
  with self.assertRaises(ValueError):study_directory(ROOT,ROOT/'results/overnight-study-2026-10-30/run-test')
  self.assertEqual(study_directory(ROOT,ROOT/'results/factorial-study-2026-10-31/run-test').parent,ROOT/'results/factorial-study-2026-10-31')

 def test_fresh_run_has_all_client_output_directories_after_file_only_sync(self):
  import tempfile
  from factorial_study import FactorialStudy
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/'results/factorial-study-2026-10-31/run-fixture'
   study=FactorialStudy(root,run)
   for name in ['cases','case-configs','client-logs','server-logs']:
    self.assertTrue((study.run/name).is_dir(),name+' must exist before native client starts')
   marker=study.run/'cases/preserved.json';marker.write_text('{}')
   FactorialStudy(root,run)
   self.assertEqual(marker.read_text(),'{}')
