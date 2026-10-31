import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class OvernightTests(unittest.TestCase):
 def setUp(self):
  p=ROOT/'scripts/overnight_metrics.py';self.assertTrue(p.exists(),'Night protocol helpers required')
  spec=importlib.util.spec_from_file_location('overnight_metrics',p);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
 def test_schedule_is_repeatable_bounded_and_preserves_poisson_intervals(self):
  a=self.m.make_schedule('periodic',7,600,1);self.assertEqual(len(a),4200);self.assertAlmostEqual(a[-1],4199/7)
  p=self.m.make_schedule('poisson',7,300,42);self.assertEqual(p,self.m.make_schedule('poisson',7,300,42));self.assertTrue(0<p[0]<p[-1]<300)
  for pattern,rate,window in [('periodic',20,600),('periodic',7,36000),('poisson',7,600)]:
   with self.assertRaises(ValueError):self.m.make_schedule(pattern,rate,window,1)
 def test_latency_accounting_uses_actual_output_length(self):
  good={'success':True,'output_len':512,'ttft':.2,'latency':.2+511*.029,'send_lag_s':.001}
  self.assertTrue(self.m.is_good(good,512));self.assertFalse(self.m.is_good(dict(good,output_len=128),512))
  self.assertFalse(self.m.is_good(dict(good,success=False),512));self.assertFalse(self.m.is_good(dict(good,ttft=float('nan')),512))
 def test_cutoff_reserves_case_and_restore_time(self):
  self.assertTrue(self.m.can_start(1000,0,600));self.assertFalse(self.m.can_start(1000,250,600))
 def test_no_instrumentation_and_single_candidate_change(self):
  import overnight_study
  base,env=overnight_study.command_for(ROOT,'fixed1',False)
  candidate,_=overnight_study.command_for(ROOT,'interval8',False)
  self.assertEqual(candidate,base+['--prefill-decode-interval','8'])
  self.assertNotIn('--enable-metrics',candidate);self.assertNotIn('--enable-request-time-stats-logging',candidate)
  self.assertEqual(candidate[candidate.index('--log-level')+1],'warning')
  import launch
  expected,_=launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)
  restored,_=overnight_study.command_for(ROOT,'fixed3',True);self.assertEqual(restored,expected)
 def test_unknown_workload_is_rejected(self):
  self.m.validate_workload(512,128,7,600,'periodic')
  self.m.validate_workload(2048,128,4,300,'poisson')
  self.m.validate_workload(512,512,1.5,300,'periodic')
  for args in [(8192,128,7,300,'periodic'),(512,128,9,300,'poisson'),(512,512,7,300,'periodic')]:
   with self.assertRaises(ValueError):self.m.validate_workload(*args)
 def test_client_gc_is_restored_after_timing_errors(self):
  import gc
  self.assertTrue(hasattr(self.m,'TimedClientGC'),'Explicit timed-client GC control is required')
  before=gc.isenabled();control=self.m.TimedClientGC()
  try:
   control.begin();self.assertFalse(gc.isenabled());self.assertTrue(control.started)
  finally:control.end()
  self.assertEqual(gc.isenabled(),before)
  was=gc.isenabled()
  try:
   gc.disable();control=self.m.TimedClientGC();control.begin();control.end();self.assertFalse(gc.isenabled())
  finally:
   if was:gc.enable()
