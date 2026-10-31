import sys,unittest,tempfile,os,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class WaitBudgetPolicyTests(unittest.TestCase):
 def load(self):
  import wait_budget_policy
  return wait_budget_policy
 def test_disabled_budget_preserves_original_decision_and_count(self):
  m=self.load()
  self.assertEqual(m.limit_wait(True,7,[0],500,None),(True,7,0))
  self.assertEqual(m.limit_wait(False,0,[0],500,None),(False,0,0))
 def test_releases_on_budget_boundary_and_keeps_wait_below_it(self):
  m=self.load()
  self.assertEqual(m.limit_wait(True,7,[10],10.099,100),(True,7,0))
  self.assertEqual(m.limit_wait(True,7,[10],10.1,100),(False,0,1))
  self.assertEqual(m.limit_wait(True,5,[10],10.5,100),(False,0,1))
 def test_uses_oldest_waiting_request(self):
  m=self.load()
  self.assertEqual(m.limit_wait(True,6,[10.15,10.0],10.2,100),(False,0,1))
 def test_no_waiting_requests_keeps_round_semantics(self):
  m=self.load();self.assertEqual(m.limit_wait(True,4,[],20,100),(True,4,0))
 def test_invalid_queue_clock_releases_instead_of_extending_wait(self):
  m=self.load()
  for stamps,now in [([None],10),([0],10),([float('nan')],10),([11],10),([10],float('nan'))]:
   self.assertEqual(m.limit_wait(True,5,stamps,now,100),(False,0,2))
 def test_original_admission_is_never_overridden_to_defer(self):
  m=self.load();self.assertEqual(m.limit_wait(False,0,[10],11,100),(False,0,0))
 def test_only_frozen_budget_is_accepted(self):
  m=self.load()
  for budget in [0,1,50,200,float('nan'),'100']:
   with self.assertRaises(ValueError):m.limit_wait(True,3,[10],10.01,budget)
 def test_loader_preserves_original_path_and_disables_cleanly(self):
  m=self.load()
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'sample.py';p.write_text('value = 3\n')
   digest=hashlib.sha256(p.read_bytes()).hexdigest()
   self.assertEqual(m.checked_source(p,digest),p.read_bytes())
   with self.assertRaises(ValueError):m.checked_source(p,'0'*64)
 def test_master_only_clock_used_and_follower_receives_same_remaining(self):
  m=self.load()
  class Batch:
   def batch_size(self):return 1
  class Req:pass
  class Stats:wait_queue_entry_time=10.0
  req=Req();req.time_stats=Stats()
  class Scheduler:
   _prefill_decode_interval_remaining=8
   waiting_queue=[req]
  calls=[]
  def original(self,batch):
   calls.append('original');self._prefill_decode_interval_remaining-=1;return True
  master=Scheduler()
  data=m.master_decision(master,Batch(),original,100,now=lambda:10.2)
  self.assertEqual(data,[0,0,1]);self.assertEqual(calls,['original'])
  follower=Scheduler()
  self.assertFalse(m.apply_consensus(follower,data))
  self.assertEqual(follower._prefill_decode_interval_remaining,0)
 def test_master_error_is_transmitted_as_failure_to_every_rank(self):
  m=self.load()
  class Scheduler:pass
  def bad(*args):raise RuntimeError('fixture')
  payload=m.master_decision(Scheduler(),None,bad,100,now=lambda:10)
  self.assertEqual(payload,[-1,0,3])
  with self.assertRaises(RuntimeError):m.apply_consensus(Scheduler(),payload)
 def test_original_source_file_is_never_rewritten(self):
  m=self.load()
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'scheduler.py';p.write_text('class Scheduler: pass\n')
   before=p.read_bytes();m.checked_source(p,hashlib.sha256(before).hexdigest())
   self.assertEqual(p.read_bytes(),before)
