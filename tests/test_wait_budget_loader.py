import sys,unittest,tempfile,hashlib,types,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class WaitBudgetLoaderTests(unittest.TestCase):
 def test_unrelated_modules_are_not_intercepted(self):
  from wait_budget_loader import BudgetFinder
  finder=BudgetFinder({'source_path':'/does/not/exist'})
  self.assertIsNone(finder.find_spec('json',None))
 def test_pinned_module_is_wrapped_without_rewriting_source_or_path(self):
  from wait_budget_loader import BudgetSourceLoader
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'scheduler.py'
   p.write_text('torch = object()\nclass Scheduler:\n    def _should_defer_prefill(self, batch):\n        return True\n')
   before=p.read_bytes();name='sglang.srt.managers.scheduler'
   settings={'source_path':str(p),'source_sha256':hashlib.sha256(before).hexdigest(),'budget_ms':100,'receipt_dir':temp}
   loader=BudgetSourceLoader(name,str(p),settings)
   spec=importlib.util.spec_from_file_location(name,p,loader=loader)
   module=importlib.util.module_from_spec(spec);loader.exec_module(module)
   self.assertTrue(module.Scheduler._infra_wait_budget_installed)
   self.assertEqual(module.__file__,str(p));self.assertEqual(p.read_bytes(),before)
 def test_wrong_source_digest_stops_before_executing_module(self):
  from wait_budget_loader import BudgetSourceLoader
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'scheduler.py';marker=Path(temp)/'executed'
   p.write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
   loader=BudgetSourceLoader('sglang.srt.managers.scheduler',str(p),{'source_sha256':'0'*64})
   spec=importlib.util.spec_from_file_location(loader.name,p,loader=loader)
   with self.assertRaises(ValueError):loader.exec_module(importlib.util.module_from_spec(spec))
   self.assertFalse(marker.exists())
 def test_unconfigured_startup_hook_is_inert(self):
  from wait_budget_loader import startup_hook_source
  namespace={'__name__':'fixture'}
  import os
  original=os.environ.pop('INFRA_WAIT_BUDGET_CONFIG',None)
  before=list(sys.meta_path)
  try:exec(startup_hook_source(),namespace)
  finally:
   if original is not None:os.environ['INFRA_WAIT_BUDGET_CONFIG']=original
  self.assertEqual(sys.meta_path,before)
