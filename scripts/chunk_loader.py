"""Transparent per-run loader for one pinned scheduler module; no installation edits."""
import importlib.abc,importlib.machinery,json,sys
from pathlib import Path
from wait_budget_policy import checked_source,install_scheduler_budget
TARGET='sglang.srt.managers.scheduler'

class BudgetSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self,name,path,settings):
        super().__init__(name,path);self.settings=settings
    def get_code(self,fullname):
        content=checked_source(self.path,self.settings['source_sha256'])
        return compile(content,self.path,'exec',dont_inherit=True)
    def exec_module(self,module):
        super().exec_module(module)
        install_scheduler_budget(module.Scheduler,module.torch,self.settings)

class BudgetFinder(importlib.abc.MetaPathFinder):
    def __init__(self,settings):self.settings=settings
    def find_spec(self,fullname,path,target=None):
        if fullname!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or Path(spec.origin).resolve()!=Path(self.settings['source_path']).resolve():
            raise ValueError('Scheduler resolved outside the pinned installation')
        spec.loader=BudgetSourceLoader(fullname,spec.origin,self.settings)
        return spec

def activate_from_file(config_path):
    path=Path(config_path).resolve();settings=json.loads(path.read_text())
    root=Path(settings['project_root']).resolve();run=Path(settings['run_dir']).resolve()
    if (run.parent!=root/'results/chunk-study-2026-10-31' or not run.name.startswith('run-')
        or path.parent!=run/'runtime-configs' or settings['budget_ms'] not in [None,100]
        or Path(settings['receipt_dir']).resolve().parent!=run/'patch-receipts'):
        raise ValueError('Wait-budget configuration is outside the frozen project run')
    expected=root/'.venv/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py'
    if Path(settings['source_path']).resolve()!=expected.resolve():
        raise ValueError('Wrong pinned scheduler path')
    checked_source(expected,settings['source_sha256'])
    checked_source(root/'scripts/wait_budget_policy.py',settings['policy_sha256'])
    if TARGET in sys.modules:raise RuntimeError('Scheduler imported before wait-budget activation')
    sys.meta_path.insert(0,BudgetFinder(settings))

def startup_hook_source():
    return '''import os
if os.environ.get("INFRA_WAIT_BUDGET_CONFIG"):
    try:
        from chunk_loader import activate_from_file
        activate_from_file(os.environ["INFRA_WAIT_BUDGET_CONFIG"])
    except Exception as error:
        raise SystemExit("Wait-budget activation failed: " + repr(error))
'''
