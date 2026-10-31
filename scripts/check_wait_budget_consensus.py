#!/usr/bin/env python3
"""CPU-only, four-process Gloo validation of the actual wait-budget wrapper."""
import argparse,datetime,json,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

def worker(rank,rendezvous,folder):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['GLOO_SOCKET_IFNAME']='lo'
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from wait_budget_policy import install_scheduler_budget
    import time
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=datetime.timedelta(seconds=25))
    checks=[]
    try:
        for budget in [None,100]:
            class Scheduler:
                require_mlp_sync=False
                def __init__(self):
                    self.tp_group=SimpleNamespace(rank_in_group=rank,first_rank=0)
                    self.tp_cpu_group=dist.group.WORLD
                    self._prefill_decode_interval_remaining=8
                    self.waiting_queue=[]
                    self.raise_fixture=False
                def _should_defer_prefill(self,batch):
                    if self.raise_fixture:raise ValueError('coordinator error fixture')
                    if self._prefill_decode_interval_remaining==0:return False
                    self._prefill_decode_interval_remaining-=1
                    return True
            receipt=Path(folder)/('sync' if budget is None else 'budget100')
            install_scheduler_budget(Scheduler,torch,{'budget_ms':budget,'receipt_dir':str(receipt),'source_sha256':'cpu-fixture'})
            scheduler=Scheduler()
            for case,age,expected in [('overdue',.2,budget is None),('fresh',.001,True),('empty',None,True)]:
                scheduler._prefill_decode_interval_remaining=8
                stamp=time.perf_counter()-age if age is not None else None
                if rank!=0 and stamp is not None:stamp+=10000
                scheduler.waiting_queue=[] if stamp is None else [SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=stamp))]
                decision=scheduler._should_defer_prefill(None)
                assert decision==expected,(rank,budget,case,decision)
                wanted=0 if budget==100 and case=='overdue' else 7
                assert scheduler._prefill_decode_interval_remaining==wanted
                states=[None]*4
                dist.all_gather_object(states,[decision,scheduler._prefill_decode_interval_remaining])
                assert all(s==states[0] for s in states),states
                checks.append({'budget_ms':budget,'case':case,'decision':decision,'remaining':wanted})
            scheduler.raise_fixture=True
            caught=False
            try:scheduler._should_defer_prefill(None)
            except RuntimeError:caught=True
            states=[None]*4;dist.all_gather_object(states,caught)
            assert all(states),'Coordinator exception not transmitted to all ranks'
            checks.append({'budget_ms':budget,'case':'coordinator_error','all_ranks_raised':True})
        (Path(folder)/f'rank-{rank}.json').write_text(json.dumps({'rank':rank,'passed':True,'checks':checks},indent=2)+'\n')
    finally:dist.destroy_process_group()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];folder=args.output.resolve()
    assert folder.is_relative_to(root/'results/wait-budget-study-2026-10-31')
    folder.mkdir(parents=True,exist_ok=False)
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='infra-budget-consensus-',dir=folder) as temp:
        mp.spawn(worker,args=(str(Path(temp)/'gloo-rendezvous'),str(folder)),nprocs=4,join=True)
    rows=[json.loads((folder/f'rank-{r}.json').read_text()) for r in range(4)]
    assert all(r['passed'] for r in rows)
    result={'passed':True,'world_size':4,'backend':'gloo','device':'cpu','rank_reports':rows}
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'world_size':4,'backend':'gloo','device':'cpu','rank_reports':4}))

if __name__=='__main__':main()
