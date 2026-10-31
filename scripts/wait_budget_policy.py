"""Opt-in, coordinator-owned wait-budget policy for one pinned TP4 experiment."""
import hashlib,json,math,os,time
from pathlib import Path


def checked_source(path,expected_sha256):
    content=Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise ValueError('Pinned source digest does not match')
    return content


def limit_wait(defer,remaining,waiting_entries,now,budget_ms):
    if budget_ms is not None and (type(budget_ms) is not int or budget_ms!=100):
        raise ValueError('This experiment supports only the frozen100ms budget or no cap')
    if not defer or budget_ms is None or not waiting_entries:
        return bool(defer),remaining,0
    valid=lambda x:type(x) in (int,float) and math.isfinite(x) and x>0
    if not valid(now) or any(not valid(x) or x>now for x in waiting_entries):
        return False,0,2
    if now+1e-9>=min(waiting_entries)+budget_ms/1000:
        return False,0,1
    return True,remaining,0


def master_decision(scheduler,batch,original,budget_ms,now=time.perf_counter):
    try:
        defer=original(scheduler,batch)
        remaining=scheduler._prefill_decode_interval_remaining
        entries=[]
        if budget_ms is not None and defer:
            entries=[getattr(getattr(req,'time_stats',None),'wait_queue_entry_time',None)
                     for req in scheduler.waiting_queue]
        decision,count,reason=limit_wait(defer,remaining,entries,
            now() if entries else None,budget_ms)
        return [int(decision),count,reason]
    except Exception:
        return [-1,0,3]


def apply_consensus(scheduler,payload):
    if (len(payload)!=3 or payload[0] not in (-1,0,1)
            or type(payload[1]) is not int or not 0<=payload[1]<=8):
        raise RuntimeError('Invalid wait-budget consensus payload')
    if payload[0]==-1:
        raise RuntimeError('Wait-budget coordinator failed; all TP ranks stop together')
    scheduler._prefill_decode_interval_remaining=payload[1]
    return bool(payload[0])


def install_scheduler_budget(scheduler_class,torch,settings):
    """Wrap only the existing prefill deferral hook; never mutate installed source."""
    if getattr(scheduler_class,'_infra_wait_budget_installed',False):
        raise RuntimeError('Wait-budget wrapper was already installed')
    original=scheduler_class._should_defer_prefill
    budget=settings['budget_ms']
    if budget not in (None,100):raise ValueError('Unfrozen budget')
    def wrapped(self,running_batch):
        group=self.tp_group
        if self.require_mlp_sync or torch.distributed.get_world_size(group=self.tp_cpu_group)!=4:
            raise RuntimeError('Wait-budget prototype only supports current non-DP TP4')
        if not hasattr(self,'_infra_wait_budget_packet'):
            self._infra_wait_budget_packet=torch.zeros(3,dtype=torch.int64,device='cpu')
            self._infra_wait_budget_calls=0
            self._infra_wait_budget_releases=0
            folder=Path(settings['receipt_dir']);folder.mkdir(parents=True,exist_ok=True)
            receipt={'pid':os.getpid(),'tp_rank':group.rank_in_group,'world_size':4,
                'budget_ms':budget,'installed_scheduler_sha256':settings['source_sha256'],
                'policy_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mechanism':'coordinator decision and remaining counter broadcast on CPU TP group',
                'record_stage':'first_scheduler_call_before_formal_cases'}
            with (folder/f'tp{group.rank_in_group}-pid{os.getpid()}.json').open('x') as f:
                json.dump(receipt,f,indent=2)
        packet=self._infra_wait_budget_packet
        if group.rank_in_group==0:
            payload=master_decision(self,running_batch,original,budget)
            for i,value in enumerate(payload):packet[i]=value
        torch.distributed.broadcast(packet,src=group.first_rank,group=self.tp_cpu_group)
        payload=packet.tolist()
        self._infra_wait_budget_calls+=1
        self._infra_wait_budget_releases+=payload[2]==1
        return apply_consensus(self,payload)
    scheduler_class._should_defer_prefill=wrapped
    scheduler_class._infra_wait_budget_installed=True
