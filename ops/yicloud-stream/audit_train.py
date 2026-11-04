"""Observe real optimizer updates without importing training in spawned NFC helpers."""
def main():
    import hashlib,json,os,sys
    from pathlib import Path
    import torch
    from specforge.training.backend import FSDPTrainingBackend
    from specforge.training.strategies.base import DSparkTrainStrategy
    from runtime import install
    install(os.environ['STREAM_ROOT'])
    root=Path(os.environ['RUN_DIR']);rank=int(os.environ['RANK']);phase='full'
    assert int(os.environ['WORLD_SIZE'])==8
    observed={};losses=[];norms=[];consumed=[];eval_count=0;consumed_slots=[];eval_slots=[]
    def checksum(p):
        h=hashlib.sha256();p=p.detach().reshape(-1)
        for start in range(0,p.numel(),4<<20):h.update(p[start:start+(4<<20)].cpu().view(torch.uint8).numpy())
        return h.hexdigest()
    orig_prepare=FSDPTrainingBackend.prepare_model
    orig_step=FSDPTrainingBackend.step
    orig_loss=DSparkTrainStrategy.forward_loss
    def prepare(self,model,**kw):
        result=orig_prepare(self,model,**kw)
        frozen=[p for m in self.ignored_frozen_modules for p in m.parameters()]
        assert len(frozen)==2 and all(not p.requires_grad and p.grad is None for p in frozen)
        assert {id(p) for p in frozen}.isdisjoint(id(p) for p in self.optimizer.model_params)
        observed.update(frozen=frozen,hashes=[checksum(p) for p in frozen])
        return result
    def loss(self,batch,ctx=None):
        nonlocal eval_count
        training=self.trainable_module().training
        if training:
            consumed_slots.extend(batch.sample_ids)
            ids=batch.tensors['input_ids'].detach().cpu().contiguous()
            consumed.append(hashlib.sha256(ids.view(torch.uint8).numpy()).hexdigest())
        else:
            eval_count+=1
            eval_slots.extend(batch.sample_ids)
        result=orig_loss(self,batch,ctx);assert torch.isfinite(result.loss).all()
        losses.append(float(result.loss.detach()))
        return result
    def step(self,**kw):
        value=orig_step(self,**kw);assert value is not None and torch.isfinite(value).all() and float(value)>0
        assert all(p.grad is None for p in observed['frozen']);norms.append(float(value))
        (root/f'{phase}-progress-rank{rank}.json').write_text(json.dumps({'rank':rank,'phase':phase,'updates':len(norms),'losses':losses,'grad_norms':norms}))
        return value
    FSDPTrainingBackend.prepare_model=prepare;FSDPTrainingBackend.step=step;DSparkTrainStrategy.forward_loss=loss
    from specforge.cli import main as cli
    assert cli(sys.argv[1:])==0
    assert len(norms)==796 and len(consumed)==12736 and eval_count==737
    assert observed['hashes']==[checksum(p) for p in observed['frozen']]
    result={'passed':True,'rank':rank,'phase':phase,'updates':len(norms),'losses':losses,'grad_norms':norms,'consumed_input_sha256':consumed,'consumed_slots':consumed_slots,'eval_slots':eval_slots,'eval_micro':eval_count,'frozen_unchanged':True,'frozen_excluded_from_optimizer':True,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    (root/f'{phase}-result-rank{rank}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
