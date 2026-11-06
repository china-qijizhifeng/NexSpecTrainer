"""Capture with native writer; accept only complete hybrid round trips."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from specforge.data.feature_file import load_hybrid_features
from specforge.inference.server_capture_writer import ServerCaptureFileWriter

original=ServerCaptureFileWriter._persist
work=Path(os.environ['CAPTURE_WORK_DIR'])
started=time.time()
completed=0


def atomic(path,value):
    temporary=path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def persist(self,task,ref,prepared,store):
    global completed
    assert shutil.disk_usage(work).free > 1_000_000_000_000, 'less than 1 TB output headroom'
    path=Path(prepared[1]);assert path.suffix=='.hybrid' and not path.is_symlink()
    assert path.parent.resolve()==(work/'features').resolve()
    result=original(self,task,ref,prepared,store)
    with path.open('rb') as stream:assert stream.read(8)==b'NSACTH01'
    # Native reader validates each tensor checksum, covering all persisted bits.
    restored=load_hybrid_features(path,workers=12)
    n=len(task.payload['input_ids'])
    assert torch.equal(restored['input_ids'].reshape(-1),torch.tensor(task.payload['input_ids'],dtype=torch.int64))
    assert torch.equal(restored['loss_mask'].reshape(-1),torch.tensor(task.payload['loss_mask'],dtype=torch.int64))
    assert restored['hidden_states'].shape==(1,n,15360)
    assert restored['target_last_hidden_states'].shape==(1,n,5120)
    for key in ['hidden_states','target_last_hidden_states']:
        tensor=restored[key];assert tensor.dtype==torch.bfloat16
        for block in tensor.reshape(-1).split(1<<24):assert torch.isfinite(block).all()
    if completed == 0 and os.environ['KUBEBRAIN_REPLICA'] == '0':
        from ops.northgate_capture.cp_acceptance import compare_features

        reference_root = Path(os.environ['NORTHGATE_CP_REFERENCE_RUN'])
        baseline = reference_root/'worker-0/features'/path.name
        receipt_path = reference_root/'worker-0/receipts'/(task.task_id+'.json')
        try:
            old_receipt = json.loads(receipt_path.read_text())
            assert old_receipt['native_readback_checksums_passed']
            reference = load_hybrid_features(baseline, workers=12)
            comparison = compare_features(restored, reference)
            comparison.update(reference=str(baseline), candidate=str(path),
                              tokens=n, scope='one complete short real trace; not long-input equivalence')
            del reference
            marker = 'CP_ACCEPTED.json' if comparison['passed'] else 'CP_REJECTED.json'
            atomic(Path(os.environ['RUN_ROOT'])/marker, comparison)
            assert comparison['passed'], 'CP short-input numerical gate failed'
        except Exception as error:
            rejected = Path(os.environ['RUN_ROOT'])/'CP_REJECTED.json'
            if not rejected.exists():atomic(rejected, {'passed':False, 'error':str(error)})
            raise
    del restored
    completed+=1
    receipt={'id':task.task_id,'tokens':n,'file':str(path),'bytes':path.stat().st_size,
             'codec':'conditional_brotli_zstd','brotli_quality':10,'layers':[37,38,39],
             'full_context':True,'native_readback_checksums_passed':True,'input_ids_loss_mask_exact':True,
             'metadata':task.metadata,'capture_result':result,'completed_at':time.time()}
    receipts=work/'receipts';receipts.mkdir(exist_ok=True)
    atomic(receipts/(task.task_id+'.json'),receipt)
    atomic(work/'progress.json',{'completed':completed,'last_id':task.task_id,'last_tokens':n,
                                'elapsed_seconds':time.time()-started,'last_file':str(path)})
    print(json.dumps({'captured':task.task_id,'tokens':n,'hybrid_bytes':path.stat().st_size,
                      'completed':completed}),flush=True)
    return result

if __name__=='__main__':
    ServerCaptureFileWriter._persist=persist
    from scripts.prepare_server_hidden_states import main
    main(sys.argv[1:])
