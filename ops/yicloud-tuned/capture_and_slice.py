"""Fetch once, validate, then persist verified slices without a full-record disk roundtrip."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from specforge.data.feature_file import load_zstd_features, save_zstd_features
from specforge.algorithms.common.hidden_states_data import normalize_dspark_offline_sample
from specforge.inference.server_capture_writer import ServerCaptureFileWriter
from specforge.runtime.data_plane.feature_store import drain_feature_store_sample_removals

ROOT=Path(os.environ['CAPTURE_WORK_DIR']).resolve()
STAGE=ROOT.parent


def atomic(path,value):
    temporary=path.with_suffix('.partial')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def sliced_persist(self, task, ref, prepared, store):
    # Guard each trajectory before allocating/writing it. Initial plan leaves >1 TB for checkpoints/headroom.
    assert shutil.disk_usage(STAGE).free > 850_000_000_000, 'shared disk below reserved checkpoint/headroom budget'
    group=task.metadata['split']
    assert group in ('train','validation')
    candidate=Path(prepared[1])
    expected=ROOT/'features'/(task.task_id+'.ckpt.zst')
    # The native CLI passes --output full-train as a relative Path. Normalize
    # both operands before checking ownership; reject redirects before writing.
    assert not candidate.is_symlink() and candidate.resolve()==expected
    started=time.perf_counter()
    tensors,handle=store.get(ref,device='cpu')
    fetched=time.perf_counter()
    try:
        full=self._validate(tensors,task)
        validated=time.perf_counter()
        return persist_slices(task,full,started,fetched,validated)
    finally:
        store.release(handle)
        drain_feature_store_sample_removals(store,[ref.sample_id])


def persist_slices(task,full,started,fetched,validated):
    group=task.metadata['split']
    ids,mask=full['input_ids'].reshape(-1),full['loss_mask'].reshape(-1)
    n=len(task.payload['input_ids'])
    result=dict(sample=task.task_id,status='captured',tokens=n,tensor_bytes=sum(v.numel()*v.element_size() for v in full.values()),file_bytes=0,direct_slices=True)
    output=ROOT/(group+'-features');output.mkdir(exist_ok=True)
    receipt=[]
    for start in task.metadata['slice_starts']:
        end=min(start+4096,n)
        block_mask=mask[start:end].clone()
        if start:block_mask[:128]=0
        assert torch.any((block_mask[:-1]>0)&(block_mask[1:]>0))
        record={'input_ids':ids[start:end].clone(),'loss_mask':block_mask,
            'hidden_states':full['hidden_states'][:,start:end].contiguous(),
            'target_last_hidden_states':full['target_last_hidden_states'][:,start:end].contiguous()}
        normalize_dspark_offline_sample(record,4096)
        name=f'{task.task_id}-s{start:06d}.ckpt.zst'
        final=output/name;tmp=output/(name+'.partial')
        assert not final.exists()
        save_zstd_features(record,tmp,workers=8,checksums=False)
        with tmp.open('rb') as stream:os.fsync(stream.fileno())
        restored=load_zstd_features(tmp,workers=8)
        assert set(restored)==set(record) and all(torch.equal(restored[k],v) for k,v in record.items())
        digest=hashlib.sha256()
        with tmp.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
        tmp.rename(final)
        receipt.append({'file':str(final),'id':name,'split':group,'trajectory_id':task.metadata['trajectory_id'],
            'instance_id':task.metadata['instance_id'],'target_context':[0,n],'training_slice':[start,end],
            'draft_position_convention':'SpecForge local positions','prefix_loss_masked':128 if start else 0,
            'input_token_sha256':hashlib.sha256(record['input_ids'].contiguous().view(torch.uint8).numpy()).hexdigest(),
            'bytes':final.stat().st_size,'sha256':digest.hexdigest(),'codec_bit_exact':True})
    directory_fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    receipt_dir=ROOT/'sliced';receipt_dir.mkdir(exist_ok=True)
    atomic(receipt_dir/(task.task_id+'.json'),{'input':task.metadata,'capture':result,'windows':receipt,'complete':True})
    progress={'completed_trajectories':len(list(receipt_dir.glob('*.json'))),'last_id':task.task_id,
              'last_tokens':n,'last_windows':len(receipt),'free_bytes':shutil.disk_usage(STAGE).free}
    atomic(ROOT/'progress.json',progress)
    elapsed=time.perf_counter()-started
    with (ROOT/'timings.jsonl').open('a') as f:f.write(json.dumps(dict(id=task.task_id,tokens=n,get_decode_s=fetched-started,validate_s=validated-fetched,slice_verify_write_s=time.perf_counter()-validated,consume_s=elapsed,time=time.time()))+'\n')
    print(json.dumps({'sliced':progress}),flush=True)
    return result


if __name__=='__main__':
    ServerCaptureFileWriter._persist=sliced_persist
    from scripts.prepare_server_hidden_states import main
    raise SystemExit(main(sys.argv[1:]))
