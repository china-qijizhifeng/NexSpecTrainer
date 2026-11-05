"""Direct slicing preserves the native tensors, overlap masks and ready receipts."""
import importlib.util,json,os,tempfile,time,types
from pathlib import Path
import torch
from specforge.data.feature_file import load_feature_file

with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);os.environ['CAPTURE_WORK_DIR']=str(root)
 spec=importlib.util.spec_from_file_location('direct_slices',Path(__file__).with_name('capture_and_slice.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 n=4200;torch.manual_seed(1);torch.set_num_threads(4)
 full={'input_ids':torch.arange(n,dtype=torch.int64),'loss_mask':torch.ones(n,dtype=torch.int64),'hidden_states':torch.randn(1,n,15360,dtype=torch.bfloat16),'target_last_hidden_states':torch.randn(1,n,5120,dtype=torch.bfloat16)}
 task=types.SimpleNamespace(task_id='slice-test',payload={'input_ids':full['input_ids'].tolist(),'loss_mask':full['loss_mask'].tolist()},metadata={'split':'train','slice_starts':[0,3967],'trajectory_id':'trajectory','instance_id':'instance'})
 t=time.perf_counter();module.persist_slices(task,full,t,t,t)
 receipt=json.loads((root/'sliced/slice-test.json').read_text());assert receipt['complete'] and len(receipt['windows'])==2
 for w in receipt['windows']:
  record=load_feature_file(w['file']);a,b=w['training_slice']
  assert torch.equal(record['input_ids'],full['input_ids'][a:b])
  expected=full['loss_mask'][a:b].clone()
  if a:expected[:128]=0
  assert torch.equal(record['loss_mask'],expected)
  for k in ['hidden_states','target_last_hidden_states']:assert torch.equal(record[k],full[k][:,a:b])
 assert not list(root.rglob('*.partial'))
 assert not (root/'features/slice-test.ckpt.zst').exists()
 print(json.dumps({'passed':True,'tokens':n,'windows':2,'all_tensors_bitwise_equal':True}))
