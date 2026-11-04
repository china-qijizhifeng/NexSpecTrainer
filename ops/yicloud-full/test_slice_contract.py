"""CPU regression using native writer._prepare and real BF16 codec, without a target server."""
import importlib.util,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from specforge.inference.capture import CaptureConfig
from specforge.inference.server_capture_writer import ServerCaptureFileWriter

previous=Path.cwd()
with tempfile.TemporaryDirectory(prefix='denovo-slice-regression-') as directory:
 root=Path(directory)/'capture-w0';root.mkdir();os.environ['CAPTURE_WORK_DIR']=str(root)
 os.chdir(root)
 try:
  spec=importlib.util.spec_from_file_location('capture_and_slice',Path(__file__).with_name('capture_and_slice.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  m.shutil=SimpleNamespace(disk_usage=lambda _:SimpleNamespace(free=6_000_000_000_000))
  n=4200;ids=torch.arange(n,dtype=torch.int64);mask=torch.ones(n,dtype=torch.int64)
  full={'input_ids':ids,'loss_mask':mask,
        'hidden_states':torch.ones((1,n,15360),dtype=torch.bfloat16),
        'target_last_hidden_states':torch.full((1,n,5120),2,dtype=torch.bfloat16)}
  capture=CaptureConfig.from_strategy(full.keys(),(37,38,39),target_repr='hidden_state',target_hidden_size=5120)
  layout=SimpleNamespace(aux_feature='hidden_states',last_hidden_feature='target_last_hidden_states',
       passthrough=(('input_ids','input_ids'),('loss_mask','loss_mask')))
  def writer(output,resume=False):
   return ServerCaptureFileWriter(output,capture=capture,layout=layout,algorithm='dspark',target_model_version='pinned',zstd=True,resume=resume)
  def task(name,split):
   return SimpleNamespace(task_id=name,target_model_version='pinned',payload={'input_ids':ids.tolist(),'loss_mask':mask.tolist()},
       metadata={'split':split,'slice_starts':[0,3967],'trajectory_id':name,'instance_id':split+'-instance'})
  calls=[]
  def persist(self,task,ref,prepared,store):
   # Exercise native field/shape validation as well as the real path preparation.
   tensors={k:v.unsqueeze(0) if k in ('input_ids','loss_mask') else v for k,v in full.items()}
   self._validate(tensors,task)
   calls.append(str(prepared[1]))
   m.save_zstd_features(full,prepared[1],workers=2)
   return {'status':'captured'}
  m.original_persist=persist
  cases=[('relative-train','train',Path('features'),False),
         ('absolute-validation','validation',root/'features',True),
         ('relative-validation','validation',Path('features'),True)]
  for name,group,output,resume in cases:
   t=task(name,group)
   with writer(output,resume) as w:
    prepared=w._prepare(t)
    assert prepared[1].is_absolute()==output.is_absolute()
    m.sliced_persist(w,t,None,prepared,None)
    assert not prepared[1].exists()
   receipt=json.loads((root/f'sliced/{name}.json').read_text())
   assert receipt['complete'] and len(receipt['windows'])==2
   for window,start in zip(receipt['windows'],[0,3967]):
    record=m.load_feature_file(window['file'])
    assert torch.equal(record['input_ids'],ids[start:start+4096])
    assert torch.equal(record['hidden_states'],full['hidden_states'][:,start:start+4096])
    if start:assert not record['loss_mask'][:128].any() and record['loss_mask'][128:].all()
  # A foreign path fails before the persistence callback, even with a valid name.
  before=len(calls)
  with writer(root.parent/'outside') as w:
   t=task('foreign','train');prepared=w._prepare(t)
   try:m.sliced_persist(w,t,None,prepared,None)
   except AssertionError:pass
   else:raise AssertionError('foreign output accepted')
  assert len(calls)==before
  # A dangling symlink is also rejected before any write through it.
  with writer('features',True) as w:
   t=task('redirect','train');prepared=w._prepare(t)
   target=root.parent/'should-not-exist.ckpt.zst';prepared[1].symlink_to(target)
   try:m.sliced_persist(w,t,None,prepared,None)
   except AssertionError:pass
   else:raise AssertionError('symlink redirect accepted')
   assert not target.exists() and len(calls)==before
  # Invalid captured IDs retain the complete record for inspection.
  def invalid(self,task,ref,prepared,store):
   m.save_zstd_features({'input_ids':torch.zeros_like(ids),'loss_mask':mask},prepared[1])
   return {}
  m.original_persist=invalid
  with writer('features',True) as w:
   t=task('bad','train');prepared=w._prepare(t)
   try:m.sliced_persist(w,t,None,prepared,None)
   except AssertionError:pass
   else:raise AssertionError('invalid IDs accepted')
   assert prepared[1].exists()
  # Resolving a symlinked output directory must not turn an outside path into an owned path.
  saved=root/'features-retained'
  (root/'features').rename(saved)
  (root/'features').symlink_to(root.parent/'outside',target_is_directory=True)
  with writer('features',True) as w:
   t=task('parent-redirect','train');prepared=w._prepare(t)
   try:m.sliced_persist(w,t,None,prepared,None)
   except AssertionError:pass
   else:raise AssertionError('directory redirect accepted')
   assert not prepared[1].exists()
  result={'passed':True,'native_writer_prepare':True,'actual_bf16_codec':True,'success_cases':len(cases),'verified_windows':6,
          'relative_absolute_train_validation':True,'overlap_mask_checked':True,'invalid_capture_retained':True,
          'foreign_and_symlink_rejected_before_write':True}
  print(json.dumps(result),flush=True)
 finally:os.chdir(previous)
