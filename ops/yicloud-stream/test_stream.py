"""Exercise crash boundaries, late arrivals, full coverage, native reads and seek."""
import collections
import json
import tempfile
import threading
import unittest
from pathlib import Path
from schedule import Publisher, atomic, sha256, QUANTUM
from runtime import read_slot


class ScheduleTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);self.capture=self.root/'capture';self.out=self.root/'stream'
  self.capture.mkdir()
  self.samples=[]
  for i in range(384):
   self.samples.append(dict(id=f's{i:04d}',worker=i%4,split='train' if i<368 else 'validation',trajectory_id=f't{i}',instance_id=f'i{i}',num_tokens=3,slice_starts=[0],token_sha256='a'*64,mask_sha256='b'*64,file_bytes=1,file_sha256='c'*64))
  self.plan=dict(samples=self.samples,train_windows=368,padding_duplicates=16,padded_train_windows=384,validation_windows=16,total_steps=3,counts=dict(train=368,validation=16))
  atomic(self.capture/'run.json',{'yicloud':{}})

 def receipt(self,index):
  s=self.samples[index];name=f"{s['id']}-s000000.ckpt.zst";p=self.capture/f"capture-w{s['worker']}"/(s['split']+'-features')/name
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(index).encode())
  w=dict(id=name,file=str(p),split=s['split'],trajectory_id=s['trajectory_id'],instance_id=s['instance_id'],target_context=[0,3],training_slice=[0,3],codec_bit_exact=True,prefix_loss_masked=0,bytes=p.stat().st_size,sha256=sha256(p),input_token_sha256=f'{index:064x}')
  atomic(self.capture/f"capture-w{s['worker']}"/'sliced'/(s['id']+'.json'),dict(complete=True,input={k:v for k,v in s.items() if k not in {'id','worker','file_bytes','file_sha256'}},windows=[w]))

 def test_late_arrivals_full_epoch_and_replay(self):
  publisher=Publisher(self.capture,self.out,self.plan,reserve=128)
  for i in range(255):self.receipt(i)
  publisher.tick();self.assertFalse((self.out/'ready.json').exists())
  self.receipt(255);publisher.tick();self.assertEqual(len(publisher.blocks),1)
  first=(self.out/'train/000000.json').read_bytes()
  for i in range(256,384):self.receipt(i)
  publisher.tick();self.assertEqual(len(publisher.blocks),1)  # keep the 128-window reserve until terminal
  self.assertEqual(first,(self.out/'train/000000.json').read_bytes())
  for rank in range(4):atomic(self.capture/f'capture-w{rank}/RESULT.json',dict(passed=True,complete_partition=True,worker=rank))
  atomic(self.capture/'run.json',{'yicloud':{'terminal_phase':'Succeeded'}})
  publisher.tick();self.assertTrue(publisher.finished)
  actual=[read_slot(self.out,'train',i) for i in range(384)]
  self.assertEqual(len({w['id'] for w in actual}),384)
  counts=collections.Counter(w['input_token_sha256'] for w in actual)
  self.assertEqual(counts,collections.Counter([f'{i:064x}' for i in range(368)]+[f'{i:064x}' for i in range(16)]))
  ranks=[actual[r::8] for r in range(8)]
  self.assertEqual([len(r) for r in ranks],[48]*8)
  with self.assertRaises(AssertionError):Publisher(self.capture,self.out,self.plan,reserve=128)

 def test_uncommitted_receipt_and_capture_failure(self):
  self.receipt(0);receipt=self.capture/'capture-w0/sliced/s0000.json';receipt.rename(receipt.with_suffix('.partial'))
  publisher=Publisher(self.capture,self.out,self.plan,reserve=128);publisher.tick();self.assertEqual(len(publisher.seen),0)
  atomic(self.capture/'run.json',{'yicloud':{'terminal_phase':'Failed'}})
  with self.assertRaisesRegex(AssertionError,'capture terminated'):publisher.tick()

 def test_late_slot_and_fail_closed(self):
  timer=threading.Timer(.05,lambda:atomic(self.out/'train/000000.json',[{'id':'ready'}]*128));timer.start()
  self.assertEqual(read_slot(self.out,'train',0,timeout=3),{'id':'ready'});timer.join()
  atomic(self.out/'failed.json',{'error':'producer failed'})
  with self.assertRaisesRegex(RuntimeError,'producer failed'):read_slot(self.out,'train',0)

 def test_wrong_metadata_and_file_size_rejected(self):
  self.receipt(0);p=self.capture/'capture-w0/sliced/s0000.json';r=json.loads(p.read_text());r['input']['token_sha256']='d'*64;atomic(p,r)
  with self.assertRaises(AssertionError):Publisher(self.capture,self.out,self.plan,reserve=128).tick()


class NativeTests(unittest.TestCase):
 def test_native_refs_shards_seek_and_corruption(self):
  import torch
  import specforge.launch as launch
  from specforge.data.feature_file import save_zstd_features
  from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
  from specforge.runtime.data_plane.feature_store import LocalFeatureStore
  from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
  from runtime import install
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);atomic(root/'contract.json',dict(train_slots=128,validation_slots=8))
   p=root/'native.ckpt.zst';save_zstd_features({'input_ids':torch.tensor([1,2,3])},p)
   w=dict(file=str(p),bytes=p.stat().st_size,sha256=sha256(p))
   atomic(root/'train/000000.json',[w]*128)
   originals=(OfflineManifestReader.__iter__,launch._shard_offline_refs,LocalFeatureStore.get,launch._assemble_trainer)
   try:
    install(root)
    refs=OfflineManifestReader(str(root/'train'),feature_keys=('input_ids',),run_id='stream').read()
    all_ids=[]
    for rank in range(8):
     shard=launch._shard_offline_refs(refs,use_usp_preprocess=False,dp_rank=rank,dp_size=8,shuffle=True)
     self.assertEqual([r.metadata['stream_slot']['index'] for r in shard],list(range(rank,128,8)))
     loader=FeatureDataLoader(LocalFeatureStore(),refs=shard,num_workers=2)
     loader.seek(3);batches=list(loader)
     self.assertEqual([b.sample_ids[0] for b in batches],[r.sample_id for r in shard[3:]])
     self.assertTrue(all(torch.equal(b.tensors['input_ids'],torch.tensor([[1,2,3]])) for b in batches))
     loader.close();all_ids.extend(r.sample_id for r in shard)
    self.assertEqual(len(set(all_ids)),128)
    p.write_bytes(b'x'*p.stat().st_size)
    with self.assertRaisesRegex(AssertionError,'corrupt feature'):LocalFeatureStore().get(refs[0])
   finally:
    OfflineManifestReader.__iter__,launch._shard_offline_refs,LocalFeatureStore.get,launch._assemble_trainer=originals


if __name__=='__main__':unittest.main()
