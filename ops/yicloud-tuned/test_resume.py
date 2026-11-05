"""Verify replay preservation and restart failure boundaries using real receipts."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
import schedule
from schedule import Publisher,atomic
spec=importlib.util.spec_from_file_location('prior_tests',Path(__file__).parents[1]/'yicloud-stream/test_stream.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

class ResumeTests(unittest.TestCase):
 setUp=prior.ScheduleTests.setUp
 receipt=prior.ScheduleTests.receipt
 test_late_slot_and_fail_closed=prior.ScheduleTests.test_late_slot_and_fail_closed
 test_uncommitted_receipt_and_capture_failure=prior.ScheduleTests.test_uncommitted_receipt_and_capture_failure
 test_wrong_metadata_and_file_size_rejected=prior.ScheduleTests.test_wrong_metadata_and_file_size_rejected
 def test_resume_keeps_published_order_and_full_coverage(self):
  for i in range(256):self.receipt(i)
  first=Publisher(self.capture,self.out,self.plan,reserve=128);first.tick()
  saved=(self.out/'train/000000.json').read_bytes()
  new=self.root/'new';new.mkdir();os.environ['PREVIOUS_CAPTURE']=str(self.capture)
  atomic(new/'run.json',{'yicloud':{}})
  for rank in range(4):
   d=new/f'capture-w{rank}/sliced';d.mkdir(parents=True)
   for r in (self.capture/f'capture-w{rank}/sliced').glob('*.json'):(d/r.name).symlink_to(r)
  second=Publisher(new,self.out,self.plan,reserve=128)
  self.assertEqual(second.blocks,first.blocks);self.assertEqual(second.published,first.published)
  for i in range(256,384):
   self.receipt(i);r=self.capture/f"capture-w{i%4}/sliced"/f's{i:04d}.json';(new/f'capture-w{i%4}/sliced'/r.name).symlink_to(r)
  for rank in range(4):atomic(new/f'capture-w{rank}/RESULT.json',dict(passed=True,complete_partition=True,worker=rank))
  atomic(new/'run.json',{'yicloud':{'terminal_phase':'Succeeded'}});second.tick()
  self.assertEqual(saved,(self.out/'train/000000.json').read_bytes())
  self.assertTrue(second.finished);self.assertEqual(len(second.blocks),3)
  self.assertEqual(len(second.published),368)

if __name__=='__main__':unittest.main()
