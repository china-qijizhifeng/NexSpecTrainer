"""Supervise the receipt publisher and trainer together; either failure aborts."""
import json
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from schedule import atomic

root=Path(os.environ['RUN_DIR']); stream=Path(os.environ['STREAM_ROOT']); here=Path(__file__).resolve().parent
lock=(stream/'tuned-publisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
publisher=trainer=None
try:
 with (root/'publisher.log').open('a') as log:
  started=time.time()
  publisher=subprocess.Popen([sys.executable,str(here/'schedule.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while not (stream/'status.json').exists() or json.loads((stream/'status.json').read_text()).get('time',0) < started:
   if publisher.poll() is not None: raise RuntimeError(f'publisher ended before ready: {publisher.returncode}')
   time.sleep(1)
  trainer=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(here/'audit_train.py'),'train','--config',str(root/'config.yaml')],start_new_session=True)
  while trainer.poll() is None:
   if publisher.poll() not in (None,0): raise RuntimeError(f'publisher failed: {publisher.returncode}')
   time.sleep(1)
  if trainer.returncode: raise RuntimeError(f'trainer failed: {trainer.returncode}')
  assert publisher.wait(timeout=60)==0 and (stream/'complete.json').exists()
  subprocess.run([sys.executable,str(here/'verify_train.py')],check=True)
except BaseException as exc:
 atomic(stream/'failed.json',{'error':repr(exc),'time':time.time()})
 raise
finally:
 for process in (trainer,publisher):
  if process is not None and process.poll() is None:
   os.killpg(process.pid,signal.SIGTERM)
   try: process.wait(timeout=30)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGKILL); process.wait()
