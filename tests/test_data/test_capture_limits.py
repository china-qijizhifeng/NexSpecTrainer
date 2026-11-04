"""Large trajectory metadata must be rejected before allocation unless opted in."""
import os
import subprocess
import sys
import unittest

PROBE = '''
from types import SimpleNamespace
from scripts.prepare_server_hidden_states import check_capacity
from specforge.data.feature_file import _validate_entries, _MAX_RECORD_BYTES
from specforge.data.wire_codec import DEFAULT_MAX_DECODED_BYTES
from specforge.data.capture_limits import MAX_CAPTURE_RECORD_BYTES
assert MAX_CAPTURE_RECORD_BYTES == _MAX_RECORD_BYTES == DEFAULT_MAX_DECODED_BYTES
n=234048
capture=SimpleNamespace(target_hidden_size=5120,aux_hidden_state_layer_ids=(37,38,39))
layout=SimpleNamespace(aux_feature=True,last_hidden_feature=True,passthrough=('input_ids','loss_mask'))
expected=n*(15360*2+5120*2+16)
header={'version':1,'tensors':[{'name':name,'shape':[1,n,width],'dtype':'bfloat16','interleave_layers':None} for name,width in [('aux',15360),('last',5120)]]}
if MAX_CAPTURE_RECORD_BYTES < expected:
    try: check_capacity(n,capture=capture,layout=layout,client_bytes=16<<30,server_bytes=16<<30)
    except ValueError as exc: assert 'decoded-record limit' in str(exc)
    else: raise AssertionError('oversized capture accepted')
    try: _validate_entries(header, _MAX_RECORD_BYTES)
    except ValueError as exc: assert 'byte limit' in str(exc)
    else: raise AssertionError('oversized disk record accepted')
else:
    assert check_capacity(n,capture=capture,layout=layout,client_bytes=16<<30,server_bytes=16<<30)==expected
    _validate_entries(header,_MAX_RECORD_BYTES)
    try: check_capacity(n,capture=capture,layout=layout,client_bytes=8<<30,server_bytes=8<<30)
    except ValueError as exc: assert 'uncompressed capture needs' in str(exc)
    else: raise AssertionError('undersized transport accepted')
header['tensors'][0]['shape']=[1,1000000,15360]
try: _validate_entries(header,_MAX_RECORD_BYTES)
except ValueError: pass
else: raise AssertionError('unbounded record accepted')
'''
class CaptureLimitsTest(unittest.TestCase):
 def test_default_and_explicit_large_record_policy(self):
  for value in (None,str(16<<30)):
   env=dict(os.environ);env.pop('SPECFORGE_MAX_CAPTURE_RECORD_BYTES',None)
   if value is not None:env['SPECFORGE_MAX_CAPTURE_RECORD_BYTES']=value
   r=subprocess.run([sys.executable,'-c',PROBE],env=env,capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stderr)
 def test_invalid_policy_fails_before_capture(self):
  for value in ('0','-1',str((16<<30)+1),'unlimited'):
   r=subprocess.run([sys.executable,'-c','import specforge.data.capture_limits'],env=dict(os.environ,SPECFORGE_MAX_CAPTURE_RECORD_BYTES=value),capture_output=True)
   self.assertNotEqual(r.returncode,0)
if __name__=='__main__':unittest.main()
