"""Probe package contracts; --cpu skips CUDA-dependent extension imports."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import google.protobuf
from google.protobuf.internal.api_implementation import Type

from sglang.srt.model_executor.forward_observer import ForwardObserver
from sglang.srt.managers.deferred_output import ResponseCompletion
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpu',action='store_true')
args=parser.parse_args()
root=Path(importlib.util.find_spec('sglang').origin).parent
files={}
for path in ['srt/configs/deepseek_v41.py','srt/models/deepseek_v4.py',
             'srt/model_executor/forward_observer.py','srt/managers/deferred_output.py']:
 q=root/path
 assert q.is_file(),path
 files[path]=hashlib.sha256(q.read_bytes()).hexdigest()
assert 'hc_pre_from_prev_sublayer = True' in (root/'srt/configs/deepseek_v41.py').read_text()
mooncake_spec=importlib.util.find_spec('mooncake')
assert mooncake_spec is not None
mooncake_root=Path(next(iter(mooncake_spec.submodule_search_locations)))
assert any(mooncake_root.glob('store*.so'))
extension_import='not attempted on CPU; requires GPU driver'
if not args.cpu:
 assert torch.cuda.is_available(),'GPU worker must expose CUDA'
 assert torch.cuda.device_count()==8
 assert all(torch.cuda.get_device_capability(i)==(9,0) for i in range(8))
 from mooncake.store import MooncakeDistributedStore
 assert all(callable(getattr(MooncakeDistributedStore,k,None)) for k in ['setup','get_into','put_from','remove','is_exist'])
 extension_import='passed; method contract checked'
assert shutil.which('mooncake_master')
assert any(ep.name=='specforge_spec_capture' for ep in md.entry_points(group='sglang.srt.plugins'))
versions={}
for name in ['torch','sglang','transformers','flashinfer-python','mooncake-transfer-engine','mooncake-transfer-engine-cuda13']:
 try: versions[name]=md.version(name)
 except md.PackageNotFoundError: versions[name]=None
print(json.dumps({'versions':versions,'python':sys.version,'protobuf':google.protobuf.__version__,'protobuf_backend':Type(),'torch_path':torch.__file__,'sglang_root':str(root),'files_sha256':files,'gpu_available':torch.cuda.is_available(),'mooncake_extension_import':extension_import,'scope':'CPU API probe only' if args.cpu else 'GPU package/extension probe; no target forward'},indent=2))
