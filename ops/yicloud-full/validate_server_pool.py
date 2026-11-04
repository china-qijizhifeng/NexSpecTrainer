"""Reject an unschedulable full prefill before submitting the capture request."""
import json,math,os,re
from pathlib import Path

def validate(info, log, tokens):
 states=info.get('internal_states') or [info]
 for state in states:
  assert state['context_length']>=tokens+6, 'context too small'
  assert state['max_prefill_tokens']>=tokens, 'prefill limit too small'
  assert state['chunked_prefill_size']==-1, 'full trajectory requires unchunked prefill'
  assert not state.get('enable_decoder_swa_bounded_replay',False)
 # SGLang reserves page-alignment overhead beyond the rounded input length.
 page=int(states[0]['page_size']);need=math.ceil(tokens/page)*page+2*page
 matches=re.findall(r'TP(\d+).*DSV4 pool sizes: full=(\d+), swa=(\d+), c4=(\d+), c128=(\d+)',log)
 pools={int(rank):tuple(map(int,values)) for rank,*values in matches}
 assert set(pools)==set(range(8)), 'missing effective pool report from a TP rank'
 for rank,(full,swa,c4,c128) in pools.items():
  assert full>=need and swa>=need, f'rank {rank}: full/SWA pools {full}/{swa} cannot fit {tokens} tokens plus alignment reserve {need}'
  assert c4>=math.ceil(need/4) and c128>=math.ceil(need/128), f'rank {rank}: compressed KV pool too small'
 return dict(passed=True,tokens=tokens,required_pool_tokens=need,pools=pools,scope='effective server context and all TP full/SWA/compressed pools, before request')

def main():
 root=Path(os.environ['RUN_DIR'])
 check=root/'config-preflight.json'
 if check.exists():tokens=json.loads(check.read_text())['tokens']
 else:tokens=json.loads((root/'input-check.json').read_text())['max_tokens']
 result=validate(json.loads((root/'capture/server-info.json').read_text()),(root/'capture/serve.log').read_text(),tokens)
 (root/'capture/pool-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
