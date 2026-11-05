"""Tokenize complete curated traces with official V4.1 rendering and masks."""
import argparse
import collections
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from specforge.data.v41_supervision import encode_supervision

REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
_ENCODER = _TOKENIZER = None


def initialize(model):
    global _ENCODER, _TOKENIZER
    path = Path(model)
    spec = importlib.util.spec_from_file_location('official_v41', path / 'encoding/encoding.py')
    _ENCODER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ENCODER)
    _TOKENIZER = Tokenizer.from_file(str(path / 'tokenizer.json'))
    _TOKENIZER.no_truncation()
    _TOKENIZER.no_padding()


def messages_for_encoding(row):
    messages = copy.deepcopy(row['messages'])
    supervision = row['metadata']['training_supervision']
    masks = supervision['message_loss_mask']
    exclusions = collections.defaultdict(list)
    for span in supervision['reasoning_loss_mask_exclusions']:
        assert span['field'] == 'reasoning_content' and span['loss_mask'] == 0
        text = messages[span['message_index']]['reasoning_content'][span['start']:span['end']]
        assert hashlib.sha256(text.encode()).hexdigest() == span['sha256']
        exclusions[span['message_index']].append(span)
    for index, message in enumerate(messages):
        assert message['role'] in ('system', 'developer', 'user', 'assistant', 'tool')
        assert isinstance(message.get('content') or '', str)
        message['source_message_index'] = index
        message['use_loss'] = bool(masks[index])
        for call in message.get('tool_calls') or []:
            call['use_loss'] = bool(masks[index])
        for block in message.get('content_blocks') or []:
            if block.get('type') == 'text':block['use_loss'] = bool(masks[index])
        message['reasoning_loss_mask_exclusions'] = exclusions[index]
        if message['role'] == 'developer':
            message['role'] = 'system'  # Official V4.1 has a System token and no Developer token.
    if row['tools']:
        if not messages or messages[0]['role'] != 'system':
            messages.insert(0, {'role': 'system', 'content': '', 'use_loss': False})
        messages[0]['tools'] = copy.deepcopy(row['tools'])
    return messages


def encode_row(row, *, verify=False):
    messages = messages_for_encoding(row)
    options = dict(thinking_mode='thinking', drop_thinking=False, reasoning_effort=75)
    ids, mask, calls = encode_supervision(_ENCODER, _TOKENIZER, messages,
                                        missing_use_loss=False, **options)
    assert len(ids) == len(mask) and mask.any()
    if verify:
        # Independent unannotated rendering must preserve every token.
        plain = _ENCODER.encode_messages(copy.deepcopy(messages), **options)
        expected = _TOKENIZER.encode(plain, add_special_tokens=False)
        assert ids.tolist() == expected.ids
        baseline = copy.deepcopy(messages)
        for message in baseline:message.pop('reasoning_loss_mask_exclusions', None)
        clean_ids, clean_mask, clean_calls = encode_supervision(
            _ENCODER, _TOKENIZER, baseline, missing_use_loss=False, **options)
        assert np.array_equal(ids, clean_ids) and np.array_equal(calls, clean_calls)
        assert np.all(mask <= clean_mask)
        assert np.array_equal(mask[calls > 0], clean_mask[calls > 0])
    metadata = {
        'model_revision': REVISION,
        'source_record_sha256': row['metadata']['curation']['source_record_sha256'],
        'curation': row['metadata']['curation'],
        'supervision': {'assistant_messages': sum(row['metadata']['training_supervision']['message_loss_mask']),
                        'excluded_summary_indexes': row['metadata']['training_supervision']['excluded_summary_message_indexes'],
                        'reasoning_exclusion_spans': len(row['metadata']['training_supervision']['reasoning_loss_mask_exclusions']),
                        'supervised_tokens': int(mask.sum()), 'supervised_tool_call_tokens': int(mask[calls > 0].sum())},
        'rendering': 'official_v41_thinking_drop_thinking_false_effort75',
        'developer_role_mapping': 'system_preserve_content_and_position',
        'token_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'loss_mask_sha256': hashlib.sha256(mask.astype(np.int64).tobytes()).hexdigest(),
    }
    return {'id': row['id'], 'input_ids': ids.tolist(), 'loss_mask': mask.astype(np.int8).tolist(),
            'metadata': metadata}


def prepare_shard(arguments):
    source, target, expected_sha, expected_bytes = arguments
    source, target = Path(source), Path(target)
    deadline=time.monotonic()+3600
    while not source.exists() or source.stat().st_size!=expected_bytes:
        if (source.parent.parent/'TRANSFER_COMPLETE').exists():
            raise ValueError(f'transferred shard has wrong size: {source}')
        if time.monotonic()>deadline:raise TimeoutError(f'waiting for source shard: {source}')
        time.sleep(2)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):digest.update(block)
    assert digest.hexdigest() == expected_sha, source
    entries=[];total=0
    partial=target.with_suffix('.partial')
    with source.open() as stream, gzip.open(partial, 'wt', compresslevel=1) as output:
        for line_number, line in enumerate(stream, 1):
            row=json.loads(line)
            result=encode_row(row, verify=line_number <= 2)
            n=len(result['input_ids']);assert n<=1048576, (row['id'], n)
            output.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
            entries.append({'id':result['id'],'tokens':n,'file':target.name,'line':line_number,
                            'metadata':result['metadata']})
            total+=n
    partial.replace(target)
    target.with_suffix('.index.json').write_text(json.dumps(entries,ensure_ascii=False))
    print(json.dumps({'shard':target.name,'records':len(entries),'tokens':total}),flush=True)
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();initialize(args.model)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    if args.probe_only:
        with (args.dataset/manifest['shards'][0]['file']).open() as stream:row=json.loads(next(stream))
        result=encode_row(row,verify=True)
        print(json.dumps({'probe':'passed','id':result['id'],'tokens':len(result['input_ids']),
                          'supervision':result['metadata']['supervision']}))
        return
    args.output.mkdir(exist_ok=False)
    tasks=[(str(args.dataset/s['file']),str(args.output/(Path(s['file']).name+'.gz')),s['sha256'],s['bytes']) for s in manifest['shards']]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(args.model),)) as pool:
        entries=[entry for shard in pool.map(prepare_shard,tasks) for entry in shard]
    assert len(entries)==manifest['counts']['records'] and len({e['id'] for e in entries})==len(entries)
    # Greedy token balance, deterministic ties, disjoint complete trajectories.
    partitions=[[] for _ in range(4)];totals=[0]*4
    for entry in sorted(entries,key=lambda x:(-x['tokens'],x['id'])):
        worker=min(range(4),key=lambda i:(totals[i],i));entry['worker']=worker
        partitions[worker].append(entry);totals[worker]+=entry['tokens']
    # Begin each worker with its shortest real complete trace for read-back acceptance.
    for partition in partitions:partition.sort(key=lambda x:(x['tokens'],x['id']))
    assignments={e['id']:e['worker'] for e in entries}
    handles=[gzip.open(args.output/f'worker-{i}.jsonl.gz.partial','wt',compresslevel=1) for i in range(4)]
    # Materialize sorted partitions using seekable temporary per-record offsets in stage JSONL.
    spool=args.output/'spool.jsonl';offsets={}
    with spool.open('wb') as out:
        for source,stage,_,_ in tasks:
            with gzip.open(stage,'rb') as stream:
                for raw in stream:
                    row=json.loads(raw);offsets[row['id']]=out.tell();out.write(raw)
    with spool.open('rb') as stream:
        for i,partition in enumerate(partitions):
            for entry in partition:
                stream.seek(offsets[entry['id']]);handles[i].write(stream.readline().decode())
            handles[i].close()
            (args.output/f'worker-{i}.jsonl.gz.partial').replace(args.output/f'worker-{i}.jsonl.gz')
    spool.unlink()
    maximum=max(e['tokens'] for e in entries)
    result={'records':len(entries),'tokens':sum(totals),'max_tokens':maximum,'worker_tokens':totals,
            'worker_records':[len(x) for x in partitions],'max_decoded_bytes':maximum*(40960+16),
            'model_revision':REVISION,'samples':entries,'source_manifest':manifest,
            'encoding_files':{name:hashlib.sha256((args.model/name).read_bytes()).hexdigest()
                              for name in ['encoding/encoding.py','tokenizer.json','config.json']}}
    (args.output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'COMPLETE').write_text('verified\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples','source_manifest']}),flush=True)

if __name__=='__main__':main()
