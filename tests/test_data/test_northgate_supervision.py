"""Check exact official rendering, token boundaries, and preserved tool supervision."""
import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from ops.northgate_capture import prepare


@unittest.skipUnless(os.environ.get('NORTHGATE_MODEL_PATH'), 'requires official encoder and tokenizer')
class TestNorthgateSupervision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare.initialize(os.environ['NORTHGATE_MODEL_PATH'])

    def row(self):
        text='重复的 Unicode reasoning，second occurrence。'
        return {
            'id':'test','tools':[{'type':'function','function':{'name':'run','parameters':{'type':'object'}}}],
            'messages':[
                {'role':'developer','content':'保留这个 developer instruction'},
                {'role':'user','content':'use the tool'},
                {'role':'assistant','reasoning_content':text,'content':'visible answer','tool_calls':[
                    {'id':'c1','type':'function','function':{'name':'run','arguments':'{"x":"中"}'}}]},
                {'role':'tool','tool_call_id':'c1','content':'actual tool result'},
                {'role':'system','content':'midstream instruction'},
                {'role':'assistant','reasoning_content':'summary reasoning','content':'masked summary'}],
            'metadata':{'curation':{'source_record_sha256':'test'},'training_supervision':{
                'message_loss_mask':[0,0,1,0,0,0], 'excluded_summary_message_indexes':[5],
                'reasoning_loss_mask_exclusions':[{'message_index':2,'field':'reasoning_content',
                    'start':0,'end':len(text),'loss_mask':0,'sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}]}}}

    def test_reasoning_only_mask_preserves_context_and_calls(self):
        row=self.row();result=prepare.encode_row(row,verify=True)
        messages=prepare.messages_for_encoding(row)
        from specforge.data.v41_supervision import encode_supervision
        ids,mask,calls=encode_supervision(prepare._ENCODER,prepare._TOKENIZER,messages,
            missing_use_loss=False,thinking_mode='thinking',drop_thinking=False,reasoning_effort=75)
        text=prepare._ENCODER.encode_messages(copy.deepcopy(messages),thinking_mode='thinking',drop_thinking=False,reasoning_effort=75)
        encoded=prepare._TOKENIZER.encode(text,add_special_tokens=False)
        for fragment in ['重复的 Unicode reasoning，second occurrence。','masked summary','summary reasoning','actual tool result','midstream instruction','保留这个 developer instruction']:
            a=text.index(fragment);b=a+len(fragment)
            indexes=[i for i,(x,y) in enumerate(encoded.offsets) if x<b and y>a]
            self.assertTrue(indexes);self.assertFalse(mask[indexes].any(),fragment)
        a=text.index('visible answer');b=a+len('visible answer')
        indexes=[i for i,(x,y) in enumerate(encoded.offsets) if x>=a and y<=b and y>x]
        self.assertTrue(mask[indexes].all())
        self.assertTrue((calls>0).any());self.assertTrue(mask[calls>0].all())
        self.assertEqual(len(result['input_ids']),len(encoded.ids))

    def test_partial_reasoning_exclusion_and_invalid_boundaries(self):
        row=self.row();span=row['metadata']['training_supervision']['reasoning_loss_mask_exclusions'][0]
        span['start']=4;span['end']=11
        rc=row['messages'][2]['reasoning_content'];span['sha256']=__import__('hashlib').sha256(rc[4:11].encode()).hexdigest()
        prepare.encode_row(row,verify=True)
        messages=prepare.messages_for_encoding(row)
        messages[2]['reasoning_loss_mask_exclusions'][0]['end']=len(rc)+1
        from specforge.data.v41_supervision import encode_supervision
        with self.assertRaisesRegex(ValueError,'nonoverlapping'):
            encode_supervision(prepare._ENCODER,prepare._TOKENIZER,messages,
                missing_use_loss=False,thinking_mode='thinking',drop_thinking=False)


class TestCaptureLimits(unittest.TestCase):
    def test_explicit_bounds_are_validated_without_large_allocation(self):
        for value,valid in [(None,True),(str(16<<30),True),(str(32<<30),True),('0',False),('-1',False),(str((32<<30)+1),False),('oops',False)]:
            env=dict(os.environ);env.pop('SPECFORGE_MAX_CAPTURE_RECORD_BYTES',None)
            if value is not None:env['SPECFORGE_MAX_CAPTURE_RECORD_BYTES']=value
            code='from specforge.data.capture_limits import MAX_CAPTURE_RECORD_BYTES; print(MAX_CAPTURE_RECORD_BYTES)'
            result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode==0,valid,value)
