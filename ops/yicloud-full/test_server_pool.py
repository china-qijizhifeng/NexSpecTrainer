import unittest
from validate_server_pool import validate
class PoolTest(unittest.TestCase):
 def test_original_pool_rejects_234k_but_corrected_pool_fits(self):
  info={'internal_states':[dict(context_length=262144,max_prefill_tokens=262144,chunked_prefill_size=-1,page_size=256,enable_decoder_swa_bounded_replay=False)]}
  def logs(swa):return '\n'.join(f'TP{r} EP{r}] DSV4 pool sizes: full=262144, swa={swa}, c4=65536, c128=2048, c4_state=0, c128_state=0' for r in range(8))
  with self.assertRaisesRegex(AssertionError,'cannot fit'):validate(info,logs(209664),234048)
  self.assertTrue(validate(info,logs(262144),234048)['passed'])
  with self.assertRaisesRegex(AssertionError,'missing effective'):validate(info,'',234048)
  with self.assertRaisesRegex(AssertionError,'compressed KV'):validate(info,logs(262144).replace('c128=2048','c128=1'),234048)
if __name__=='__main__':unittest.main()
