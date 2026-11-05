import unittest
from network import select_interface

class RendezvousContract(unittest.TestCase):
    def test_platform_ip_on_second_interface_and_shuffled_membership(self):
        members=['10.100.6.186','10.100.5.193','10.100.30.143','10.100.26.168']
        self.assertEqual(select_interface(1,4,members,'10.100.5.193','10.100.26.168',{'bond0':['10.100.4.193'],'eth0':['10.100.5.193']}),'eth0')
        self.assertEqual(select_interface(1,4,list(reversed(members)),'10.100.5.193','10.100.26.168',{'bond0':['10.100.4.193'],'eth0':['10.100.5.193']}),'eth0')
    def test_rank_zero_uses_its_own_service(self):
        self.assertEqual(select_interface(0,4,['d','c','b','a'],'a','a',{'bond0':['a']}),'bond0')
        with self.assertRaises(ValueError):select_interface(0,4,['a','b','c','d'],'a','b',{'bond0':['a']})
    def test_rejects_missing_duplicate_and_nonlocal_addresses(self):
        for members,own,interfaces in [(['a','b','c','c'],'a',{'bond0':['a']}),(['a','b','c','d'],'e',{'bond0':['e']}),(['a','b','c','d'],'b',{'bond0':['a']}),(['a','b','c','d'],'b',{'eth0':['b'],'bond0':['b']})]:
            with self.assertRaises(ValueError):select_interface(1,4,members,own,'a',interfaces)

if __name__=='__main__':unittest.main()
