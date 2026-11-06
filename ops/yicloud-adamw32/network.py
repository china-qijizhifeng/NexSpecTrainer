"""Bind bootstrap transports to this replica's platform-published local address."""

import json
import os
import shlex
import socket
import time
from pathlib import Path


def select_interface(rank, nodes, members, own_ip, master_ip, addresses):
    if len(members) != nodes or len(set(members)) != nodes:
        raise ValueError(
            "platform membership must contain one unique address per replica"
        )
    if own_ip not in members or master_ip not in members:
        raise ValueError(
            "replica/master DNS addresses are absent from platform membership"
        )
    if rank == 0 and own_ip != master_ip:
        raise ValueError("replica zero does not resolve to the master address")
    interfaces = [name for name, ips in addresses.items() if own_ip in ips]
    if len(interfaces) != 1:
        raise ValueError(
            f"expected exactly one UP local interface for {own_ip}, found {interfaces}"
        )
    return interfaces[0]


def resolve(name):
    for attempt in range(60):
        try:
            return socket.gethostbyname(name)
        except socket.gaierror:
            time.sleep(2)
    raise RuntimeError(f"DNS did not resolve {name}")


def main():
    import psutil

    root = Path(os.environ["NODE_ROOT"])
    rank = int(os.environ["KUBEBRAIN_REPLICA"])
    nodes = int(os.environ["KUBEBRAIN_REPLICA_TOTAL"])
    assert nodes == 4 and 0 <= rank < nodes
    prefix = os.environ["KUBEBRAIN_HOSTNAME_PREFIX"]
    namespace = os.environ["YICLOUD_RENDEZVOUS_NAMESPACE"]
    own = f"{prefix}-{rank}.{namespace}.svc.lg.shzhisuan.local"
    master = os.environ["TRAIN_MASTER_ADDR"]
    members = [x.strip() for x in os.environ["WORKER_ALL_SOCKET_ADDRS"].split(",")]
    status = psutil.net_if_stats()
    addresses = {
        name: [a.address for a in rows if a.family == socket.AF_INET]
        for name, rows in psutil.net_if_addrs().items()
        if name in status and status[name].isup
    }
    receipt = {
        "node_rank": rank,
        "nodes": nodes,
        "hostname": socket.gethostname(),
        "own_service": own,
        "master_addr": master,
        "members": members,
        "membership_not_order": True,
        "local_ipv4_interfaces": addresses,
        "nccl_hca": os.environ.get("NCCL_IB_HCA"),
    }
    path = root / "rendezvous.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    own_ip = resolve(own)
    master_ip = resolve(master)
    receipt.update(local_ip=own_ip, master_ip=master_ip)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    interface = select_interface(rank, nodes, members, own_ip, master_ip, addresses)
    receipt.update(bootstrap_interface=interface, passed=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    # The shell uses this only for NCCL/Gloo bootstrap; the active-IB selector
    # supplied by the runtime remains authoritative for the data transport.
    (root / "network.env").write_text(
        "export GLOO_SOCKET_IFNAME="
        + shlex.quote(interface)
        + "\nexport NCCL_SOCKET_IFNAME="
        + shlex.quote("=" + interface)
        + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
