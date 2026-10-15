import os
import socket

import dpctl


local_rank = os.environ.get(
    "PALS_LOCAL_RANKID",
    os.environ.get("MPI_LOCALRANKID", os.environ.get("PMI_LOCAL_RANK", "?")),
)
device = dpctl.SyclDevice()
print(
    f"host={socket.gethostname()} local_rank={local_rank} "
    f"mask={os.environ.get('ZE_AFFINITY_MASK')} device={device.name}",
    flush=True,
)

