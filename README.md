# Qibotn

The tensor network translation module for Qibo to support large-scale simulation of quantum circuits and acceleration.

## ASC large-workload CPU study

Current verified exact-CPU work is under
[`benchmarks/asc_revision_cpu`](benchmarks/asc_revision_cpu/README.md).
Matched one-thread, untruncated complex128 comparisons against pinned official
source (one attribute typo repair only) give **4.868623693443177x** on a
16q/1824-gate mixed input and **1.877080501192603x** on the completed
20q/9991-gate/depth-677 pressure input. Both have independent complete-state
checks. The 20q original time-budget stop and explicit recovery are disclosed.
See the [current results and unfinished items](benchmarks/asc_revision_cpu/PRESSURE_RESULTS_20261009.md).

**Original high-entanglement exact30 acceptance and 7x on that input remain incomplete.**
The old 30q thread comparison in
[`benchmarks/asc_large_cpu`](benchmarks/asc_large_cpu/README.md) is preserved
as history, not accepted accurate optimization: its common truncated state
lost substantial norm. It must not be presented as a current successful 7x
result. The user has since rented a 32-vCPU / 60-GiB CPU instance. Bounded
original-30q probes stopped at their declared time limits, not OOM, and have
not completed acceptance. A new exact controlled-gate rank-two factorization
has completed one predeclared six-bridge 30q/9693-gate/depth-445 chain:
official mean 529.9836281165481 s, integrated mean 334.01531983166933 s
(1.5867045511075333x), research candidate mean 2.8484130998452506 s
(186.06276882568093x). Independent full-state and explicit-continuation audits
pass with identical QR thread scheduling. The candidate is not integrated
into production; this bounded-entanglement supplementary input does not
replace the original high-entanglement input or establish a general 7x claim.
Ring/nonlocal supplementary pairs are still incomplete. See the
[724 stage record](benchmarks/asc_revision_cpu/SERVER724_RANK2_STAGE_20261009.md)
for scope, raw-evidence locations and remaining work.

## Supported Computation

Tensor Network Types:

- Tensornet (TN)
- Matrix Product States (MPS)

Tensor Network contractions to:

- dense vectors
- expecation values of given Pauli string

The supported HPC configurations are:

- single-node CPU
- single-node GPU or GPUs
- multi-node multi-GPU with Message Passing Interface (MPI)
- multi-node multi-GPU with NVIDIA Collective Communications Library (NCCL)

Currently, the supported tensor network libraries are:

- [cuQuantum](https://github.com/NVIDIA/cuQuantum), an NVIDIA SDK of optimized libraries and tools for accelerating quantum computing workflows.
- [quimb](https://quimb.readthedocs.io/en/latest/), an easy but fast python library for ‘quantum information many-body’ calculations, focusing primarily on tensor networks.

## Installation

To get started:

```sh
pip install qibotn
```

to install the tools and dependencies. A few extras are provided, check `pyproject.toml` in
case you need them.

<!-- TODO: describe extras, after Poetry adoption and its groups -->

## Contribute

To contribute, please install using poetry:

```sh
git clone https://github.com/qiboteam/qibotn.git
cd qibotn
poetry install
```

## Sample Codes

### Single-Node Example

The code below shows an example of how to activate the Cuquantum TensorNetwork backend of Qibo.

```py
import numpy as np
from qibo import Circuit, gates
import qibo

# Below shows how to set the computation_settings
# Note that for MPS_enabled and expectation_enabled parameters the accepted inputs are boolean or a dictionary with the format shown below.
# If computation_settings is not specified, the default setting is used in which all booleans will be False.
# This will trigger the dense vector computation of the tensornet.

computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": {
        "qr_method": False,
        "svd_method": {
            "partition": "UV",
            "abs_cutoff": 1e-12,
        },
    },
    "NCCL_enabled": False,
    "expectation_enabled": False,
}


qibo.set_backend(
    backend="qibotn", platform="cutensornet", runcard=computation_settings
)  # cuQuantum
# qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings) #quimb


# Construct the circuit
c = Circuit(2)
# Add some gates
c.add(gates.H(0))
c.add(gates.H(1))

# Execute the circuit and obtain the final state
result = c()

print(result.state())
```

Other examples of setting the computation_settings

```py
# Expectation computation with specific Pauli String pattern
computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": {
        "pauli_string_pattern": "IXZ",
    },
}

# Dense vector computation using multi node through MPI
computation_settings = {
    "MPI_enabled": True,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
}
```

### Multi-Node Example

Multi-node is enabled by setting either the MPI or NCCL enabled flag to True in the computation settings. Below shows the script to launch on 2 nodes with 2 GPUs each. $node_list contains the IP of the nodes assigned.

```sh
mpirun -n 4 -hostfile $node_list python test.py
```
