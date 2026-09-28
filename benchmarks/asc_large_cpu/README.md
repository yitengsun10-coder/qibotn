# ASC large-workload QiboTN CPU benchmark

This benchmark lives in the official `qiboteam/qibotn` source tree and uses
the Quimb NumPy CPU backend for every timed variant. No GPU, cuQuantum,
qibojit, or external state-vector execution path is used.

## Scope and contribution

- Framework capability: QiboTN converts a Qibo circuit to a Quimb MPS and
  evaluates a symbolic expectation value on CPU.
- Submitted source fix: repair the expectation path so it reads the configured
  contraction-optimizer attribute.
- Submitted experiment: build and profile a deterministic large workload,
  reject inaccurate numerical shortcuts, locate CPU thread overhead, and test
  a matching thread-pool optimization under controlled conditions.

The earlier 16/18/20-qubit experiment is only an environment check. It is not
the final high-pressure result.

## Workloads

| Profile | Qubits | Layers | Gates | Depth | Use |
|---|---:|---:|---:|---:|---|
| quick | 12 | 5 | 187 | 21 | installation check |
| tune | 25 | 67 | 4,983 | 269 | parameter screening |
| formal | 30 | 112 | 9,998 | 449 | high-pressure result |
| scale | 20/25/30 | 33/67/112 | 1,967/4,983/9,998 | 133/269/449 | scaling evidence |

Every result records the circuit QASM SHA-256 so baseline and optimized runs
can be audited as the same workload. A 30-qubit complex128 dense state alone
would require 16 GiB. The benchmark records scalar observables and validates
candidate numerical approximations by directly contracting normalized MPS
overlaps, avoiding dense-state materialization.

## Reproduction

Use Python 3.11 and install this fork in an isolated environment. Start with a
quick check:

```bash
python benchmarks/asc_large_cpu/benchmark.py \
  --output results/quick --profile quick --repeats 2 --threads 1
```

Screen parameters on 25 qubits:

```bash
python benchmarks/asc_large_cpu/benchmark.py \
  --output results/tune_q25 --profile tune --repeats 3 --threads 1
```

The 25-qubit scalar screen suggested that a relaxed SVD cutoff might help, but
the 30-qubit state-overlap test rejected every relaxed candidate. Reproduce
that negative result before accepting any approximate configuration:

```bash
python benchmarks/asc_large_cpu/fidelity_scan.py \
  --output results/fidelity_scan_q30.json --qubits 30 --layers 112 \
  --variant baseline --variant cutoff_1e10_b128 \
  --variant cutoff_1e8_b128 --variant cutoff_1e7_b128 \
  --min-fidelity 0.999
```

The accepted formal optimization keeps `cutoff=1e-12`, `max_bond=128`,
precision, backend, circuit, observable, and contraction optimizer unchanged.
An environment-variable-free probe reported a 20-thread OpenBLAS default; the
optimized run controls the pool at one thread. Each setting runs in a fresh
process:

```bash
python benchmarks/asc_large_cpu/thread_sweep.py \
  --output results/formal_thread_sweep_q30 --qubits 30 --layers 112 \
  --threads 1 20 --repeats 3
```

Each timing run writes `results.csv`, `summary.json`, `environment.json`, and
`run.log`. Cold time is kept separate; speedup uses repetitions after the
first whenever available. Peak RSS, gate count, depth, QASM hash, output
error, and speedup are recorded.

The measured result and interpretation are in `RESULTS.md`. Curated raw
evidence is under `evidence/` and can be checked without rerunning the long
benchmark:

```bash
python benchmarks/asc_large_cpu/verify_evidence.py
```
