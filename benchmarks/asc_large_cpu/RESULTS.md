# ASC QiboTN large-workload CPU result

## Conclusion

The final measured workload contains **30 qubits, 9,998 gates, and circuit
depth 449**. It runs through QiboTN's Quimb NumPy CPU backend; no GPU,
cuQuantum, qibojit, or external state-vector timing path is used.

The accepted personal optimization sets the CPU math-library thread pool to
one thread instead of the machine's 16-thread default. With all numerical
settings and the circuit fixed, steady-state runtime fell from **324.6081 s**
to **57.5065 s**, a **5.6447x speedup**. Peak process RSS fell from
**288.93 MiB** to **267.36 MiB**. The absolute output difference was
**5.16e-16**, below the `1e-12` tolerance.

## Controlled formal comparison

| Item | 16-thread baseline | 1-thread optimized |
|---|---:|---:|
| Qubits / gates / depth | 30 / 9,998 / 449 | 30 / 9,998 / 449 |
| Repetitions | 3 | 3 |
| Steady mean (repeats 2-3) | 324.6081 s | 57.5065 s |
| Steady standard deviation | 13.5710 s | 0.4357 s |
| Peak process RSS | 288.93 MiB | 267.36 MiB |
| ZZ expectation | 3.6185198e-11 | 3.6184682e-11 |
| Absolute output difference | 5.16e-16 | reference |
| `svd_cutoff` / max bond | 1e-12 / 128 | 1e-12 / 128 |
| Contraction optimizer | auto-hq | auto-hq |

Both runs used QASM SHA-256
`1ea5c95ebdcc651a1291dadfb7a193009377732bba8d03a01ac37ef7b82b37d4`.
The identical hash checks that both configurations received the same circuit.
The first execution is reported separately as cold time; speedup uses the mean
of repetitions 2 and 3 so one-time initialization does not dominate.

## Bottleneck diagnosis

A 25-qubit screening sweep showed that more CPU threads made the same
conservative calculation slower: 1, 2, 4, 8, and 16 threads took approximately
23.69, 41.15, 53.10, 81.95, and 128.61 seconds respectively for one run. The
workload repeatedly invokes relatively small and medium tensor/SVD operations.
Creating and synchronizing a large BLAS/OpenMP thread team costs more than the
extra parallel arithmetic saves, so thread overhead and oversubscription are
the evidenced bottleneck on this machine.

The matching intervention sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` before
Python imports NumPy/Quimb. The launcher starts each setting in a fresh process
so an initialized thread pool cannot contaminate the next result.

## Rejected optimization and correctness lesson

Relaxing SVD truncation looked acceptable when only a near-zero scalar ZZ
expectation was compared. A normalized MPS-overlap check on the formal circuit
disproved that conclusion:

| Variant | Fidelity vs conservative baseline | Required | Decision |
|---|---:|---:|---|
| cutoff 1e-10, bond 128 | 0.451171 | 0.999 | reject |
| cutoff 1e-8, bond 128 | 0.361843 | 0.999 | reject |
| cutoff 1e-7, bond 128 | 0.280895 | 0.999 | reject |

These candidates remain as negative evidence and are not counted as personal
performance gains. The formal speedup changes no numerical approximation.

## Framework capability versus personal contribution

- **Framework capability:** QiboTN converts a Qibo circuit to Quimb MPS and
  computes symbolic expectation values on CPU.
- **Personal source fix:** the expectation path now reads the configured
  `contractions_optimizer` attribute; a regression test checks the Bell-state
  ZZ expectation.
- **Personal analysis:** construct a deterministic high-pressure workload,
  record hashes/timings/memory, identify thread overhead, reject inaccurate
  truncation by state fidelity, and validate the final setting repeatedly.

## Evidence and limits

- `evidence/formal_thread_sweep_q30/`: raw CSV, logs, environment captures,
  and summaries for the formal 1-thread and 16-thread runs.
- `evidence/fidelity_scan_q30.json`: state-level cutoff accuracy scan.
- `verify_evidence.py`: offline checks for workload size, circuit identity,
  repetitions, correctness tolerance, and derived speedup.

The captured environment files show `profile: quick` because the launcher used
an explicit `--workload 30:112`, which overrides the parser's historical
default profile. Their `workloads` fields, result rows, gate/depth fields, and
QASM hashes are authoritative. The current script labels such future runs as
`profile: custom` to remove this ambiguity.

This result is specific to the recorded CPU/software environment and this
structured circuit. It demonstrates a reproducible bottleneck and optimization
there; it is not claimed as a universal thread rule for all tensor networks.
