# ASC QiboTN large-workload CPU result

> **2026-10-09 current exact CPU evidence:** The same-six-module 20q/9991-gate/
> depth-677 pressure comparison has completed and passed strict recovery/source
> audit: 531.2482918999934 s official mean vs 283.01838496668887 s optimized,
> **1.877080501192603x**. Both complete states pass an independent dense oracle.
> The original time-budget stop (exit 1) and successful explicit continuation
> (exit 0) are both preserved. The separate mixed-gate 16q result is 4.8686236934x.
> Neither is a universal or 30q claim. Exact 30q remains the user's incomplete
> hard acceptance item; no valid current 7x result exists. See
> `../asc_revision_cpu/PRESSURE_RESULTS_20261009.md`.

> **2026-10-08 audit correction:** The historical measurements below support
> only a particular CPU configuration and a single structured circuit. They
> do not establish universal performance or full-state accuracy. The archived
> conservative MPS had norm squared 4.2367233491907075e-08; agreement of four
> unnormalized ZZ outputs is insufficient as independent correctness evidence.
> Consequently the prior "fully completed" acceptance interpretation is
> withdrawn. Raw timings are preserved, not relabeled as fabricated, and the
> 7.026x result is not a core-algorithm speedup. See `../asc_revision_cpu/` for
> normalized evaluation, norm-loss rejection, shared-environment code changes,
> and new exact-reference multi-topology validation. Formal-scale acceptance
> remains pending.

## Historical result — withdrawn as acceptance evidence

The archived measured workload contains **30 qubits, 9,998 gates, and circuit
depth 449**. It runs through QiboTN's Quimb NumPy CPU backend; no GPU,
cuQuantum, qibojit, or external state-vector timing path is used.

The historical configuration experiment sets the CPU math-library thread pool to
one thread instead of the machine's measured **20-thread OpenBLAS default**.
With all numerical settings and the circuit fixed, steady-state runtime fell
from **404.0428 s** to **57.5065 s**, a **7.0260x speedup**. Peak process RSS
fell from **292.51 MiB** to **267.36 MiB**. The absolute output difference was
**5.79e-16**, below the `1e-12` tolerance.

These timings are retained as a thread-overhead diagnosis only. A tiny change
in a near-zero, unnormalized expectation does not establish simulation
accuracy: the archived state norm squared was approximately 4.24e-08. The
7.0260x ratio must not be submitted as an accepted accurate 30q optimization.
For the current same-one-thread source comparison, see
`../asc_revision_cpu/INTEGRATED_UPDATE_20261009.md`: 4.868623693443177x on one
16q mixed-gate input, with independent exact full-state validation. The
20q/9991-gate/depth-677 pressure follow-up is separate and now complete at
1.877080501192603x after timing and recovery audit, not merely full-state checks.

## Archived configuration comparison — not formal acceptance

| Item | 20-thread default baseline | 1-thread optimized |
|---|---:|---:|
| Qubits / gates / depth | 30 / 9,998 / 449 | 30 / 9,998 / 449 |
| Repetitions | 3 | 3 |
| Steady mean (repeats 2-3) | 404.0428 s | 57.5065 s |
| Steady standard deviation | 1.4983 s | 0.4357 s |
| Peak process RSS | 292.51 MiB | 267.36 MiB |
| ZZ expectation | 3.6185261e-11 | 3.6184682e-11 |
| Absolute output difference | 5.79e-16 | reference |
| `svd_cutoff` / max bond | 1e-12 / 128 | 1e-12 / 128 |
| Contraction optimizer | auto-hq | auto-hq |

An independent fresh-process probe with the thread environment variables unset
reported OpenBLAS `num_threads=20`. Both runs used QASM SHA-256
`1ea5c95ebdcc651a1291dadfb7a193009377732bba8d03a01ac37ef7b82b37d4`.
The identical hash checks that both configurations received the same circuit.
The first execution is reported separately as cold time; speedup uses the mean
of repetitions 2 and 3 so one-time initialization does not dominate. The
earlier 16-thread sweep remains scaling evidence, but is not the formal
default setting in this historical experiment, not today's fair code baseline.

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
performance gains. The old thread comparison changed no approximation between
its two paths, but that alone did not make their common truncated state accurate.

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
  and summaries for the 1-thread and 16-thread scaling runs.
- `evidence/formal_default20_q30/`: formal default-thread baseline CSV,
  environment capture, summary, and exact reproduction notes.
- `evidence/fidelity_scan_q30.json`: state-level cutoff accuracy scan.
- `evidence/provenance.json`: software versions, default OpenBLAS probe, and
  official upstream source commit.
- `verify_evidence.py`: historical offline checks for workload size, circuit
  identity, repetitions, scalar tolerance and arithmetic. Its PASS is not an
  independent full-state accuracy certificate and cannot restore acceptance.

The captured environment files show `profile: quick` because the launcher used
an explicit `--workload 30:112`, which overrides the parser's historical
default profile. Their `workloads` fields, result rows, gate/depth fields, and
QASM hashes are authoritative. The current script labels such future runs as
`profile: custom` to remove this ambiguity.

This archive demonstrates a thread-overhead phenomenon in its recorded
CPU/software environment. It is neither universal performance evidence nor a
completed accurate large-workload result. The raw files remain unchanged.
