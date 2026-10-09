# QiboTN CPU correctness and core-code revision (2026-10-09)

## Exact deep-pressure completion (17:04, 2026-10-09)

The predeclared 20q/9991-gate/depth-677 input has now completed all three
measured pairs: pinned-official mean 531.2482918999934 s, integrated mean
283.01838496668887 s, **1.877080501192603x**, not 7x. Both full states pass
an independent dense oracle. Strict audit verifies eight planned rows, two
additional restart warm-ups, ten source snapshots and original recovery hashes.
The original 2400-second safety stop (exit 1, not OOM) and explicit resumed
controller (exit 0) remain disclosed. See `PRESSURE_RESULTS_20261009.md`.

The user now requires exact 30q as a personal hard acceptance item. That is
**still incomplete**. `exact30_acceptance.py` fixes the existing 30q/9998-gate
input and zero-truncation complex128 settings; its 18 helper/small-state tests
pass, but local resource preflight fails. No new long test or paid resource
has been started. These tests do not establish a 30q runtime or accuracy result.

## Current production update (2026-10-09)

The bounded exact mixed CNOT/CY/CZ scheduler is now integrated in
`src/qibotn/backends/_mixed_schedule.py`. It is enabled by default only within
the existing uncapped NumPy routed-MPS path. Small (<14-qubit), pure-CNOT,
unbounded-search-size and truncated inputs retain the previous path. Set
`mixed_gate_schedule=False` for a paired compiler ablation. No GPU, alternate
simulator, state-dependent choice or precision reduction is introduced.

Production regression after integration: **322 passed**, four warnings.
The prior frozen five-module performance results below are historical, not
timings of this newly integrated six-module version. The fresh integrated
16q/1824-gate/depth-507 comparison completed successfully: official-reference
mean **38.811612566642 s**, integrated mean **7.971783200026645 s**, ratio
**4.868623693443177x**. Disabling only the new scheduler took
**10.818613033334259 s** (the integration improves that path by 1.3571133035x).
One warm-up and three measured repetitions used identical one-thread CPU,
untruncated complex128 settings and the original four-ZZ output. Independent
full-state fidelity is 0.9999999999999997; maximum phase-aligned state error
is 3.802471223049542e-16. This one input is not a universal or 30q claim.
The 12 raw rows, matched source snapshots and successful controller are in
`results/revision_cpu_20261009_integrated_mixed16`; strict current-source
audit is `integrated_mixed_audit_20261009.json`. Research/audit tests: 73 passed.

The official denominator is now pinned upstream plus exactly one attribute
typo repair. The added norm contraction in the older guarded reference is not
included. A pre-integration mixed-gate holdout measured **5.3293411285x**
against that denominator; two phase-polynomial research variants did not
establish a useful further gain and are not in the production dispatcher.

The instrumented 30q/9998-gate/depth-449 diagnostic in
`results/revision_cpu_20261009_large30_resource_probe` stopped at its explicit
900-MiB process-tree safety budget. This is NOT an OS out-of-memory event,
proof that every 30q circuit needs a large server, or completed acceptance.
There is no formal 30q timing or 7x claim from that partial diagnostic.
**All competition requirements are not yet complete.** Do not submit the old
blanket-completion interpretation or call the prior 7.27x a new official-
reference result. The original reports/packages remain untouched history.

## Latest frozen evidence (22:24, 2026-10-08)

The owned-buffer version completed its fixed 15 four-ZZ tasks: sum of
reference means divided by sum of optimized means is **7.273519193428899x**.
The 18q/2268-gate/depth-560 random nonlocal CNOT task is
**7.353715991351757x** (320.6639880667s vs 43.6057074333s). Production
regressions: **306 passed**; raw timing rows including warm-ups: 184.
Both paths use one BLAS thread and affinity **[2,4,6,8]** in this dataset.
Older sections below preserve earlier iterations and their different affinity.

The reference retains official native preparation and per-term contraction,
with a shared attribute-name repair, normalization, norm-loss guard and
diagnostics. Its extra norm contraction is timed. These ratios are against
that repaired/guarded reference, **not a completely untouched upstream timer**.
Upstream already prepares the circuit state once; the shared-environment
contribution avoids per-term full-MPS copies and repeated global contractions,
not four separate state simulations.

A predeclared fresh 18q CNOT/CY/CZ pressure supplement completed at 22:56:
315.4266958334s vs 80.2143624000s, **3.9322969901627496x**. Combining all
16 four-ZZ tasks yields **5.234866605055921x**, not 7x. The original fixed
15-task set remains separate and unchanged. Full evidence is in
`one_hour_delivery_audit.json`; no partial or old-version rows are merged. Representative
30q/~10000-gate acceptance is still **not complete**. See
`ONE_HOUR_ACCEPTANCE_20261008.md` and `REPRODUCE_OWNED_FINAL.md` for the
current scope and reproduction boundaries.

This revision supersedes the blanket "fully completed" interpretation of the
historical 30-qubit thread-only experiment, but preserves its raw evidence.

## Implemented code changes

- Shared left/right identity environments evaluate many Pauli terms without
  copying the full state for every term or globally contracting it again.
- Related term prefixes and the identity intervals between operators reuse
  intermediates; final operator insertions reuse right environments. Physical
  channels/conjugates are made contiguous once. Cache storage is byte-bounded
  to 32 MiB, not an unbounded number of potentially large matrices.
- Circuit connectivity is not assumed to be a chain. The *representation* is
  an open-boundary MPS; gate connectivity and contraction interval are distinct.
- Physical expectation values are normalized by the measured state norm.
- A configurable norm-loss guard rejects severe truncation. Its default 0.99
  is a screening threshold, **not a fidelity certificate**. Normalization never
  substitutes for independent accuracy checks.
- `expectation_method="reference"` retains the per-term contraction evaluator,
  with the same normalization and guard, as the matched-thread code baseline.
- The follow-up Hermitian transfer implementation computes one X/Y cross term
  and obtains its partner by adjoint symmetry, with direct two-channel I/Z
  transfers. This does not restrict gate connectivity or truncate any bonds.
  The added tests also cover random complex noncanonical environments.
- The latest implementation reuses left prefixes and right suffixes at a
  representation midpoint. It also fuses consecutive single-qubit matrices
  only on the same qubit, with the order `U_last @ ... @ U_first`. It never
  reorders across other sites or entangling gates. The original logical circuit
  and its QASM hash remain unchanged. Conversion/fusion cost is timed too.
- Fusion is used only by the NumPy MPS shared-expectation path, not the
  reference path or generic TN/JAX/Torch fallback. Gradient support for this
  numeric expectation API is not newly claimed.
- Exact NumPy two-site gate/SWAP kernels and canonicalization work directly
  on the official Quimb MPS tensors. Cached axis layouts avoid repeated
  generic Tensor bookkeeping. This is not a separate dense simulator.
- Persistent logical-to-physical routing avoids swapping back after each
  supported nonadjacent two-qubit gate. Logical order is restored before
  native special/multi-qubit gates and before returning the state. It does
  not branch on test seeds, target values, or measured quality.
- `mps_execution="routed_exact"` is the default for the shared expectation
  path; `"exact_kernels"` and `"native"` are explicit alternatives. The new
  route is enabled only for exact NumPy qubit/open-boundary MPS with
  `max_bond=None`, `cutoff=0`. Approximate/unsupported configurations retain
  native execution. Native Quimb already selects QR for some exact splits;
  QR itself is not claimed as a new algorithm contribution.
- The latest source also compiles contiguous CNOT-only blocks via exact
  Boolean linear maps and GF(2) elimination. It considers forward/reverse
  synthesis and retains the original unless static routing cost decreases
  without increasing gate count. Sparse forest blocks are left unchanged
  to bound compile overhead. No observable, seed, target value or topology
  name is consulted; any intervening non-CNOT gate ends a block.
- `cnot_optimization=False` disables this compiler. It is never applied to
  truncated runs or `mps_execution="native"`, and the reference performance
  path never compiles the circuit. Original Qibo queues/QASM remain untouched;
  compiled gate counts are explicitly recorded. Large gains on reducible
  CNOT networks are **structure-dependent**, not universal simulation gains.

## Reproduce (CPU only)

The delivery ZIP is a **patch/evidence bundle**, not a complete standalone
checkout or Python environment. Apply its `src/`, `tests/`, and `benchmarks/`
files over the official QiboTN source checkout at upstream commit
`7ebbd4d4a00f37b898c552922953591b78b0af7e` (or the existing local fork), preserving
the other official files. The upstream URL is recorded in the included
historical `provenance.json`. This avoids hiding framework code as a personal
contribution and keeps the original CPU entry point reproducible.

Use the existing Python 3.11 environment containing QiboTN/Quimb. Set
`PYTHONPATH=src` and run:

```powershell
python -m pytest tests/test_linear_cnot.py tests/test_exact_numpy.py tests/test_shared_expectation.py tests/test_quimb_backend.py -o addopts= -q
python benchmarks/asc_revision_cpu/evaluate.py --output results/revision_cpu_q12 --sizes 12 --seed 817 --thread-counts 1 --repeats 3 --affinity-cpu 2
python benchmarks/asc_revision_cpu/evaluate.py --output results/revision_cpu_q16 --sizes 16 --seed 1907 --thread-counts 1 --repeats 3 --affinity-cpu 2 --memory-mib 750 --case-seconds 240 --total-seconds 420
```

The suite tests chain, ring, long-range random matchings, and all-to-all gates,
with both the original four-ZZ workload and a mixed many-term observable.
Qibo's independent NumPy dense state is the correctness reference only; both
performance paths remain within the QiboTN Quimb NumPy CPU backend. The
optimized path includes the source modifications supplied in this revision;
framework capabilities are not claimed as personal contributions.
`max_bond=None`, `cutoff=0`, and complex128 preserve exactness at verified sizes.
Each timed call includes conversion, gate application, norm, and expectation.
Imports and independent correctness work are outside both timers. One warm-up
and three measured repeats are balanced in execution order.
Both paths are pinned to the same logical CPU (2) in the current dataset,
with one BLAS thread. This controls scheduling variability on this laptop;
it does not establish the best production thread count. Choose an available
logical CPU on another machine and record it rather than assuming CPU 2
is available or a particular core type.

Default limits are 1,000 MiB per worker, 120 seconds per topology/thread job,
and 600 seconds per suite. Failed/budget-limited jobs remain in the output; no
best-case selection or candidate output replacement is allowed. Outputs cannot
overwrite a prior evidence directory.

## Not yet established

- Multi-topology 30-qubit / ~10,000-gate accuracy and performance are **pending**.
- Local small/medium tests are not a substitute for that formal scale.
- No new universal speedup or final score is claimed.
- The old 7.026x result is a default-20-thread vs 1-thread configuration result,
  not this new core-code speedup. Historical fixed-bond approximation accuracy
  against an exact state was not established.
- Hardware-respecting production-scale accuracy requires a reference and an
  explicit memory/time plan. No new paid server has been started.

## Preserved iteration evidence

`results/revision_cpu_20261008_q12_v1` preserves the first correct but slower
implementation. `..._v2` preserves interval caching alone, still slower on these
inputs. `..._v3` adds contiguous channels and cached final insertions and shows
positive end-to-end improvements for all four recorded 12-qubit topologies.
These are engineering iterations, not output candidates selected for quality;
all numerical accuracy thresholds, circuits, and output definitions are fixed.
The independent seed-1907 16-qubit follow-up is recorded separately and must not
be promoted to a result before all requested cases finish.

The earlier Hermitian-only optimization has separate fixed-input evidence in
`results/revision_cpu_20261008_hermitian_q12` and `..._hermitian_q16`, with
five measured repeats. Do not mix measurements from different source hashes.
Its 16q all-to-all job was manually stopped after unstable timings under low
available memory; the cause is not confirmed. Raw rows and the failure exit
code are preserved. That suite is partial, not falsely marked complete.

`candidate_bidirectional.py`, `candidate_fused.py`, and their pilot capture the
engineering exploration. Their transformations have now been integrated into
`src/qibotn/backends/quimb.py`; the production backend does not import benchmark
scripts. The final source was retested independently, and its measured speed,
not the higher pilot speed, is reported.

Previous fused-only-source evidence: `results/revision_cpu_20261008_final_fused_q12` covers
all four 12q topologies (three measured repeats). `..._final_fused_q16_chain`
covers the unchanged 16q chain follow-up (five measured repeats). Those
source snapshots do not contain the newer exact kernels and routing.

`audit_latest.py` belongs to that previous fused-only version. It verifies its measured source snapshots, every timing and
correctness row, and the 46-test JUnit result. It reports the best comparison
as well as combined time ratios over the explicitly completed grid. A best
comparison is not a universal or formal 30q speedup. Fresh full-grid performance
and representative large-scale acceptance still need stable CPU resources.

## Previous exact-kernel/routing evidence

`results/revision_cpu_20261008_routed_exact_q12` and `..._q16` contain that
version's source snapshots and all four topologies for each size, 3 measured
repeats plus warm-up per workload. There are 128 timing/correctness rows over
8 size/topology jobs and 16 family comparisons. The full optimized state,
not only its observable, is checked against independent Qibo NumPy states.
77 tests are recorded in `validation_tests_exact_kernels.xml`.

The 16q all-to-all job was interrupted with its original exit status unknown.
The original 3 rows, log, metadata, and 3 completed topology entries are
preserved. `recover_interrupted_grid.py` checked that no original benchmark
process remained and that both source files, input and CPU affinity matched;
it continued only the 13 missing rows. Completed rows were not replaced.
Its own process completed successfully, recorded separately as
`recovery_suite_summary.json`; no clean original-parent exit is invented.
The continuation helper is deliberately single-use for this evidence set.
The recovery RSS monitor initially sampled the Windows Python launcher only,
so its stored 4.90625 MiB value is not a worker peak or proof of the memory
cap. Actual per-row worker RSS is available. This limitation and the original
monitor snapshot are preserved; future monitoring includes child processes.
Timing and independent correctness checks are unaffected.

`audit_routed_exact.py` verifies both production source hashes, all rows,
preserved rows across recovery, fixed input QASM hashes, independent full
states, and the JUnit result. It produces `ROUTED_EXACT_RESULTS.md`,
`routed_exact_audit.json`, and `routed_exact_comparisons.csv`, and packages
the evidence separately from earlier bundles. All weak/failed pilot evidence
is retained. Do not combine speedups from the pinned and unpinned datasets.

The current best local many-term result exceeds 7x; original four-ZZ and
fixed-grid aggregate results are reported separately. None of these replaces
representative 30q/~10000-gate/deep multi-topology formal acceptance.

## Previous linear-block compilation evidence

`..._linear_q12_tuning` tests the original 4 topologies with 1/2/4 BLAS
threads, both paths on the same logical CPU set [2,4,6,8]. `..._linear_q16`
retests all 4 held-out-size topologies with 1 thread and the same allowed CPU
set. Every measurement includes compilation and fresh MPS preparation.
All input QASM hashes remain those of the original fixed test grid.

`..._linear_holdout` adds unseen-seed random CNOT networks and circuits with
CNOT/CY/CZ mixed entanglers. These results, including weak gains, prevent the
ordered all-to-all example from being misrepresented as universal.
`audit_linear.py` audits the three current production source hashes, snapshots,
all raw rows, independent states, and 108-test JUnit. The 30q/10000-CNOT test
inside pytest checks only the exact Boolean operator transformation; it is
**not** a 30q simulation or performance acceptance result.

`..._nonlocal_pilot_v2` preserves an exact MPO-array prototype that was not
integrated: it regressed on long-range/all-to-all tests. Its first attempt
failed on fSim QASM metadata serialization before any performance rows;
that directory and reason are retained. `..._cnot_pilot` preserves the
development compiler measurements separately from integrated-source results.
Previous audits are version-specific and must not certify current source
with old timing data.

`summarize_and_package.py` belongs to the prior source snapshot and
intentionally refuses to certify changed code using that old timing dataset.

## CPU-resource and deeper-workload follow-up

`..._linear_q16_tuning24` adds 2/4-thread measurements on the same input and
CPU set as the 1-thread data. Seven of eight jobs completed. The ordered
all-to-all 4-thread reference was much slower than its 1-thread counterpart
and reached the 240-second per-job budget; its partial rows, full-state check,
log and exit code remain preserved. It is not a completed 4-thread result or
a reason to inflate the performance denominator. Best-observed thread means
are tuning evidence, not independent post-selection verification.

`pressure_cpu.py` keeps exact complex128/no bond cap/no cutoff and tests an
18-qubit 1,272-gate deeper chain plus a 16-qubit 1,216-gate random CNOT circuit
of depth 324. These are actual MPS simulations with independent dense complete
states for correctness. The original four-ZZ task, preparation, compiler and
all semantic gate action are timed on both QiboTN paths. Process-tree RSS and
elapsed budgets are sampled; low system free RAM also stops the test. These
checks are more demanding than the earlier shallow grid, but still do not
replace representative 30-qubit performance acceptance.

`probe_qr.py` profiles that fixed deep chain and compares a non-integrated
private-array SciPy QR candidate. Only the private exact-kernel NumPy facade
is temporarily replaced; global NumPy/reference behavior is not patched.
The production files remain unchanged during this research. Both weak gains
and regressions must be kept, and a pilot is not a delivered speedup.

`audit_cpu_resources.py` recomputes thread choices, pressure measurements and
hardware observations, and writes `CPU_RESOURCE_AUDIT.json` and
`CPU_RESOURCE_RESULTS.md`. The accessible AutoDL browser required login and
two historical Featurize project SSH connections closed; these observations
do not establish that a cloud CPU instance or paid runtime is available.
No new instance, rental or power/billing setting was created or changed.

## Integrated section synthesis and QR follow-up

The current core adds a bounded-width integer-row adaptation of the published
[Patel–Markov–Hayes section-pattern algorithm](https://arxiv.org/abs/quant-ph/0302002).
The algorithm is attributed, not claimed as original. Widths 1..4 and both wire
orientations compete with the previous synthesis, retaining lower static cost
without increasing the original CNOT count. Every candidate is checked against
the entire exact Boolean map, not output values. `rank(A-I)` is used only as a
rigorous CNOT-count lower bound to avoid redundant search when a nearest-neighbor
incumbent already reaches that bound. Sparse forest blocks are still skipped.

Larger private exact-array QR factorizations now use economy SciPy QR with an
owned Fortran workspace. Small matrices retain NumPy. No input tensor view is
overwritten, no singular values are discarded, and dtype/precision are retained.
This is a workspace/dispatch optimization, not the invention of QR or a new
SVD-to-QR algorithm. SciPy is now an explicit dependency (already required by
the installed Quimb environment).

`..._gauge_pilot` preserves a rejected attempt to omit global canonical-center
moves. Although 23 small-state checks passed, the deep workload regressed to
about 30-34 seconds per call from about 5 seconds. Its workers reached the
360/500-second limits; it was **not integrated**. The wrapper's exit 0 does not
mean its workloads passed. A later scoped stop check killed no process because
the budgeted workers had already ended.

`..._section_pilot` preserves the old-core research comparison. The current
implementation is independently measured under `..._adaptive`, including the
unchanged 12/16q fixed inputs, the two deeper pressure inputs, and new-seed
random-CNOT/mixed-entangler circuits. Every timing path includes compilation,
fresh state preparation, normalized expectation and the same original output.
Do not reuse candidate timings for the integrated implementation.

The current audit is `audit_adaptive.py`: 145 production tests and 176 raw rows,
with fixed-grid and expanded-14-case original-four-ZZ totals kept separate from
mixed-observable tasks. Earlier auditors are source-version-specific and are
not current-source certificates. A fixed-grid speedup above 7x is not a universal
7x or representative 30q acceptance result.

## Full-row exact isometric factorization follow-up

The private MPS kernels require an isometric factor Q and reconstruction
Q@R=A, not triangular R. `_left_isometric_factor` now uses Q=I and R=A when
rows<=columns, retaining the same economy bond dimension and dtype. Tall
matrices keep exact QR. This is a shape-based algebraic transformation for
arbitrary inputs, not a state/observable/circuit-name special case. Canonical
center moves remain in place; public split absorption direction is respected.

`..._isometry_pilot` measured this simple helper against the previous source.
Early complex-candidate pilots only patched an instance preparation method,
while the timing API dispatched to a module-level function. Their full-state
checks exercised the intended candidates, but timed gains did not: the
`dispatch_disqualification.json` files explicitly disqualify those performance
attributions. The unitary v1 pilot was stopped and all partial rows retained.
Corrected `..._unitary_isometry_pilot_v2` patches module dispatch in its isolated
worker and asserts the actual engine in each row. Its complex candidate is
not integrated, even though it passed correctness and had a deep-chain gain.

Fresh integrated-source evidence is `..._isometric`, with new holdout seeds
20261010/20261011. Use `audit_isometric.py`, not old source-bound auditors.
No historical/pilot timings are merged into the new aggregate. No paid CPU
instance was started, and representative 30q acceptance remains pending.

The additional hardware-bounded random workload uses 18 qubits, 12 layers,
seed 20261018, and 2,268 original gates. It is a complete exact simulation,
not a Boolean-only compiler test or a structured nearest-neighbor substitute.
Its outputs are separate from the 14-case collection; both its own ratio and
the expanded 15-case sum-of-times ratio must be shown when completed.

```powershell
python benchmarks/asc_revision_cpu/pressure_cpu.py --output results/revision_cpu_20261008_random18_isometric --only-kind random_connectivity --qubits 18 --layers 12 --seed 20261018 --threads 1 --repeats 3 --memory-mib 1000 --case-seconds 2700 --total-seconds 2700
python benchmarks/asc_revision_cpu/audit_isometric.py
```

The first command refuses existing output paths. It has one full warm-up and
three measured repeats per path, and stops on low available RAM or resource
limits. A budget stop is evidence of an incomplete job, never a speedup.
The 45-minute bound is a limit, not a completion-time guarantee. This is a
same-one-thread code comparison, not a certificate of optimal CPU threading.

The first 18-qubit attempt was found interrupted at 19:22 local time: no
worker/controller or live terminal session remained, no runner summary was
written, and only four of eight required rows were recorded. The actual cause
is unknown. `interruption_audit.json` records these observations without
inventing a timeout/OOM diagnosis or exit code. Its full-state check remains
valid evidence, but the partial timing is excluded from completed speedup
totals. A follow-up must use a new output directory, not overwrite or silently
resume the interrupted experiment.

## Integrated bounded-lookahead routing follow-up

The current exact NumPy MPS path additionally chooses between two equivalent
SWAP chains using at most six upcoming two-qubit gate locations and the current
canonical-center distance. It never examines amplitudes, observables, expected
answers, input names or performance ground truth. Logical order is restored
before output. Unsupported native special gates retain their fallback, and
research circuit subclasses retain their existing absorption/routing policy.
This is a heuristic, not a guarantee that every circuit is faster.

The correctly dispatched `..._lookahead_routing_pilot` contains one warm-up and
three measurements for the unchanged deep-chain, random-connectivity and
mixed-entangler tasks. It passed independent complete-state checks and actual
engine assertions. It is separately snapshotted research evidence, not part of
the new production aggregate. The integrated source passed 197 regression
tests. Its fresh full collection uses new unseen seeds 20261020/20261021.

```powershell
python benchmarks/asc_revision_cpu/run_adaptive_followup.py --output results/revision_cpu_20261008_lookahead --holdout-seeds 20261020 20261021
python benchmarks/asc_revision_cpu/pressure_cpu.py --output results/revision_cpu_20261008_random18_lookahead --only-kind random_connectivity --qubits 18 --layers 12 --seed 20261018 --threads 1 --repeats 3 --memory-mib 1000 --case-seconds 2700 --total-seconds 2700
python benchmarks/asc_revision_cpu/audit_lookahead.py
```

Run the two measurement commands sequentially, never concurrently. Do not
overwrite existing output or silently resume interrupted rows. The second
command is an optional larger follow-up and is not complete merely because
this command is documented. The auditor includes it only after all eight rows,
correctness, new-source hashes and the successful runner summary pass.

Timing JSON now uses atomic replacement; runner metadata and UTF-8 child logs
are persisted, and budget/failure summaries return a nonzero shell exit code.
An externally killed controller can still leave an incomplete experiment;
its partial results must never be promoted to a completed speedup.
