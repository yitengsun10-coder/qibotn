# Formal default-thread run notes

The first launch accidentally imported the older site-packages copy of
QiboTN and stopped with the known singular/plural contraction-optimizer
attribute error. It produced no timing row and is not part of the comparison.

The formal run explicitly selected this checkout through `PYTHONPATH=src` and
used the following command:

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='20'
$env:OPENBLAS_NUM_THREADS='20'
$env:MKL_NUM_THREADS='20'
$env:NUMEXPR_NUM_THREADS='20'
$env:QUIMB_NUM_PROCS='20'
python benchmarks/asc_large_cpu/benchmark.py `
  --output results/formal_default20_q30 `
  --workload 30:112 --variant baseline --repeats 3 `
  --threads 20 --optimizer auto-hq
```

The 20-thread value was not chosen after timing. Before the formal run, a
fresh Python process with the five thread variables unset reported OpenBLAS
`num_threads=20`; see `../provenance.json`.
