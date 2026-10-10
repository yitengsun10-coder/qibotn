"""Isolated identity-guard allocation check, never a quantum speed benchmark.

Uses one deterministic identity tensor in a synthetic redundant-bond MPS.
No circuit is executed. tracemalloc reports traced Python/NumPy allocations,
NOT process RSS, final original30 memory or an end-to-end speed ratio.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tracemalloc

import numpy as np
import quimb.tensor as qtn
from qibotn.backends._structured_numpy import StructuredExactKernelMPS
from benchmarks.asc_revision_cpu.candidate_analytic_identity import (
    AnalyticIdentityRank2MPS,exact_identity_isometry)

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCES=(Path(__file__),HERE/'candidate_analytic_identity.py',
    HERE/'candidate_controlled_rank2.py',HERE/'candidate_short_cnot_rank2.py',
    ROOT/'src/qibotn/backends/_structured_numpy.py',ROOT/'src/qibotn/backends/_exact_numpy.py')


def check(width=512,chunk_bytes=1024*1024):
    if not isinstance(width,int) or not 1<=width<=1024:
        raise ValueError('Isolated check is bounded to width 1..1024')
    if not isinstance(chunk_bytes,int) or not 32<=chunk_bytes<=1024*1024:
        raise ValueError('Guard chunk is bounded to 32 B..1 MiB')
    if tracemalloc.is_tracing():
        raise ValueError('Use an isolated process, no existing tracer')
    tracemalloc.start()
    marker=np.empty(131072,dtype=np.uint8)
    registered=tracemalloc.get_traced_memory()[0]>=marker.nbytes
    del marker
    tracemalloc.stop()
    if not registered:
        raise RuntimeError('NumPy allocation tracing unavailable')
    matrix=np.eye(2*width,dtype=np.complex128)
    array=matrix.reshape(width,2,2*width).transpose(0,2,1)
    digest=hashlib.sha256(memoryview(matrix).cast('B')).hexdigest()
    rows=[]
    for label,cls in (('legacy_dense_snapshot',StructuredExactKernelMPS),
                      ('analytic_weak_chunked',AnalyticIdentityRank2MPS)):
        # Bond-consistent synthetic three-site representation. This is not
        # normalized and is NOT a correctness/performance workload.
        state=cls(qtn.MatrixProductState([
            np.zeros((width,2),dtype=complex),array,
            np.zeros((2*width,2),dtype=complex)],shape='lrp'))
        if label=='analytic_weak_chunked':state.guard_chunk_bytes=chunk_bytes
        layout=state._layout(1)
        gc.collect()
        tracemalloc.start()
        try:
            state._put(1,layout,array,'left')
            accepted=state._known_identity(1)=='left'
            current,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if not accepted:raise ValueError('Exact identity was not recognized')
        entry=state._identity_gauges[1]
        snapshot_bytes=entry[1].nbytes if isinstance(entry[1],np.ndarray) else 0
        rows.append({'method':label,'exact_guard_accepted':accepted,
            'attached_dense_guard_snapshot_bytes':snapshot_bytes,
            'current_traced_extra_bytes':current,'peak_traced_extra_bytes':peak,
            'live_guard_stats':getattr(state,'identity_guard_stats',None)})
        del state,layout,entry
    if rows[0]['attached_dense_guard_snapshot_bytes']!=array.nbytes or rows[1]['attached_dense_guard_snapshot_bytes']!=0:
        raise ValueError('Snapshot allocation contract differs')
    if rows[1]['live_guard_stats']['max_scan_chunk_bytes']>chunk_bytes:
        raise ValueError('Guard scan exceeds declared view bound')
    return {'status':'isolated_guard_allocation_check_only',
        'recorded_at':datetime.now().astimezone().isoformat(),
        'numpy_version':np.__version__,'synthetic_tensor_shape':list(array.shape),
        'quimb_version':importlib.metadata.version('quimb'),
        'tensor_bytes':array.nbytes,'identity_matrix_sha256':digest,
        'guard_chunk_bytes':chunk_bytes,'numpy_tracing_registration_checked':registered,
        'methods':rows,'quantum_circuit_executed':False,
        'observed_process_rss':False,'formal_30q_acceptance_complete':False,
        'publishable_speedup':None,
        'scope':'isolated guard allocations, not whole-circuit peak memory or speed',
        'source_hashes':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCES}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--chunk-bytes',type=int,default=1024*1024)
    args=parser.parse_args()
    report=check(args.width,args.chunk_bytes)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
