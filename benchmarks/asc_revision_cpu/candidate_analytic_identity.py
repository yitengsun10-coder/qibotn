"""Research-only exact identity guards without dense value snapshots.

Only generated left/right identity isometries are tagged. Every use verifies
array identity AND all current values, in bounded views, against the analytic
identity pattern. In-place edits invalidate a tag; tags hold weak references
and never keep replaced tensors alive. No numerical tolerance, rank estimate,
gate omission, dtype conversion or external read-only assumption is used.

This removes a guard snapshot, not the explicit MPS tensors themselves. It is
not an original30 completion, peak-RSS measurement or speedup claim.
"""
from types import MethodType
import weakref

import numpy as np
import quimb.tensor as qtn
from qibotn.backends import _exact_numpy as exact, quimb as production
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS, clone
from benchmarks.asc_revision_cpu.candidate_short_cnot_rank2 import compile_linear, compile_mixed


def exact_identity_isometry(array, side, chunk_bytes=1024*1024, observe=None):
    """Exactly check A[l,r,p]=delta(r,2*l+p) or delta(l,2*r+p).

    The observed bound is on each scanned view/value selection, not whole
    process RSS. Advanced diagonal index arrays are bounded by the same chunk
    policy. Nonzero counts plus exact diagonal ones check EVERY entry.
    """
    if not isinstance(array,np.ndarray) or array.ndim!=3 or array.shape[2]!=2:
        return False
    if array.dtype.kind not in 'buifc' or side not in {'left','right'}:
        return False
    left,right,_=array.shape
    if not left or not right or (right!=2*left if side=='left' else left!=2*right):
        return False
    if not isinstance(chunk_bytes,int) or chunk_bytes < 2*array.dtype.itemsize:
        raise ValueError('Guard chunk must hold at least one physical qubit row')
    record = observe if observe is not None else lambda size: None
    columns = min(right,max(1,chunk_bytes//(2*array.dtype.itemsize)))
    rows = max(1,chunk_bytes//(2*columns*array.dtype.itemsize))
    expected = right if side=='left' else left
    count = 0
    for start in range(0,left,rows):
        for column in range(0,right,columns):
            block = array[start:start+rows,column:column+columns,:]
            record(block.nbytes)
            count += int(np.count_nonzero(block))
            if count > expected:
                return False
    if count != expected:
        return False
    width = left if side=='left' else right
    diagonal_rows = max(1,chunk_bytes//(4*np.dtype(np.intp).itemsize+array.dtype.itemsize))
    for start in range(0,width,diagonal_rows):
        positions = np.arange(start,min(width,start+diagonal_rows))
        for physical in (0,1):
            diagonal = (array[positions,2*positions+physical,physical] if side=='left'
                        else array[2*positions+physical,positions,physical])
            record(diagonal.nbytes)
            if not np.all(diagonal==1):
                return False
    return True


class AnalyticIdentityRank2MPS(ControlledRank2MPS):
    guard_chunk_bytes = 1024*1024

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identity_guard_stats = {
            'created_tags':0,'checks':0,'accepted_checks':0,'rejected_checks':0,
            'max_scan_chunk_bytes':0,'max_tagged_array_bytes':0,
            'dense_guard_snapshot_allocations':0}

    def _record_guard_chunk(self,size):
        stats=self.identity_guard_stats
        stats['max_scan_chunk_bytes']=max(stats['max_scan_chunk_bytes'],size)

    def _known_identity(self,site):
        item=self._identity_gauges.get(site)
        if item is None:
            return None
        reference,side=item
        stats=self.identity_guard_stats
        stats['checks']+=1
        if (reference() is self[site].data and
                exact_identity_isometry(self._layout(site)[1],side,self.guard_chunk_bytes,
                                        self._record_guard_chunk)):
            stats['accepted_checks']+=1
            return side
        self._identity_gauges.pop(site,None)
        stats['rejected_checks']+=1
        return None

    def _put(self,site,layout,array,identity=None):
        # Deliberately do not call the snapshot-creating parent implementation.
        if identity is not None and identity not in {'left','right'}:
            raise ValueError('Unknown identity side')
        self._store(layout,array)
        self._identity_gauges.pop(site,None)
        if identity is not None:
            self._identity_gauges[site]=(weakref.ref(self[site].data),identity)
            self.identity_guard_stats['created_tags']+=1
            active = {id(self[k].data):self[k].data.nbytes
                      for k,(reference,_) in self._identity_gauges.items()
                      if reference() is self[k].data}
            self.identity_guard_stats['max_tagged_array_bytes']=max(
                self.identity_guard_stats['max_tagged_array_bytes'],sum(active.values()))


class AnalyticIdentityCircuit(qtn.CircuitMPS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._psi=AnalyticIdentityRank2MPS(self._psi)


route=clone(exact.prepare_routed_mps,{'ExactKernelCircuitMPS':AnalyticIdentityCircuit},
            defaults=(AnalyticIdentityCircuit,))
preparation=clone(production._prepare_optimized_mps,{'prepare_routed_mps':route})
short_preparation=clone(production._prepare_optimized_mps,{
    'prepare_routed_mps':route,'optimize_and_schedule':compile_linear,
    'optimize_mixed_schedule':compile_mixed})


def prepare(backend,circuit,short_schedule=False):
    state,details=(short_preparation if short_schedule else preparation)(backend,circuit)
    return state,{**details,'research_candidate':'analytic_identity_controlled_rank2',
        'production_integrated':False,'short_schedule_enabled':short_schedule,
        'identity_guard':'weak array identity plus exact chunked analytic value check',
        'guard_snapshot_policy':'no dense guard value snapshot; explicit MPS remains',
        'counter_scope':'observe live state; Circuit.psi copies can reset private counters'}


def make_backend(short_schedule=False):
    backend=production.QuimbNumpyBackend()
    def selected(backend,circuit):
        return prepare(backend,circuit,short_schedule=short_schedule)
    entry=clone(production.exp_value_observable_symbolic,{'_prepare_optimized_mps':selected})
    backend.exp_value_observable_symbolic=MethodType(entry,backend)
    return backend
