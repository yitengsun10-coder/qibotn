"""Research-only rank2 MPS plus size-aware BLAS limits, never a speed claim.

Use the same CPU/MPS/gates/complex128 arithmetic. Small QR work does not need
the same thread team as a large factorization. Restore the caller's BLAS
limits after every private operation. No answer/circuit/topology inspection.
"""
from functools import lru_cache
import types
import quimb.tensor as qtn
from threadpoolctl import ThreadpoolController
from qibotn.backends import _exact_numpy as exact
from qibotn.backends import quimb as production
from benchmarks.asc_revision_cpu.candidate_controlled_rank2 import ControlledRank2MPS,clone


def thread_policy(rows,columns):
    width=min(rows,columns)
    if width<512:return 1
    if width<1024:return 4
    if width<2048:return 8
    return 16


@lru_cache(maxsize=1)
def controller():
    return ThreadpoolController()


class AdaptiveControlledMPS(ControlledRank2MPS):
    def _two_site(self,G,i,j,absorb,reverse=False):
        a,b=self._layout(i)[1],self._layout(j)[1]
        # The rank2 path factors a matrix with 2*old columns. For unsupported
        # gates use the generic merged QR width. This changes only scheduling.
        import numpy as np
        controlled=False
        if G is not None:
            gate=np.asarray(G).reshape(4,4)
            controlled=(np.array_equal(gate[:2,:2],np.eye(2)) and
                        np.count_nonzero(gate[:2,2:])==0 and np.count_nonzero(gate[2:,:2])==0 and
                        2*a.shape[1]<min(2*a.shape[0],2*b.shape[1]))
        rows,columns=(2*(a.shape[0] if absorb=='right' else b.shape[1]),2*a.shape[1]) if controlled else (2*a.shape[0],2*b.shape[1])
        with controller().limit(limits=thread_policy(rows,columns),user_api='blas'):
            return super()._two_site(G,i,j,absorb,reverse=reverse)

    def left_canonize_site(self,i,bra=None,create_bond=False):
        a=self._layout(i)[1]
        with controller().limit(limits=thread_policy(2*a.shape[0],a.shape[1]),user_api='blas'):
            return super().left_canonize_site(i,bra=bra,create_bond=create_bond)

    def right_canonize_site(self,i,bra=None,create_bond=False):
        a=self._layout(i)[1]
        with controller().limit(limits=thread_policy(2*a.shape[1],a.shape[0]),user_api='blas'):
            return super().right_canonize_site(i,bra=bra,create_bond=create_bond)


class AdaptiveControlledCircuit(qtn.CircuitMPS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._psi=AdaptiveControlledMPS(self._psi)


route=clone(exact.prepare_routed_mps,{'ExactKernelCircuitMPS':AdaptiveControlledCircuit},
            defaults=(AdaptiveControlledCircuit,))
preparation=clone(production._prepare_optimized_mps,{'prepare_routed_mps':route})


def prepare(backend,circuit):
    state,details=preparation(backend,circuit)
    return state,{**details,'research_candidate':'controlled_rank2_adaptive_threads',
                  'controlled_rank2_splits':getattr(state,'controlled_rank2_splits',0),
                  'thread_policy':'min QR width <512:1, <1024:4, <2048:8, else16; max16',
                  'performance_validated':False}


def make_backend():
    backend=production.QuimbNumpyBackend()
    entry=clone(production.exp_value_observable_symbolic,{'_prepare_optimized_mps':prepare})
    backend.exp_value_observable_symbolic=types.MethodType(entry,backend)
    return backend
