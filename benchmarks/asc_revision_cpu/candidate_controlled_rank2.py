"""Exact controlled-gate operator-Schmidt factorization inside the same MPS.

For controlled-U, P0 x I + P1 x U has operator bond at most two. Construct
that factorization explicitly when 2*old_bond is strictly below a generic
merged split dimension, then restore the requested canonical side using QR.
No SVD cutoff, numerical-rank guess, input simplification or output selection.
"""
import types
import numpy as np
import quimb.tensor as qtn
from qibotn.backends import _exact_numpy as exact, _structured_numpy as structured
from qibotn.backends import quimb as production


class ControlledRank2MPS(structured.StructuredExactKernelMPS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.controlled_rank2_splits=0

    def _two_site(self,G,i,j,absorb,reverse=False):
        if G is None:
            return super()._two_site(G,i,j,absorb,reverse=reverse)
        gate=np.asarray(G).reshape(4,4)
        if not (np.array_equal(gate[:2,:2],np.eye(2)) and
                np.count_nonzero(gate[:2,2:])==0 and np.count_nonzero(gate[2:,:2])==0):
            return super()._two_site(G,i,j,absorb,reverse=reverse)
        lhs,rhs=self._layout(i),self._layout(j)
        a,b=lhs[1],rhs[1]
        left,old,right=a.shape[0],a.shape[1],b.shape[1]
        if old!=b.shape[0] or 2*old>=min(2*left,2*right):
            return super()._two_site(G,i,j,absorb,reverse=reverse)
        u=gate[2:,2:]
        anew=np.zeros((left,2*old,2),dtype=np.result_type(a,b,gate))
        bnew=np.zeros((2*old,right,2),dtype=anew.dtype)
        if not reverse:
            anew[:,:old,0]=a[:,:,0]
            anew[:,old:,1]=a[:,:,1]
            bnew[:old]=b
            bnew[old:]=np.einsum('pq,lrq->lrp',u,b,optimize=True)
        else:
            anew[:,:old]=a
            anew[:,old:]=np.einsum('pq,lrq->lrp',u,a,optimize=True)
            bnew[:old,:,0]=b[:,:,0]
            bnew[old:,:,1]=b[:,:,1]
        if absorb=='right':
            q,r=exact._left_isometric_factor(anew.transpose(0,2,1).reshape(2*left,2*old))
            x=q.reshape(left,2,q.shape[1]).transpose(0,2,1)
            y=(r@bnew.reshape(2*old,-1)).reshape(r.shape[0],right,2)
        else:
            q,r=exact._left_isometric_factor(bnew.reshape(2*old,2*right).T)
            y=q.T.reshape(q.shape[1],right,2)
            updated=anew.transpose(0,2,1).reshape(2*left,2*old)@r.T
            x=updated.reshape(left,2,r.shape[0]).transpose(0,2,1)
        self._put(i,lhs,x)
        self._put(j,rhs,y)
        self.controlled_rank2_splits+=1


class ControlledRank2Circuit(qtn.CircuitMPS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._psi=ControlledRank2MPS(self._psi)


def clone(original,updates,defaults=None):
    return types.FunctionType(original.__code__,{**original.__globals__,**updates},original.__name__,
        original.__defaults__ if defaults is None else defaults,original.__closure__)


route=clone(exact.prepare_routed_mps,{'ExactKernelCircuitMPS':ControlledRank2Circuit},
            defaults=(ControlledRank2Circuit,))
preparation=clone(production._prepare_optimized_mps,{'prepare_routed_mps':route})


def prepare(backend,circuit):
    state,details=preparation(backend,circuit)
    return state,{**details,'research_candidate':'controlled_rank2',
                  'controlled_rank2_splits':getattr(state,'controlled_rank2_splits',0)}


def make_backend():
    backend=production.QuimbNumpyBackend()
    entry=clone(production.exp_value_observable_symbolic,{'_prepare_optimized_mps':prepare})
    backend.exp_value_observable_symbolic=types.MethodType(entry,backend)
    return backend
