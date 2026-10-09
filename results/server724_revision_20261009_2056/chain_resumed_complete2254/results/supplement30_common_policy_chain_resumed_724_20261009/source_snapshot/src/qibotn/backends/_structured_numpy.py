"""Use exact generated identity-isometry tensors as index maps, not GEMMs.

Tags are guarded by both array identity and an exact value snapshot. In-place
external edits invalidate the optimization, rather than trusting stale data.
The Quimb MPS tensors remain explicit arrays, with unchanged physical order.
Existing identity isometries may be reused only after the same value guard.
Gate buffers avoid a second copy only when they cannot alias either input.
"""
import numpy as np
from qibotn.backends import _exact_numpy as exact


class StructuredExactKernelMPS(exact.ExactKernelMPS):
    minimum_merged_bytes = 256 * 1024

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._identity_gauges={}

    def _known_identity(self,site):
        item=self._identity_gauges.get(site)
        if item is not None:
            data,snapshot,side=item
            if self[site].data is data and np.array_equal(data,snapshot):
                return side
            self._identity_gauges.pop(site,None)
        return None

    def _put(self,site,layout,array,identity=None):
        self._store(layout,array)
        self._identity_gauges.pop(site,None)
        if identity is not None:
            data=self[site].data
            self._identity_gauges[site]=(data,data.copy(),identity)

    def _two_site(self, G, i, j, absorb, reverse=False):
        lhs, rhs = self._layout(i), self._layout(j)
        a, b = lhs[1], rhs[1]
        left, right = a.shape[0], b.shape[1]
        if left * right * 4 * a.dtype.itemsize < self.minimum_merged_bytes:
            self._identity_gauges.pop(i, None)
            self._identity_gauges.pop(j, None)
            return exact.ExactKernelMPS._two_site(self, G, i, j, absorb, reverse=reverse)
        left_identity = self._known_identity(i) == "left"
        right_identity = self._known_identity(j) == "right"
        if left_identity:
            theta = b.reshape(left, 2, right, 2)
        elif right_identity:
            theta = a.reshape(left, right, 2, 2).transpose(0, 3, 1, 2)
        else:
            theta = np.tensordot(a, b, axes=(1, 0))
        if G is None:
            matrix = theta.transpose(0, 3, 2, 1).reshape(left * 2, right * 2)
        else:
            gate = np.asarray(G).reshape(2, 2, 2, 2)
            if reverse:
                gate = gate.transpose(1, 0, 3, 2)
            gate = gate.reshape(4, 4)
            nonzero = gate != 0
            if np.all(nonzero.sum(axis=1) == 1) and np.all(nonzero.sum(axis=0) == 1):
                matrix = np.empty((left * 2, right * 2), dtype=np.result_type(theta, gate))
                for output, source in enumerate(np.argmax(nonzero, axis=1)):
                    target = matrix[output // 2::2, output % 2::2]
                    value = theta[:, source // 2, :, source % 2]
                    np.multiply(value, gate[output, source], out=target)
            else:
                physical = theta.transpose(0, 2, 1, 3).reshape(left * right, 4)
                matrix = (physical @ gate.T).reshape(left, right, 2, 2)
                matrix = matrix.transpose(0, 2, 1, 3).reshape(left * 2, right * 2)
        # A routing reshape can still be a view of an old input tensor. Such
        # cases retain the private copy; only independent new buffers can move
        # directly into the result. Mutable external references stay isolated.
        independent = (not np.may_share_memory(matrix, a)
                       and not np.may_share_memory(matrix, b))
        if absorb == "right" and matrix.shape[0] <= matrix.shape[1]:
            x = (a.transpose(0, 2, 1).reshape(left * 2, left * 2)
                 if left_identity else np.eye(left * 2, dtype=matrix.dtype))
            y = matrix if independent else matrix.copy()
            reuse_left, reuse_right = left_identity, False
            li, ri = "left", None
        elif absorb == "left" and matrix.shape[1] <= matrix.shape[0]:
            x = matrix if independent else matrix.copy()
            y = (b.reshape(right * 2, right * 2)
                 if right_identity else np.eye(right * 2, dtype=matrix.dtype))
            reuse_left, reuse_right = False, right_identity
            li, ri = None, "right"
        elif absorb == "right":
            x, y = exact._reduced_qr(matrix)
            reuse_left = reuse_right = False
            li = ri = None
        else:
            q, r = exact._reduced_qr(matrix.T)
            x, y = r.T, q.T
            reuse_left = reuse_right = False
            li = ri = None
        bond = x.shape[1]
        if not reuse_left:
            self._put(i, lhs, x.reshape(left, 2, bond).transpose(0, 2, 1), li)
        if not reuse_right:
            self._put(j, rhs, y.reshape(bond, right, 2), ri)

    @staticmethod
    def _factor_temporary(matrix):
        # Canonicalization consumes R in a newly allocated matrix product
        # before publishing it, unlike the _two_site output-storage case.
        if matrix.shape[0] <= matrix.shape[1]:
            return np.eye(matrix.shape[0], dtype=matrix.dtype), matrix
        return exact._reduced_qr(matrix)

    def left_canonize_site(self,i,bra=None,create_bond=False):
        if not self._identity_gauges:
            return exact.ExactKernelMPS.left_canonize_site(self, i, bra=bra, create_bond=create_bond)
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().left_canonize_site(i,bra=bra,create_bond=create_bond)
        if self._known_identity(i)=="left":
            return
        lhs,rhs=self._layout(i),self._layout(i+1)
        a,b=lhs[1],rhs[1]
        matrix=a.transpose(0,2,1).reshape(a.shape[0]*2,a.shape[1])
        q,r=self._factor_temporary(matrix)
        updated=(r@b.reshape(b.shape[0],-1)).reshape(r.shape[0],b.shape[1],2)
        self._put(i,lhs,q.reshape(a.shape[0],2,q.shape[1]).transpose(0,2,1),
                  "left" if matrix.shape[0]<=matrix.shape[1] else None)
        self._put(i+1,rhs,updated)

    def right_canonize_site(self,i,bra=None,create_bond=False):
        if not self._identity_gauges:
            return exact.ExactKernelMPS.right_canonize_site(self, i, bra=bra, create_bond=create_bond)
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().right_canonize_site(i,bra=bra,create_bond=create_bond)
        if self._known_identity(i)=="right":
            return
        lhs,rhs=self._layout(i-1),self._layout(i)
        a,b=lhs[1],rhs[1]
        matrix=b.reshape(b.shape[0],b.shape[1]*2).T
        q,r=self._factor_temporary(matrix)
        updated=a.transpose(0,2,1).reshape(-1,a.shape[1])@r.T
        self._put(i,rhs,q.T.reshape(q.shape[1],b.shape[1],2),
                  "right" if matrix.shape[0]<=matrix.shape[1] else None)
        self._put(i-1,lhs,updated.reshape(a.shape[0],2,r.shape[0]).transpose(0,2,1))
