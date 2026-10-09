"""Research-only analytic movement of generated scaled identity gauges.

This ports the earlier scalar-isometry experiment onto the CURRENT routing,
compiler, size threshold and split implementation. No output/state cache,
truncation, rank guess, approximate orthogonality test or global monkeypatch.
Generated tags retain exact value snapshots; outside edits invalidate them.
The production source remains untouched until paired evaluation accepts this.
"""
import types

import numpy as np
import quimb.tensor as qtn

from qibotn.backends import _exact_numpy as exact, _structured_numpy as structured
from qibotn.backends import quimb as production


class GuardedScaledMPS(structured.StructuredExactKernelMPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analytic_gauge_shifts = 0

    def _scaled_identity(self, site):
        item = self._identity_gauges.get(site)
        if item is not None:
            data, snapshot, side, scale = item
            if self[site].data is data and np.array_equal(data, snapshot):
                return side, scale
            self._identity_gauges.pop(site, None)
        return None

    def _known_identity(self, site):
        info = self._scaled_identity(site)
        return info[0] if info is not None and info[1] == 1 else None

    def _put(self, site, layout, array, identity=None, scale=1.):
        self._store(layout, array)
        self._identity_gauges.pop(site, None)
        if identity is not None:
            data = self[site].data
            self._identity_gauges[site] = (data, data.copy(), identity, scale)

    def _multiply(self, site, layout, scalar):
        if scalar == 1:
            return
        info = self._scaled_identity(site)
        self._put(site, layout, layout[1]*scalar, info[0] if info else None,
                  info[1]*scalar if info else 1.)

    def left_canonize_site(self, i, bra=None, create_bond=False):
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().left_canonize_site(i, bra=bra, create_bond=create_bond)
        info = self._scaled_identity(i)
        if info is None or info[1] == 0:
            return super().left_canonize_site(i, bra=bra, create_bond=create_bond)
        side, scale = info
        if side == 'left' and scale == 1:
            return
        lhs, rhs = self._layout(i), self._layout(i+1)
        factor = 1. if side == 'left' else np.sqrt(2.)
        divisor = scale*factor
        self._put(i, lhs, lhs[1]/divisor, side, 1./factor)
        self._multiply(i+1, rhs, divisor)
        self.analytic_gauge_shifts += 1

    def right_canonize_site(self, i, bra=None, create_bond=False):
        if not self._kernel_compatible or bra is not None or create_bond:
            return super().right_canonize_site(i, bra=bra, create_bond=create_bond)
        info = self._scaled_identity(i)
        if info is None or info[1] == 0:
            return super().right_canonize_site(i, bra=bra, create_bond=create_bond)
        side, scale = info
        if side == 'right' and scale == 1:
            return
        lhs, rhs = self._layout(i-1), self._layout(i)
        factor = 1. if side == 'right' else np.sqrt(2.)
        divisor = scale*factor
        self._put(i, rhs, rhs[1]/divisor, side, 1./factor)
        self._multiply(i-1, lhs, divisor)
        self.analytic_gauge_shifts += 1


class GuardedScaledCircuit(qtn.CircuitMPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_type = (GuardedScaledMPS if (1 << self._psi.L)*16 >=
                      GuardedScaledMPS.minimum_merged_bytes else exact.ExactKernelMPS)
        self._psi = state_type(self._psi)


def _clone(original, updates, defaults=None):
    namespace = {**original.__globals__, **updates}
    return types.FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__ if defaults is None else defaults,
                              original.__closure__)


# Retain the current six-gate lookahead and static mixed scheduler verbatim.
# A private function namespace sets the research circuit as BOTH its default
# and its lookahead identity check. No production globals are modified.
_route = _clone(exact.prepare_routed_mps,
                {'ExactKernelCircuitMPS': GuardedScaledCircuit},
                defaults=(GuardedScaledCircuit,))
_prepare = _clone(production._prepare_optimized_mps, {'prepare_routed_mps': _route})


def prepare(backend, circuit):
    state, details = _prepare(backend, circuit)
    return state, {**details, 'research_candidate': 'guarded_scaled_identity',
                   'analytic_gauge_shifts': getattr(state, 'analytic_gauge_shifts', 0)}


def make_backend():
    backend = production.QuimbNumpyBackend()
    entry = _clone(production.exp_value_observable_symbolic,
                   {'_prepare_optimized_mps': prepare})
    backend.exp_value_observable_symbolic = types.MethodType(entry, backend)
    return backend
