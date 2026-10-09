from functools import partialmethod
from types import SimpleNamespace

import pytest

from benchmarks.asc_revision_cpu.profile_fixed30_bounded import Recorder, effective_memory


def test_host_memory_only(tmp_path):
    assert effective_memory(12345, tmp_path)['available_bytes'] == 12345


@pytest.mark.parametrize('limit,used,expected', [('1000', '200', 800),
    ('1000', '1100', 0), ('max', '0', 5000), ('invalid', '0', 5000),
    (str(2**63), '0', 5000)])
def test_cgroup_limit(tmp_path, limit, used, expected):
    (tmp_path / 'memory.max').write_text(limit)
    (tmp_path / 'memory.current').write_text(used)
    assert effective_memory(5000, tmp_path)['available_bytes'] == expected


def test_cgroup_host_minimum(tmp_path):
    (tmp_path / 'memory.max').write_text('1000')
    (tmp_path / 'memory.current').write_text('200')
    assert effective_memory(700, tmp_path)['available_bytes'] == 700


def test_cgroup_v1(tmp_path):
    (tmp_path / 'memory').mkdir()
    (tmp_path / 'memory/memory.limit_in_bytes').write_text('1000')
    (tmp_path / 'memory/memory.usage_in_bytes').write_text('400')
    assert effective_memory(5000, tmp_path)['available_bytes'] == 600


def test_patch_preserves_value_and_restores():
    owner = SimpleNamespace(f=lambda x: x+1)
    original = owner.f
    recorder = Recorder()
    recorder.patch(owner, 'f', 'f')
    assert owner.f(4) == 5
    assert recorder.stages['f']['completed_calls'] == 1
    recorder.restore()
    assert owner.f is original


def test_patch_restores_inherited_method():
    class Base:
        def f(self, x):
            return x*2
    class Derived(Base):
        pass
    recorder = Recorder()
    recorder.patch(Derived, 'f', 'inherited')
    assert Derived().f(4) == 8
    recorder.restore()
    assert 'f' not in vars(Derived)


def test_patch_restores_partialmethod_descriptor():
    class Example:
        def f(self, x, factor=1):
            return x*factor
        twice = partialmethod(f, factor=2)
    raw = vars(Example)['twice']
    recorder = Recorder()
    recorder.patch(Example, 'twice', 'partialmethod')
    assert Example().twice(4) == 8
    recorder.restore()
    assert vars(Example)['twice'] is raw
    assert Example().twice(5) == 10


def test_nested_exclusive_accounting():
    owner = SimpleNamespace(inner=lambda: 42)
    owner.outer = lambda: owner.inner()
    recorder = Recorder()
    recorder.patch(owner, 'inner', 'inner')
    recorder.patch(owner, 'outer', 'outer')
    assert owner.outer() == 42
    inner, outer = recorder.stages['inner'], recorder.stages['outer']
    assert outer['inclusive_s'] >= inner['inclusive_s']
    assert outer['exclusive_s'] <= outer['inclusive_s']
    assert recorder.stack == []
    recorder.restore()


def test_exception_keeps_original_behavior():
    def fail():
        raise ValueError('original failure')
    owner = SimpleNamespace(f=fail)
    recorder = Recorder()
    recorder.patch(owner, 'f', 'failure')
    with pytest.raises(ValueError, match='original failure'):
        owner.f()
    assert recorder.stages['failure']['calls'] == 1
    assert recorder.stages['failure']['completed_calls'] == 0
    assert recorder.stack == []
    recorder.restore()


def test_guard_runs_before_original():
    called = []
    def guard(extra):
        raise RuntimeError('guard')
    owner = SimpleNamespace(f=lambda: called.append(True))
    recorder = Recorder(guard)
    recorder.patch(owner, 'f', 'guarded')
    with pytest.raises(RuntimeError, match='guard'):
        owner.f()
    assert called == []
    recorder.restore()
