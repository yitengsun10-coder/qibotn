"""Load the pinned official CPU implementation, fixing only its attribute typo.

The benchmark denominator must not include the extra norm contraction added by
our guarded reference. Keep that older result distinct from this reference.
"""
import hashlib
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "7ebbd4d4a00f37b898c552922953591b78b0af7e"
UPSTREAM_PATH = "src/qibotn/backends/quimb.py"


def source_pair():
    original = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_COMMIT}:{UPSTREAM_PATH}"], cwd=ROOT
    ).decode("utf-8")
    old = "optimize=self.contraction_optimizer"
    new = "optimize=self.contractions_optimizer"
    if original.count(old) != 1:
        raise RuntimeError("Pinned upstream typo no longer has one exact match")
    repaired = original.replace(old, new)
    return original, repaired


def load_reference(output=None):
    original, repaired = source_pair()
    module = types.ModuleType("qibotn_benchmark_pinned_upstream")
    exec(compile(repaired, "<pinned-upstream-quimb-typo-only>", "exec"), module.__dict__)
    if output is not None:
        output = Path(output)
        (output / "upstream_quimb_original.py").write_text(original, encoding="utf-8")
        (output / "upstream_quimb_typo_only.py").write_text(repaired, encoding="utf-8")
    metadata = {
        "commit": UPSTREAM_COMMIT,
        "path": UPSTREAM_PATH,
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "typo_only_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
        "sole_change": "contraction_optimizer -> contractions_optimizer at the observable contraction",
        "extra_norm_contraction": False,
        "state_preparations_per_call": 1,
        "algorithm_or_backend_replacement": False,
    }
    return module.QuimbNumpyBackend, metadata
