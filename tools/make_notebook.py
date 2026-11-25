"""Build the Kaggle submission notebook (and a plain .py twin for local gateway tests).

    python tools/make_notebook.py [--policy policy/sb_policy.py] [--title "..."] [--slug ...]
"""
import argparse, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEADER = '''"""Smart Buildings HVAC controller: model-based, rule-driven, no learned weights, no internet."""
import os
import sys

COMPETITION_DATA_FOLDER = os.environ.get(
    "SB_COMPETITION_DATA_FOLDER", "/kaggle/input/competitions/neurips-2026-smart-buildings/")
sys.path.append(os.path.join(COMPETITION_DATA_FOLDER, "kaggle_evaluation"))
sys.path.append(COMPETITION_DATA_FOLDER)

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server
'''

WRAPPER = '''

class MyPolicy(smart_buildings_inference_server.Policy):
    """Thin adapter: builds the controller per scenario and guards every action."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.n_cont = int(building_meta_data["n_continuous_action"])
        self.n_disc = int(building_meta_data["n_discrete_action"])
        self.last = None
        self.errors = 0
        try:
            self.inner = SBPolicy(building_meta_data, action_spec)
        except Exception as exc:  # never fail the submission on a construction error
            print("policy construction failed:", repr(exc), flush=True)
            self.inner = None

    def _fallback(self):
        if self.last is not None:
            return self.last
        return {
            "continuous_action": np.zeros(self.n_cont, dtype=np.float32),
            "discrete_action": np.ones(self.n_disc, dtype=np.int32),
        }

    def get_action(self, obs):
        if self.inner is None:
            return self._fallback()
        try:
            act = self.inner.get_action(np.asarray(obs, dtype=np.float64).ravel())
            cont = np.nan_to_num(np.asarray(act["continuous_action"], dtype=np.float32),
                                 nan=0.0, posinf=1.0, neginf=-1.0)
            cont = np.clip(cont, -1.0, 1.0).astype(np.float32).reshape(self.n_cont)
            disc = np.clip(np.asarray(act["discrete_action"]), 0, 1).astype(np.int32).reshape(self.n_disc)
            self.last = {"continuous_action": cont, "discrete_action": disc}
            return self.last
        except Exception as exc:
            self.errors += 1
            if self.errors <= 5:
                print("get_action failed:", repr(exc), flush=True)
            return self._fallback()


smart_buildings_inference_server.SmartBuildingsInferenceServer(MyPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(REPO / "policy/sb_policy.py"))
    ap.add_argument("--slug", default="sb-zone-model-controller")
    ap.add_argument("--title", default="SB zone model controller")
    ap.add_argument("--user", default="elariztaghiyev")
    ap.add_argument("--out", default=str(REPO / "kaggle"))
    a = ap.parse_args()
    src = Path(a.policy).read_text()
    # drop the module docstring's __future__ import position problem: keep source as is but
    # move "from __future__" lines to the very top of the cell.
    lines = src.splitlines()
    fut = [l for l in lines if l.startswith("from __future__")]
    body = "\n".join(l for l in lines if not l.startswith("from __future__"))
    code = "\n".join(fut) + ("\n" if fut else "") + HEADER + "\n\n" + body + WRAPPER
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "submission_local.py").write_text(code)
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": code.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{a.user}/{a.slug}", "title": a.title, "code_file": f"{a.slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
        "competition_sources": ["neurips-2026-smart-buildings"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
