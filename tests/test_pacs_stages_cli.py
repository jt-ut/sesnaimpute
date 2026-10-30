"""Exercises the staged-PACS command line and the submit script against a stub of `pacs_stages` (the real one is
written by another unit). Run: PYTHONPATH=src python3.9 -m unittest tests.test_pacs_stages_cli"""
import os
import re
import subprocess
import sys
import types
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(HERE, "scripts", "submit_pacs_region.sh")
CFG = "/Users/jtaylor/sesna_scratch/ff_checks/root.cfg"
CALLS = []


def _stub():
    m = types.ModuleType("sesnaimpute.sky.derived.pacs_stages")
    TASKS = {0: 10, 1: 3, 2: 0}

    def round_tasks(config, region, band, r, box=None):
        if band == "P160" and r == 0:
            raise FileNotFoundError("detect.pkl")
        return list(range(TASKS.get(r, 0)))
    m.round_tasks = round_tasks
    for name in ("detect", "inject", "pool", "assemble", "finish", "run_region"):
        setattr(m, name, (lambda n: lambda *a, **k: CALLS.append((n, a[1:], k)))(name))
    sys.modules["sesnaimpute.sky.derived.pacs_stages"] = m


class Cli(unittest.TestCase):
    def setUp(self):
        _stub()
        CALLS.clear()
        from sesnaimpute.sky.derived import pacs_stages_cli
        self.main = pacs_stages_cli.main

    def run_cli(self, *a):
        return self.main(["--config", CFG, "--region", "Orion A"] + list(a))

    def test_stages(self):
        self.run_cli("detect", "--band", "P70", "--box", "84", "-5.9", "0.4", "0.4")
        self.assertEqual(CALLS[-1], ("detect", ("Orion A", "P70"), dict(box=(84.0, -5.9, 0.4, 0.4))))
        self.run_cli("inject", "--band", "P100", "--round", "0", "--shard", "2", "--of", "4")
        self.assertEqual(CALLS[-1][:2], ("inject", ("Orion A", "P100", 0, 2, 4)))
        self.run_cli("pool", "--band", "P100", "--shard", "1", "--of", "3")
        self.assertEqual(CALLS[-1][:2], ("pool", ("Orion A", "P100", 1, 3)))
        with self.assertRaises(SystemExit):
            self.run_cli("pool", "--band", "P100", "--shard", "3", "--of", "3")
        with self.assertRaises(SystemExit):
            self.run_cli("inject", "--band", "P70", "--shard", "0")

    def test_inject_rest_stops_at_empty_round(self):
        self.run_cli("inject-rest", "--band", "P70", "--of", "2")
        # round 1 has 3 tasks -> shards 0,1 of round 1; round 2 has none -> stop
        self.assertEqual([c[1][2:] for c in CALLS], [(1, 0, 2), (1, 1, 2)])

    def test_plan(self):
        self.run_cli("plan")  # P160 raises FileNotFoundError in the stub: reported, not fatal


def submit(*a):
    return subprocess.run([SUBMIT, "--dry-run"] + list(a), capture_output=True, text=True)


class Submit(unittest.TestCase):
    def jobs(self, r):
        out = {}
        for line in r.stderr.splitlines():
            name = re.search(r"-J (\S+)", line).group(1)
            dep = re.search(r"--dependency=afterok:(\S+)", line)
            out[name] = dep.group(1).split(":") if dep else []
        return out

    def test_chain_k4_k1(self):
        r = submit("--k", "P70=4,P100=1,P160=4", "stampede3", "Orion A", "/c.cfg")
        self.assertEqual(r.returncode, 0, r.stderr)
        j = self.jobs(r)
        self.assertEqual(len(j), 29)
        p = "pacs_Orion_A_"
        self.assertEqual(j[p + "P70_irest"], [p + "P70_i0_s%dof4" % s for s in range(4)])
        self.assertEqual(j[p + "P100_pool_s0of1"], [p + "P100_irest"])
        self.assertEqual(j[p + "finish"], [p + b + "_assemble" for b in ("P70", "P100", "P160")])
        self.assertIn("-c 112", r.stderr)
        self.assertNotIn("propagate", r.stderr)
        for deps in j.values():
            for d in deps:
                self.assertIn(d, j)

    def test_vista_header(self):
        r = submit("vista", "Orion A", "/c.cfg")
        self.assertIn("-p gg", r.stderr); self.assertIn("-c 144", r.stderr); self.assertIn("--propagate=NONE", r.stderr)
        self.assertEqual(len(self.jobs(r)), 2 + 3 * 5)

    def test_limits(self):
        r = submit("--k", "P70=8,P100=8,P160=8", "stampede3", "Orion A", "/c.cfg")  # 2 + 3*19 = 59 > 36
        self.assertEqual(r.returncode, 1)
        self.assertIn("refused", r.stderr)
        r = submit("--k", "P70=7,P100=7,P160=7", "vista", "Orion A", "/c.cfg")  # 21 running > 20
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
