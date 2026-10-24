"""Checks for the variant generator (run: python -m pytest -q test_variant.py)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from variant_frame import yield_moment, build, pushover
from peer import read_at2


def test_beam_yield_moment_hand_check():
    # 250x450, As = 1125 mm2, d = 0.40 m, fy = 275 MPa: My ~ As fy (0.9 d) = 111 kN m
    assert abs(yield_moment(0.25, 0.45, 0.010, 21, 275, 0.0, "beam") - 111) / 111 < 0.10


def test_axial_load_raises_column_my():
    assert yield_moment(0.4, 0.4, 0.02, 21, 275, 400, "col") > yield_moment(0.4, 0.4, 0.02, 21, 275, 0, "col")


def test_periods_and_strength_in_plausible_range():
    T1 = build({})
    po = pushover({})
    assert 0.3 < T1 < 2.5
    assert 0.05 < po[:, 1].max() < 0.6


def test_lhs_balanced_storeys():
    d = np.genfromtxt(os.path.join(HERE, "..", "..", "data", "lhs_variants_seed2026.csv"), delimiter=",", names=True, skip_header=1)
    assert len(d) == 160 and (d["n_storey"] == 2).sum() == 80


def test_read_at2(tmp_path):
    f = tmp_path / "RSN1_TEST_X.AT2"
    f.write_text("PEER\nEQ\nACCELERATION TIME SERIES IN UNITS OF G\nNPTS=    5, DT=   .0100 SEC\n.1 .2 .3\n.4 .5\n")
    acc, dt = read_at2(str(f))
    assert dt == 0.01 and np.allclose(acc, [.1, .2, .3, .4, .5])
