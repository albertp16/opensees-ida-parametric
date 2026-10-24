"""Read PEER NGA-West2 .AT2 acceleration files (units g).

Format (PEER NGA database user guide): 3 text header lines, then a line with
'NPTS= n, DT= dt SEC', then the accelerations, several values per line.
"""
import glob, os, re
import numpy as np


def read_at2(path):
    with open(path) as fh:
        lines = fh.readlines()
    m = re.search(r"NPTS\s*=\s*(\d+)\s*,\s*DT\s*=\s*([0-9.Ee+-]+)", lines[3].upper())
    npts, dt = int(m.group(1)), float(m.group(2))
    acc = np.array(" ".join(lines[4:]).split(), dtype=float)[:npts]
    return acc, dt


def find_at2(folder, rsn, comp):
    """Locate the file for RSN and component, e.g. comp 'NORTHR/MUL009' -> RSN953_NORTHR_MUL009.AT2."""
    tail = comp.split("/")[-1].upper()
    hits = [f for f in glob.glob(os.path.join(folder, f"RSN{rsn}_*.AT2"))
            if os.path.basename(f).upper().endswith(tail + ".AT2")]
    if not hits:
        raise FileNotFoundError(f"RSN{rsn} {comp} not found in {folder}")
    return hits[0]
