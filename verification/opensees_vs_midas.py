"""Sect. 4 verification: rebuild the MIDAS Gen NX model (exported .mgtx) in OpenSeesPy and compare
periods, mass participation, gravity reactions and response-spectrum base shears.

Same input as MIDAS: nodes, elements, sections and nodal loads are read from the exported MGTX file.
Elements: elasticTimoshenkoBeam (MIDAS frame elements include shear deformation; As = 5/6 A),
St Venant J of a rectangle. Mass: member self-weight split to the end nodes (X, Y only, as
Structure Type 'convert self-weight to masses X, Y') + load-to-mass 1.05 DL + 1.05 SDL + 0.25 LL of the
nodal loads. No rigid diaphragm (none defined in the MIDAS model).
Response spectrum: modal CQC (5%), spectrum from the MGTX *SFUNCTION table, scale from *SPLDCASE.
"""
import re, sys, math, json
import numpy as np
import openseespy.opensees as ops

G = 9.806
DIAPHRAGM = False   # set True (or --diaphragm) when the MIDAS model has Story data with rigid diaphragms


def parse(path):
    txt = open(path, encoding="latin-1").read().splitlines()
    sec, cur, lc = None, {}, None
    D = dict(node={}, ele={}, sect={}, mat={}, load={"DL": {}, "SDL": {}, "LL": {}}, lat={}, sfunc=[], splc=[])
    for line in txt:
        s = line.split(";")[0].rstrip()
        if not s.strip():
            continue
        if s.startswith("*"):
            head = s.split(",")
            sec = head[0].strip()
            if sec == "*USE-STLD":
                lc = head[1].strip()
            continue
        f = [x.strip() for x in s.split(",")]
        if sec == "*NODE":
            D["node"][int(f[0])] = tuple(map(float, f[1:4]))
        elif sec == "*ELEMENT" and f[1] == "BEAM":
            D["ele"][int(f[0])] = (int(f[2]), int(f[3]), int(f[4]), int(f[5]), float(f[6]))
        elif sec == "*SECTION":
            D["sect"][int(f[0])] = (float(f[14]), float(f[15]))       # H, B for SB
        elif sec == "*MATERIAL":
            D["mat"][int(f[0])] = dict(E=float(f[10]), nu=float(f[11]), gamma=float(f[13]))
        elif sec == "*CONLOAD":
            n = int(f[0])
            if lc in D["load"]:
                D["load"][lc][n] = D["load"][lc].get(n, 0.0) + float(f[3])
            else:                                   # lateral cases (EX, EY)
                D["lat"].setdefault(lc, {})[n] = (float(f[1]), float(f[2]))
        elif sec == "*SFUNCTION":
            if f[0].startswith("FUNC="):
                D["sfunc_scale"] = float(f[3])
            elif re.match(r"^-?\d", f[0]):
                v = [float(x) for x in f if x]
                D["sfunc"] += list(zip(v[0::2], v[1::2]))
        elif sec == "*SPLDCASE" and f[0].startswith("NAME="):
            D["splc"].append(dict(name=f[0][5:], angle=float(f[2]), scale=float(f[3])))
    return D


def rect_J(b, h):
    a, c = max(b, h), min(b, h)
    return c**3 * a * (1 / 3 - 0.21 * c / a * (1 - (c / a) ** 4 / 12))


def build(D):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n, xyz in D["node"].items():
        ops.node(n, *xyz)
        if xyz[2] == 0.0:
            ops.fix(n, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 1, 0, 0)    # columns
    ops.geomTransf("Linear", 2, 0, 0, 1)    # beams: local z vertical
    mass = {n: 0.0 for n in D["node"]}
    m = D["mat"][1]
    Gs = m["E"] / (2 * (1 + m["nu"]))
    for e, (im, isec, n1, n2, ang) in D["ele"].items():
        H, B = D["sect"][isec]
        A = B * H
        p1, p2 = np.array(D["node"][n1]), np.array(D["node"][n2])
        L = np.linalg.norm(p2 - p1)
        vertical = abs(p2[2] - p1[2]) > 1e-6
        if vertical:
            Iy, Iz, tr = H * B**3 / 12, B * H**3 / 12, 1
        else:
            Iy, Iz, tr = B * H**3 / 12, H * B**3 / 12, 2     # strong axis about local y (vertical bending)
        ops.element("ElasticTimoshenkoBeam", e, n1, n2, m["E"], Gs, A, rect_J(B, H), Iy, Iz, 5 / 6 * A, 5 / 6 * A, tr)
        w = A * L * m["gamma"]
        mass[n1] += w / 2 / G
        mass[n2] += w / 2 / G
    for lc, fac in (("DL", 1.05), ("SDL", 1.05), ("LL", 0.25)):
        for n, fz in D["load"][lc].items():
            mass[n] += fac * abs(fz) / G
    for n, mm in mass.items():
        if D["node"][n][2] > 0:
            ops.mass(n, mm, mm, 0, 0, 0, 0)
    if DIAPHRAGM:     # rigid floor per level (MIDAS Story data with rigid diaphragm), master at the mass centre
        lev = {}
        for n, (x, y, z) in D["node"].items():
            if z > 0:
                lev.setdefault(z, []).append(n)
        for k, (z, ns) in enumerate(sorted(lev.items())):
            mt = sum(mass[n] for n in ns)
            xc = sum(mass[n] * D["node"][n][0] for n in ns) / mt
            yc = sum(mass[n] * D["node"][n][1] for n in ns) / mt
            m = 90000 + k
            ops.node(m, xc, yc, z)
            ops.fix(m, 0, 0, 1, 1, 1, 0)
            ops.rigidDiaphragm(3, m, *ns)
    return mass


def gravity(D, lc):
    """Static reactions for one case: DL includes member self-weight as element loads."""
    build(D)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    if lc == "DL":
        g = D["mat"][1]["gamma"]
        for e, (im, isec, n1, n2, ang) in D["ele"].items():
            H, B = D["sect"][isec]
            p1, p2 = np.array(D["node"][n1]), np.array(D["node"][n2])
            if abs(p2[2] - p1[2]) > 1e-6:
                ops.eleLoad("-ele", e, "-type", "-beamUniform", 0.0, 0.0, -B * H * g)   # axial (local x = Z up)
            else:
                ops.eleLoad("-ele", e, "-type", "-beamUniform", 0.0, -B * H * g, 0.0)   # local z vertical
    for n, fz in D["load"][lc].items():
        ops.load(n, 0, 0, fz, 0, 0, 0)
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-10, 10); ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0); ops.analysis("Static"); ops.analyze(1)
    ops.reactions()
    return {n: ops.nodeReaction(n, 3) for n, xyz in D["node"].items() if xyz[2] == 0.0}


def modal(D, nmodes=12):
    mass = build(D)
    lam = np.array(ops.eigen("-fullGenLapack", nmodes))
    w = np.sqrt(lam)
    free = [n for n in D["node"] if D["node"][n][2] > 0]
    M = np.array([mass[n] for n in free])
    tot = M.sum()
    out = []
    for k in range(nmodes):
        phx = np.array([ops.nodeEigenvector(n, k + 1, 1) for n in free])
        phy = np.array([ops.nodeEigenvector(n, k + 1, 2) for n in free])
        mn = (M * (phx**2 + phy**2)).sum()
        Lx, Ly = (M * phx).sum(), (M * phy).sum()
        out.append(dict(T=2 * math.pi / w[k], w=w[k], mx=Lx**2 / mn / tot, my=Ly**2 / mn / tot, Lx=Lx, Ly=Ly, mn=mn))
    return out, tot


def rs_base_shear(modes, spec, scale, direction, zeta=0.05):
    T = np.array([p for p, _ in spec]); S = np.array([v for _, v in spec])
    V = []
    for md in modes:
        sa = np.interp(md["T"], T, S) * scale * G
        L = md["Lx"] if direction == "X" else md["Ly"]
        V.append(L**2 / md["mn"] * sa)
    V = np.array(V); w = np.array([m["w"] for m in modes])
    r = w[:, None] / w[None, :]
    rho = 8 * zeta**2 * (1 + r) * r**1.5 / ((1 - r**2) ** 2 + 4 * zeta**2 * r * (1 + r) ** 2)
    return float(np.sqrt(V @ rho @ V))


if __name__ == "__main__":
    D = parse(sys.argv[1] if len(sys.argv) > 1 else "DPWH_EOSD_loads.mgtx")
    print("nodes", len(D["node"]), "elements", len(D["ele"]), "spectrum points", len(D["sfunc"]), "RS cases", D["splc"])
    R = {lc: gravity(D, lc) for lc in ("DL", "SDL", "LL")}
    for lc in R:
        print(lc, "total FZ", round(sum(R[lc].values()), 1))
    modes, tot = modal(D)
    print("total mass (t)", round(tot, 1), "W (kN)", round(tot * G, 0))
    for k, md in enumerate(modes):
        print(f"mode {k+1}: T={md['T']:.4f} s  mx={100*md['mx']:.2f}%  my={100*md['my']:.2f}%")
    res = dict(R={lc: {str(n): v for n, v in r.items()} for lc, r in R.items()}, modes=modes, W=tot * G)
    for c in D["splc"]:
        d = "X" if abs(c["angle"]) < 45 else "Y"
        res["V_" + c["name"]] = rs_base_shear(modes, D["sfunc"], c["scale"] * D.get("sfunc_scale", 1), d)
        print(c["name"], "base shear (kN)", round(res["V_" + c["name"]], 1))
    json.dump(res, open("opensees_results.json", "w"), indent=1, default=float)
