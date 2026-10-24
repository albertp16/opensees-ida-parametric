"""RS base shear the way MIDAS sums it: CQC of each support reaction, then summed (unsigned), versus
the CQC of the total base shear. Also EX/EY static reactions and roof displacements."""
import sys, json, numpy as np
sys.argv_orig = list(sys.argv); sys.argv = ["x"]
import opensees_vs_midas as ov
ov.DIAPHRAGM = "--diaphragm" in sys.argv_orig if hasattr(sys, "argv_orig") else False
import openseespy.opensees as ops
G = 9.806
D = ov.parse("DPWH_EOSD_loads_v3.mgtx")
mass = ov.build(D)
nm = 9 if ov.DIAPHRAGM else 12
lam = np.array(ops.eigen("-fullGenLapack", nm)); w = np.sqrt(lam)
free = [n for n in D["node"] if D["node"][n][2] > 0]
sup = [n for n in D["node"] if D["node"][n][2] == 0]
phi = {k: {n: (ops.nodeEigenvector(n, k + 1, 1), ops.nodeEigenvector(n, k + 1, 2)) for n in free} for k in range(nm)}
T = np.array([p for p, _ in D["sfunc"]]); S = np.array([v for _, v in D["sfunc"]])
z = 0.05; r = w[:, None] / w[None, :]
rho = 8 * z**2 * (1 + r) * r**1.5 / ((1 - r**2) ** 2 + 4 * z**2 * r * (1 + r) ** 2)
out = {}
for case, d in (("RSX", 0), ("RSY", 1)):
    R = np.zeros((nm, len(sup)))
    for k in range(nm):
        M = {n: mass[n] for n in free}
        mn = sum(M[n] * (phi[k][n][0] ** 2 + phi[k][n][1] ** 2) for n in free)
        L = sum(M[n] * phi[k][n][d] for n in free)
        sa = np.interp(2 * np.pi / w[k], T, S) * 0.176471 * G
        ov.build(D)
        ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
        for n in free:
            f = M[n] * L / mn * sa
            ops.load(n, f * phi[k][n][0], f * phi[k][n][1], 0, 0, 0, 0)
        ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
        ops.algorithm("Linear"); ops.integrator("LoadControl", 1.0); ops.analysis("Static"); ops.analyze(1)
        ops.reactions()
        R[k] = [ops.nodeReaction(n, d + 1) for n in sup]
    node_cqc = np.sqrt(np.einsum("ij,ik,jk->k", rho, R, R))
    base = R.sum(axis=1)
    out[case] = dict(sum_node_cqc=float(node_cqc.sum()), cqc_base=float(np.sqrt(base @ rho @ base)))
for case, d in (("EX", 0), ("EY", 1)):
    ov.build(D)
    ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
    for n, (fx, fy) in D["lat"][case].items():
        ops.load(n, fx, fy, 0, 0, 0, 0)
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.algorithm("Linear"); ops.integrator("LoadControl", 1.0); ops.analysis("Static"); ops.analyze(1)
    ops.reactions()
    roof = [n for n in free if D["node"][n][2] == 9.45]
    out[case] = dict(base=float(-sum(ops.nodeReaction(n, d + 1) for n in sup)),
                     roof_disp_mean=float(np.mean([ops.nodeDisp(n, d + 1) for n in roof])),
                     roof_disp_max=float(max(abs(ops.nodeDisp(n, d + 1)) for n in roof)),
                     reactions={str(n): ops.nodeReaction(n, d + 1) for n in sup})
lam2 = ov.build(D); per = [2*np.pi/np.sqrt(l) for l in ops.eigen("-fullGenLapack", nm)]; out["periods"] = per
json.dump(out, open("opensees_lateral%s.json" % ("_diaphragm" if ov.DIAPHRAGM else ""), "w"), indent=1)
for k, v in out.items():
    print(k, v if isinstance(v, list) else {a: (round(b, 4) if isinstance(b, float) else "...") for a, b in v.items()})
