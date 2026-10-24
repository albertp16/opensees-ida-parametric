"""Fig. 2a: exposure B prototype built in OpenSeesPy, rendered with PyVista (true extruded sections).

Model data: DPWH 2018 plans. Columns by mark (S-3 foundation plan, S-7 exposure B, 2nd floor up):
C-4 550 mm at grids 1 and 13; C-1 500 on grid C; C-2 500 on grid B; C-3 400 on corridor grid A.
Beams (S-9, exposure B, 2nd floor): X direction 350x500 on grids B and C, 250x400 on grid A;
Y direction 400x550 over the 7.0 m bay, 250x400 over the 2.5 m corridor.
Beam mark-to-location assignment is by span; check against the S-3 framing plan.
Slabs: 100 mm (S8). Orange spheres: zero-length IMK springs at member ends.
"""
import numpy as np
import openseespy.opensees as ops
import pyvista as pv

xs = np.concatenate([[0, 5.0], 5.0 + 4.5 * np.arange(1, 11), [55.0]])
ys = [0.0, 2.5, 9.5]
zs = [0, 3.15, 6.3, 9.45]
nid = lambda i, j, k: 1 + i + 100 * j + 1000 * k

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)
for k, z in enumerate(zs):
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ops.node(nid(i, j, k), x, y, z)
            if k == 0:
                ops.fix(nid(i, j, k), 1, 1, 1, 1, 1, 1)
ops.geomTransf("PDelta", 1, 0, 1, 0)
ops.geomTransf("Linear", 2, 0, 0, 1)
E = 24.7e6
info = {}
tag = [0]


def ele(n1, n2, b, h, tr, kind):
    tag[0] += 1
    A, Iz, Iy = b * h, b * h**3 / 12, h * b**3 / 12
    J = 0.14 * min(b, h) ** 3 * max(b, h)
    ops.element("elasticBeamColumn", tag[0], n1, n2, A, E, 0.4 * E, J, Iy, Iz, tr)
    info[tag[0]] = (b, h, kind)


for k in range(3):
    for j in range(3):
        for i in range(len(xs)):
            if i in (0, len(xs) - 1):
                ele(nid(i, j, k), nid(i, j, k + 1), 0.55, 0.55, 1, "C-4")
            else:
                s, mk = [(0.40, "C-3"), (0.50, "C-2"), (0.50, "C-1")][j]
                ele(nid(i, j, k), nid(i, j, k + 1), s, s, 1, mk)
for k in range(1, 4):
    for j in range(3):
        b, h = (0.25, 0.40) if j == 0 else (0.35, 0.50)
        for i in range(len(xs) - 1):
            ele(nid(i, j, k), nid(i + 1, j, k), b, h, 2, "beam")
    for i in range(len(xs)):
        ele(nid(i, 0, k), nid(i, 1, k), 0.25, 0.40, 2, "beam")
        ele(nid(i, 1, k), nid(i, 2, k), 0.40, 0.55, 2, "beam")

COL = {"C-1": "#2f6db5", "C-2": "#4f8fd6", "C-3": "#8fbbe8", "C-4": "#173f73", "beam": "#b9b7ae"}
pv.OFF_SCREEN = True
pl = pv.Plotter(off_screen=True, window_size=(3200, 1400))
pl.set_background("white")
blocks = {k: [] for k in COL}
springs = []
for t, (b, h, kind) in info.items():
    n1, n2 = ops.eleNodes(t)
    p1, p2 = np.array(ops.nodeCoord(n1)), np.array(ops.nodeCoord(n2))
    L = np.linalg.norm(p2 - p1)
    c = 0.5 * (p1 + p2)
    if kind == "beam":
        top = h / 2  # beam top flush with floor level
        c = c - np.array([0, 0, top])
        if abs(p2[0] - p1[0]) > 0:
            box = pv.Box(bounds=(c[0] - L / 2, c[0] + L / 2, c[1] - b / 2, c[1] + b / 2, c[2] - h / 2, c[2] + h / 2))
        else:
            box = pv.Box(bounds=(c[0] - b / 2, c[0] + b / 2, c[1] - L / 2, c[1] + L / 2, c[2] - h / 2, c[2] + h / 2))
    else:
        box = pv.Box(bounds=(c[0] - b / 2, c[0] + b / 2, c[1] - h / 2, c[1] + h / 2, p1[2], p2[2]))
    blocks[kind].append(box)
    u = (p2 - p1) / L
    off = 0.45
    springs += [p1 + off * u, p2 - off * u]
for kind, bx in blocks.items():
    pl.add_mesh(pv.merge(bx), color=COL[kind], smooth_shading=False, specular=0.15, ambient=0.25, diffuse=0.85)
for z in zs[1:]:
    slab = pv.Box(bounds=(0, 55, 0, 9.5, z - 0.10, z))
    pl.add_mesh(slab, color="#d9d6cc", opacity=0.22)
pts = pv.PolyData(np.array(springs))
pl.add_mesh(pts.glyph(geom=pv.Sphere(radius=0.16, theta_resolution=12, phi_resolution=12), scale=False),
            color="#e0662f", ambient=0.3)
pl.add_mesh(pv.Box(bounds=(-1, 56, -1, 10.5, -0.06, 0)), color="#eeede8")
pl.enable_anti_aliasing("ssaa")
pl.enable_parallel_projection()
pl.camera_position = [(27.5 - 60, 4.75 - 95, 4.7 + 55), (27.5, 4.75, 4.0), (0, 0, 1)]
pl.reset_camera()
pl.camera.zoom(1.45)
pl.enable_eye_dome_lighting() if False else None
pl.screenshot("fig2a-render.png", transparent_background=False)
print("rendered", len(info), "elements,", len(springs), "springs")
