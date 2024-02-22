import numpy as np
from collections import defaultdict

def read_mesh(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    v = []
    vt = []
    f = []
    ft = []
    for l in lines:
        if l.startswith('v '):
            v.append(list(map(lambda x: float(x), l.split(' ')[1:])))
        if l.startswith('vt '):
            vt.append(list(map(lambda x: float(x), l.split(' ')[1:])))
        if l.startswith('f '):
            if '//' in l: # ignore normal
                f.append(list(map(lambda x: int(x[0]), map(lambda x: x.split('//'), l[1:].strip().split(' ')))))
            elif '/' in l:
                f.append(list(map(lambda x: int(x[0]), map(lambda x: x.split('/'), l[1:].strip().split(' ')))))
                ft.append(list(map(lambda x: int(x[1]), map(lambda x: x.split('/'), l[1:].strip().split(' ')))))
            else:
                f.append(list(map(lambda x: int(x), l.split(' ')[1:])))
    ml = max(map(lambda x: len(x), f))
    f = list(map(lambda x: [i for i in x+[-1] * (ml - len(x))] if len(x) < ml else x, f))
    return np.array(v), np.array(vt), np.array(f) - 1, np.array(ft) - 1 if ft else [] # 0 based idx

def write_mesh(path, v, f):
    f = f + 1
    with open(path, 'w') as mf:
        for vertex in v:
            mf.write('v ')
            mf.write("%6.5f %6.5f %6.5f\n" % tuple(vertex))
        for face in f:
            mf.write('f ')
            if type(face) == np.ndarray and len(face.shape) == 2:
                mf.write("%d %d %d\n" % tuple(face[:, 0]))
            elif face.shape[-1] == 4:
                if face[-1] == -1:
                    mf.write("%d %d %d\n" % tuple(face[:-1]))
                else:
                    mf.write("%d %d %d %d\n" % tuple(face))
            else:
                mf.write("%d %d %d\n" % tuple(face))

def get_valid_tex_coords(vt: np.ndarray, f:np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    v2u = [0] * vt.shape[0]
    for face in f:
        for vtxs in face:
            vtx, tex = vtxs
            v2u[tex] = vtx
    
    tex_nonzeros = np.bitwise_or(np.bitwise_or(vt[:, 0] == 0, vt[:, 0] == 1), np.bitwise_or(vt[:, 1] == 0, vt[:, 1] == 1))
    tex_nonzeros = tex_nonzeros == False

    v2u = np.array(v2u)[tex_nonzeros]

    u2v = defaultdict(list)
    for i, u in enumerate(v2u): # tex idx, vtx idx
        u2v[u].append(i)

    return tex_nonzeros, v2u, u2v


if __name__ == "__main__":
    import os
    path = r"E:\Works\2021\ML\face-denoising\raw\FAobjs"
    actors = os.listdir(path)
    vs = []
    for actor in actors:
        actorpath = os.path.join(path, actor)
        exprs = os.listdir(actorpath)
        for expr in exprs:
            exprpath = os.path.join(actorpath, expr)
            anim = os.listdir(exprpath)[0]
            v, _, f = read_mesh(os.path.join(exprpath, anim))
            vs.append(v)
    vm = np.mean(vs, axis=0)
    write_mesh("coma_mean.obj", vm, f)
