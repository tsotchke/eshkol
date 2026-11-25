"""Independent sparse-dictionary polynomial oracle over Python Fractions. MIT."""
from fractions import Fraction


def add(a, b, scale=1):
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0) + scale*c
        if not out[m]:
            del out[m]
    return out


def mul(a, b):
    out = {}
    for x, c in a.items():
        for y, d in b.items():
            z = tuple(i+j for i, j in zip(x, y))
            out[z] = out.get(z, 0)+c*d
    return {m:c for m,c in out.items() if c}


def power(a, n, width):
    out = {(0,)*width:Fraction(1)}
    for _ in range(n):
        out = mul(out, a)
    return out


def expand(e, vs):
    if isinstance(e, (int, Fraction)) and type(e) is not bool:
        return {(0,)*len(vs):Fraction(e)} if e else {}
    if isinstance(e, str):
        m = [0]*len(vs)
        m[vs.index(e)] = 1
        return {tuple(m):Fraction(1)}
    op, *xs = e
    if op == 'expt':
        return power(expand(xs[0], vs), xs[1], len(vs))
    ps = [expand(x, vs) for x in xs]
    if op == '*':
        out = {(0,)*len(vs):Fraction(1)}
        for p in ps:
            out = mul(out, p)
        return out
    if op == '+':
        out = {}
        for p in ps:
            out = add(out, p)
        return out
    if op == '-':
        if len(ps) == 1:
            return add({}, ps[0], -1)
        out = ps[0]
        for p in ps[1:]:
            out = add(out, p, -1)
        return out
    raise ValueError(op)


def derivative(p, j):
    out = {}
    for m, c in p.items():
        if m[j]:
            v = list(m)
            v[j] -= 1
            out[tuple(v)] = c*m[j]
    return out


def fiber(vs, f, h, depth, task):
    n = len(vs)
    all_vs = vs+['observation$'+v for v in vs]
    fs = [expand(x, all_vs) for x in f]
    row = [expand(x, all_vs) for x in h]
    jets = []
    for k in range(depth+1):
        jets.extend(row)
        if k < depth:
            nxt = []
            for p in row:
                dp = {}
                for j, fj in enumerate(fs):
                    dp = add(dp, mul(fj, derivative(p, j)))
                nxt.append(dp)
            row = nxt
    def diff(p):
        return add(p, {(0,)*n+m[:n]:c for m,c in p.items()}, -1)
    return all_vs, [diff(p) for p in jets], diff(expand(task, all_vs))


def check(vs, f, h, depth, task, power_, qs):
    if (not isinstance(vs,list) or not vs or not all(isinstance(v,str) for v in vs)
        or len(set(vs))!=len(vs) or not isinstance(f,list) or len(f)!=len(vs)
        or not isinstance(h,list) or type(depth) is not int or depth<0
        or type(power_) is not int or power_<=0 or not isinstance(qs,list)):
        return False
    av, gs, target = fiber(vs, f, h, depth, task)
    if power_ <= 0 or len(qs) != len(gs):
        return False
    lhs = power(target, power_, len(av))
    rhs = {}
    for q, g in zip(qs, gs):
        rhs = add(rhs, mul(expand(q, av), g))
    return lhs == rhs


def expression(p, vs):
    return ['+'] + [['*', c]+[['expt', v, k] for v,k in zip(vs,m) if k]
                    for m,c in p.items()]


def symmetry(vs,f,h,task,ps,embed,pf,perm):
    if (len(vs)!=len(f) or len(vs)!=len(embed) or len(ps)!=len(pf)
        or len(set(ps))!=len(ps) or sorted(perm)!=sorted(ps)):
        return False
    def sub(e,names,values):
        if isinstance(e,str):return dict(zip(names,values)).get(e,e)
        if isinstance(e,list):return [e[0]]+[sub(x,names,values) for x in e[1:]]
        return e
    if [sub(p,ps,perm) for p in perm]!=ps:return False
    fields=[expand(x,ps) for x in pf]
    for fi,ei in zip(f,embed):
        e=expand(ei,ps);de={}
        for j,fj in enumerate(fields):de=add(de,mul(derivative(e,j),fj))
        if de!=expand(sub(fi,vs,embed),ps):return False
    for sensor in h:
        e=sub(sensor,vs,embed)
        if expand(e,ps)!=expand(sub(e,ps,perm),ps):return False
    for fi,p in zip(pf,perm):
        if expand(sub(fi,ps,perm),ps)!=expand(sub(p,ps,pf),ps):return False
    e=sub(task,vs,embed)
    return expand(e,ps)!=expand(sub(e,ps,perm),ps)


if __name__ == '__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
    receipt=json.loads(a.receipt.read_text());models=receipt['source_models']
    if not models or not all(check(*m) for m in models):raise SystemExit('FAIL source model replay')
    symmetries=receipt['symmetry_models']
    if not symmetries or any(symmetry(*m[:-1])!=m[-1] for m in symmetries):raise SystemExit('FAIL symmetry source replay')
    print('PASS independent observation source replay',len(models),'and symmetry controls',len(symmetries))
