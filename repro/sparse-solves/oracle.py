"""Independent exact model/implicit-derivative checker. Python stdlib, MIT."""
from fractions import Fraction


def q(x):
    if type(x) is int:return Fraction(x)
    if isinstance(x,Fraction):return x
    raise ValueError('Certificate contains inexact/nonrational data')


def matrix(n,ts):
    if type(n) is not int or n<=0 or not isinstance(ts,list):raise ValueError('Matrix shape')
    a={}
    for i,j,v in ts:
        if type(i) is not int or type(j) is not int or not (0<=i<n and 0<=j<n):raise ValueError('Index')
        a[i,j]=a.get((i,j),Fraction(0))+q(v)
    return a


def mv(n,a,x):
    if len(x)!=n:raise ValueError('Vector shape')
    out=[Fraction(0)]*n
    for (i,j),v in a.items():out[i]+=v*q(x[j])
    return out


def enclosure(c,n,a,b):
    if not isinstance(c,list) or len(c)!=4 or c[0]!='sparse-enclosure-v1' or not isinstance(c[1],list) or len(c[1])!=n or len(b)!=n:return False
    x=list(map(q,c[1]));eta=q(c[2]);radius=q(c[3])
    off=[Fraction(0)]*n
    for (i,j),v in a.items():
        if i!=j:off[i]+=abs(v)
    gap=min(abs(a.get((i,i),0))-off[i] for i in range(n))
    residual=max(abs(q(bi)-axi) for bi,axi in zip(b,mv(n,a,x)))
    return gap>0 and eta==gap and radius==residual/gap


def dense_solve(n,a,b):
    m=[[a.get((i,j),Fraction(0)) for j in range(n)]+[q(b[i])] for i in range(n)]
    for j in range(n):
        p=next(i for i in range(j,n) if m[i][j])
        m[j],m[p]=m[p],m[j]
        z=m[j][j];m[j]=[v/z for v in m[j]]
        for i in range(n):
            if i!=j:
                z=m[i][j];m[i]=[x-z*y for x,y in zip(m[i],m[j])]
    return [row[-1] for row in m]


def validate(record, dense=True):
    ident,n,ts,b,seed,dv,db,jvp,vjp=record
    a=matrix(n,ts)
    if any(not isinstance(v,list) or len(v)!=n for v in [b,seed,db]):raise AssertionError('Source vector shape')
    if any(a.get((j,i),0)!=v for (i,j),v in a.items()):raise AssertionError('Nonsymmetric model')
    slots=sorted(a)
    if len(dv)!=len(slots):raise AssertionError('Parameter pattern')
    da=dict(zip(slots,map(q,dv)))
    if len(jvp)!=8 or jvp[:2]!=['sparse-jvp-v1','proved']:raise AssertionError('JVP not proved')
    if len(vjp)!=7 or vjp[:2]!=['sparse-vjp-v1','proved']:raise AssertionError('VJP not proved')
    pc,tc=jvp[4:6]
    if not enclosure(pc,n,a,b):raise AssertionError('Primal source enclosure')
    rhs=[q(x)-y for x,y in zip(db,mv(n,da,pc[1]))]
    if not enclosure(tc,n,a,rhs):raise AssertionError('Tangent source enclosure')
    row_norm=max(sum(abs(da.get((i,k),0)) for k in range(n)) for i in range(n))
    if jvp[6]!=tc[1] or q(jvp[7])!=q(tc[3])+row_norm*q(pc[3])/q(pc[2]):raise AssertionError('JVP error propagation')
    pc,ac=vjp[4:6]
    if not enclosure(pc,n,a,b) or not enclosure(ac,n,a,seed):raise AssertionError('VJP source enclosures')
    gv,gr,gb,gbr=vjp[6]
    if len(gv)!=len(slots) or len(gr)!=len(slots) or gb!=ac[1] or q(gbr)!=q(ac[3]):raise AssertionError('Cotangent shapes')
    for k,(i,j) in enumerate(slots):
        x,l,rx,rl=q(pc[1][j]),q(ac[1][i]),q(pc[3]),q(ac[3])
        if q(gv[k])!=-l*x or q(gr[k])!=abs(l)*rx+abs(x)*rl+rx*rl:raise AssertionError('Cotangent enclosure')
    if dense:
        x=dense_solve(n,a,b);lam=dense_solve(n,a,seed)
        dx=dense_solve(n,a,[q(z)-y for z,y in zip(db,mv(n,da,x))])
        if any(abs(q(y)-z)>q(jvp[7]) for y,z in zip(jvp[6],dx)):raise AssertionError('Exact dense JVP falsified')
        if any(abs(q(gv[k])+lam[i]*x[j])>q(gr[k]) for k,(i,j) in enumerate(slots)):raise AssertionError('Exact dense VJP falsified')
    return True


if __name__ == '__main__':
    import argparse,json,re
    from pathlib import Path
    def decode(x):
        if isinstance(x,list):return [decode(y) for y in x]
        if isinstance(x,str) and re.fullmatch(r'-?\d+/\d+',x):return Fraction(x)
        return x
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
    r=json.loads(a.receipt.read_text());validate(decode(r['small_witness']))
    for witness in r['large_witnesses']:validate(decode(witness),dense=False)
    print('PASS independent sparse solution/JVP/VJP replay 3')
