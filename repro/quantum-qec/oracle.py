"""Independent matrix-derived Pauli conjugation and finite QEC replay. MIT."""
import itertools

CODES = [
 (7,['IIIXXXX','IXXIIXX','XIXIXIX','IIIZZZZ','IZZIIZZ','ZIZIZIZ']),
 (9,['ZZIIIIIII','IZZIIIIII','IIIZZIIII','IIIIZZIII','IIIIIIZZI','IIIIIIIZZ','XXXXXXIII','IIIXXXXXX']),
 (9,['ZZIZZIIII','IIIIZZIZZ','IXXIXXIII','IIIXXIXXI','XXIIIIIII','IIIIIIIXX','IIZIIZIII','IIIZIIZII'])]
PAULI = {'I':[[1,0],[0,1]],'X':[[0,1],[1,0]],'Y':[[0,-1j],[1j,0]],'Z':[[1,0],[0,-1]]}


def mm(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def adj(a):
    return [[complex(x).conjugate() for x in row] for row in zip(*a)]


def tensor(a,b):
    return [[x*y for x in rowa for y in rowb] for rowa in a for rowb in b]


def local_table(u, width, scale=1):
    words = [''.join(x) for x in itertools.product('IXYZ',repeat=width)]
    matrices = {}
    for w in words:
        p = [[1]]
        for c in w:p=tensor(p,PAULI[c])
        matrices[w]=p
    out={}
    for w,p in matrices.items():
        image=[[scale*x for x in row] for row in mm(mm(u,p),adj(u))]
        for v,q in matrices.items():
            for sign in (1,-1):
                if image==[[sign*x for x in row] for row in q]:out[w]=(sign,v)
        if w not in out:raise AssertionError('Not a Clifford matrix')
    return out


TABLES = {
 'H':local_table([[1,1],[1,-1]],1,0.5),
 'S':local_table([[1,0],[0,1j]],1),
 'Sdg':local_table([[1,0],[0,-1j]],1),
 'X':local_table(PAULI['X'],1), 'Z':local_table(PAULI['Z'],1),
 'CNOT':local_table([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],2)}


def parse(s,n):
    if not isinstance(s,str):raise ValueError('Pauli string')
    sign=-1 if s.startswith('-') else 1
    w=s[1:] if s[:1] in '+-' else s
    if len(w)!=n or any(c not in PAULI for c in w):raise ValueError('Pauli width/alphabet')
    return sign,w


def conjugate(p,g,n):
    sign,w=p
    if not isinstance(g,list) or not g or g[0] not in TABLES:raise ValueError('Gate')
    wires=g[1:]
    if len(wires)!=(2 if g[0]=='CNOT' else 1) or len(set(wires))!=len(wires):raise ValueError('Gate arity')
    if any(type(q) is not int or not 0<=q<n for q in wires):raise ValueError('Wire')
    s,v=TABLES[g[0]][''.join(w[q] for q in wires)]
    chars=list(w)
    for q,c in zip(wires,v):chars[q]=c
    return sign*s,''.join(chars)


def prepare_valid(n,gs,gates):
    for s in gs:
        p=parse(s,n)
        for g in reversed(gates):
            inv=([{'S':'Sdg','Sdg':'S'}[g[0]]]+g[1:]) if g[0] in ('S','Sdg') else g
            p=conjugate(p,inv,n)
        if p[0]!=1 or any(c not in 'IZ' for c in p[1]):return False
    return True


def bits(s,n):
    _,w=parse(s,n)
    return [int(c in 'XY') for c in w]+[int(c in 'YZ') for c in w]


def product(a,b):
    phase=a[0]*b[0];out=''
    for x,y in zip(a[1],b[1]):
        p=mm(PAULI[x],PAULI[y]);found=None
        for c,m in PAULI.items():
            for s in (1,-1,1j,-1j):
                if p==[[s*v for v in row] for row in m]:found=s,c
        if found is None:raise AssertionError('Pauli multiplication')
        phase*=found[0];out+=found[1]
    return phase,out


def inner(a,b,n):
    return sum(a[i]*b[n+i]+a[n+i]*b[i] for i in range(n))%2


def errors(n):
    return ['I'*n]+['I'*q+c+'I'*(n-q-1) for q in range(n) for c in 'XYZ']


def decoder_valid(cert,n,gs):
    if type(n) is not int or n<=0 or not isinstance(gs,list):return False
    if not isinstance(cert,list) or len(cert)!=4 or type(cert[1]) is not int or cert[:3]!=['qec-decoder-v1',n,gs]:return False
    cb=[bits(s,n) for s in gs]
    if any(inner(a,b,n) for a in cb for b in cb):return False
    signed={(1,'I'*n)}
    for s in gs:
        g=parse(s,n)
        signed|={product(v,g) for v in tuple(signed)}
    if (-1,'I'*n) in signed or any(s not in (1,-1) for s,_ in signed):return False
    group={tuple(bits(w,n)) for _,w in signed}
    table={}
    try:
        for row in cert[3]:
            if not isinstance(row,list) or len(row)!=2:return False
            key,rec=row
            if not isinstance(key,list) or len(key)!=len(gs) or any(type(x) is not int or x not in (0,1) for x in key):return False
            if tuple(key) in table:return False
            table[tuple(key)]=bits(rec,n)
    except (ValueError,TypeError):return False
    for s in errors(n):
        e=bits(s,n);key=tuple(inner(c,e,n) for c in cb)
        if key not in table:return False
        if tuple((a+b)%2 for a,b in zip(e,table[key])) not in group:return False
    return True


def lean_source(cert,n,gs):
    """Emit even a false table: the Lean kernel must decide the actual predicate."""
    def lit(x):
        if isinstance(x,list):return '['+', '.join(map(lit,x))+']'
        return str(x)
    rows='['+', '.join('('+lit(key)+', '+lit(bits(s,n))+')' for key,s in cert[3])+']'
    return r'''set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
namespace EshkolQEC
abbrev Bits := List Nat
def xorBits (a b : Bits) : Bits := List.zipWith (fun x y => (x+y)%2) a b
def dot (a b : Bits) : Nat := ((List.zipWith (· * ·) a b).foldl (· + ·) 0)%2
def inner (n : Nat) (a b : Bits) : Nat := (dot (a.take n) (b.drop n)+dot (a.drop n) (b.take n))%2
''' + f'def n : Nat := {n}\ndef checks : List Bits := {lit([bits(s,n) for s in gs])}\ndef recovery : List (Bits × Bits) := {rows}\n'+r'''
def zero : Bits := List.replicate (2*n) 0
def syndrome (e : Bits) : Bits := checks.map (fun c => inner n c e)
def singles : List Bits := (List.range n).flatMap (fun q =>
  [[1,0], [1,1], [0,1]].map (fun xz =>
    (List.range n).map (fun i => if i == q then xz[0]! else 0) ++
    (List.range n).map (fun i => if i == q then xz[1]! else 0)))
def group : List Bits := checks.foldl (fun acc g => acc ++ acc.map (xorBits g)) [zero]
def corrects (e : Bits) : Bool :=
  match recovery.find? (fun row => row.1 == syndrome e) with
  | none => false
  | some row => group.any (fun g => g == xorBits row.2 e)
theorem detects_all_weight_one_errors : singles.all (fun e => (syndrome e).any (fun b => b == 1)) = true := by decide
theorem recovers_identity_and_all_weight_one_errors_mod_phase : (zero :: singles).all corrects = true := by decide
end EshkolQEC
'''


if __name__ == '__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--lean-dir',type=Path);a=p.parse_args()
    records=json.loads(a.receipt.read_text())['witnesses']
    if len(records)!=len(CODES):raise SystemExit('FAIL QEC corpus length')
    for i,(record,expected) in enumerate(zip(records,CODES)):
        n,gs,gates,cert=record
        if (n,gs)!=expected or not prepare_valid(n,gs,gates) or not decoder_valid(cert,n,gs):raise SystemExit('FAIL independent QEC replay')
        if a.lean_dir:
            a.lean_dir.mkdir(exist_ok=True,parents=True)
            (a.lean_dir/f'code{i}.lean').write_text(lean_source(cert,n,gs))
    print('PASS independent signed Clifford/QEC replay',len(records))
