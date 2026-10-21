from pathlib import Path

def balance(path):
    text=Path(path).read_text(); depth=0; state='code'; i=0; line=1
    while i < len(text):
        c=text[i]
        if c=='\n': line+=1
        if state=='comment':
            if c=='\n': state='code'
        elif state=='string':
            if c=='\\': i+=1
            elif c=='"': state='code'
        elif state=='char':
            if c.isspace() or c in '()': state='code'; continue
        else:
            if c==';': state='comment'
            elif c=='"': state='string'
            elif c=='#' and text[i:i+2]=='#\\': state='char'; i+=1
            elif c=='(': depth+=1
            elif c==')':
                depth-=1
                if depth<0: raise ValueError(f'{path}:{line}: close before open')
        i+=1
    if state=='string': raise ValueError(f'{path}: unterminated string')
    if depth: raise ValueError(f'{path}: paren depth {depth}')
    print(f'BALANCE_PASS {path}')

balance('lib/core/msgpack.esk')
balance('tests/v1_2_edge_cases/msgpack_utf8_test.esk')
