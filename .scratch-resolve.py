import re,sys
ref=sys.argv[1]
p='CMakeLists.txt'
s=open(p).read()
pat=re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> '+re.escape(ref)+r'\n', re.S)
n=0
def rep(m):
    global n; n+=1
    return m.group(2)+"\n\n"+m.group(1)+"\n"
s=pat.sub(rep,s)
open(p,'w').write(s)
print('cmake hunks resolved:',n)
