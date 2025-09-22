import sys, re
p=sys.argv[1]
leaves=[]
gc=None
with open(p) as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('# STOCK') or line=='//':
            continue
        if line.startswith('#=GR'):
            m=re.match(r"#=GR\s+(\S+)\s+SS\s+(\S+)",line)
            if m:
                name,ss=m.group(1),m.group(2)
                leaves.append((name,ss))
        elif line.startswith('#=GC'):
            m=re.match(r"#=GC\s+CONS\s+(\S+)",line)
            if m:
                gc=m.group(1)
print('leaves',len(leaves),'gc_len',len(gc) if gc else None)
bad_cols=[]
for i,ch in enumerate(gc):
    if ch=='1':
        col=[(name,ss[i]) for name,ss in leaves]
        if any(c not in '()' for _,c in col):
            bad_cols.append((i,col))
print('bad_cols_count',len(bad_cols))
for i,(idx,col) in enumerate(bad_cols[:10]):
    print('idx',idx, 'vals', ''.join(c for _,c in col), 'names',[n for n,_ in col])
