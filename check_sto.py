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
if gc:
    bad_cols=[]
    for i,ch in enumerate(gc):
        if ch=='1':
            if any(ss[i] not in '()' for _,ss in leaves):
                bad_cols.append(i)
    print('bad_cols_count',len(bad_cols))
    if bad_cols:
        print('first 10 bad cols:',bad_cols[:10])
