#!/usr/bin/env python3
import csv,hashlib,sys
from pathlib import Path
from PIL import Image

manifest_path=Path('data/processed/manifest.csv')
if not manifest_path.exists():
    print('ERROR: manifest missing', manifest_path)
    sys.exit(2)
rows=list(csv.DictReader(manifest_path.open()))
processed_root=Path('data/processed')
missing=[]
unreadable=[]
wrong_size=[]
wrong_mode=[]
rel_output_paths=set()
rel_output_dup=[]
processed_hash_map={}
source_to_splits={}
per_class_counts={}
split_counts={'train':0,'validation':0,'test':0}
for r in rows:
    rel_out=r['rel_output_path']
    out_path=processed_root/rel_out
    if not out_path.exists():
        missing.append(rel_out)
        continue
    try:
        with Image.open(out_path) as img:
            img.verify()
        with Image.open(out_path) as img:
            mode=img.mode; size=img.size
    except Exception as e:
        unreadable.append((rel_out,str(e)))
        continue
    if size!=(128,128): wrong_size.append(rel_out)
    if mode!='RGB': wrong_mode.append(rel_out)
    if rel_out in rel_output_paths: rel_output_dup.append(rel_out)
    rel_output_paths.add(rel_out)
    h=hashlib.md5(out_path.read_bytes()).hexdigest()
    processed_hash_map.setdefault(h,[]).append((rel_out,r['split']))
    key=r['source_rel_path']
    source_to_splits.setdefault(key,set()).add(r['split'])
    per_class_counts[(r['split'],r['class_name'])]=per_class_counts.get((r['split'],r['class_name']),0)+1
    split_counts[r['split']]=split_counts.get(r['split'],0)+1
hash_dup_groups=[v for k,v in processed_hash_map.items() if len(v)>1]
hash_dup_count=len(hash_dup_groups)
bad_source_paths=[r['source_rel_path'] for r in rows if ('extracted' in r['source_rel_path']) or (not (r['source_rel_path'].startswith('D:/ml-datasets/vehicle-10') or r['source_rel_path'].startswith('D:\\\\ml-datasets\\\\vehicle-10')))]
raw_root=Path('D:/ml-datasets/vehicle-10')
raw_exists=raw_root.exists()
raw_total=None
raw_counts={}
if raw_exists:
    total_raw=0
    for cls in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        n=sum(1 for p in cls.iterdir() if p.is_file())
        raw_counts[cls.name]=n
        total_raw+=n
    raw_total=total_raw
actual_processed_count=sum(1 for p in processed_root.rglob('*') if p.is_file() and p.name!='manifest.csv')
multi_splits=[k for k,v in source_to_splits.items() if len(v)>1]

print('TOTAL_MANIFEST_ROWS',len(rows))
print('ACTUAL_PROCESSED_COUNT',actual_processed_count)
print('MISSING_PROCESSED',len(missing))
print('UNREADABLE',len(unreadable))
print('WRONG_SIZE',len(wrong_size))
print('WRONG_MODE',len(wrong_mode))
print('REL_OUTPUT_DUPLICATES',len(rel_output_dup))
print('HASH_DUPLICATE_GROUPS',hash_dup_count)
print('MULTI_SPLIT_SOURCES',len(multi_splits))
print('BAD_SOURCE_PATHS',len(bad_source_paths))
print('RAW_EXISTS',raw_exists)
if raw_exists:
    print('RAW_TOTAL_FILES',raw_total)
print('SPLIT_COUNTS',split_counts)
print('PER_CLASS_COUNTS_SAMPLE')
for k in sorted(per_class_counts.keys()):
    print('  ',k[0],k[1],per_class_counts[k])
if missing:
    print('\nSAMPLE_MISSING',missing[:10])
if unreadable:
    print('\nSAMPLE_UNREADABLE',unreadable[:10])
if wrong_size:
    print('\nSAMPLE_WRONG_SIZE',wrong_size[:10])
if wrong_mode:
    print('\nSAMPLE_WRONG_MODE',wrong_mode[:10])
if hash_dup_groups:
    print('\nSAMPLE_HASH_DUP_GROUPS_COUNT',len(hash_dup_groups))
    for grp in hash_dup_groups[:5]:
        print('  group:',grp)
if multi_splits:
    print('\nSAMPLE_MULTI_SPLITS',multi_splits[:10])
if bad_source_paths:
    print('\nSAMPLE_BAD_SOURCE_PATHS',bad_source_paths[:10])
