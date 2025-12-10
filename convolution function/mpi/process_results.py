import csv
import math

rows = []
with open('results.csv', newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

keys = ['H', 'W', 'K', 'S', 'P', 'Cin', 'Cout', 'batch']
baseline = {}
for row in rows:
    try:
        if int(row['ranks']) == 1:
            key = tuple(row[k] for k in keys)
            baseline[key] = float(row['t_total'])
    except ValueError:
        continue

for row in rows:
    key = tuple(row[k] for k in keys)
    try:
        tt = float(row['t_total'])
        ranks = int(row['ranks'])
    except ValueError:
        row['speedup_calc'] = ''
        row['efficiency_calc'] = ''
        row['imgs_per_sec'] = ''
        continue
    base = baseline.get(key)
    if base and tt > 0:
        speedup = base / tt
        eff = speedup / ranks
    else:
        speedup = math.nan
        eff = math.nan
    imgs_per_sec = float(row['batch']) / tt if tt > 0 else math.nan
    row['speedup_calc'] = f"{speedup:.6f}" if not math.isnan(speedup) else ''
    row['efficiency_calc'] = f"{eff:.6f}" if not math.isnan(eff) else ''
    row['imgs_per_sec'] = f"{imgs_per_sec:.6f}" if not math.isnan(imgs_per_sec) else ''

fieldnames = list(rows[0].keys()) if rows else []
for extra in ['speedup_calc', 'efficiency_calc', 'imgs_per_sec']:
    if extra not in fieldnames:
        fieldnames.append(extra)

with open('results_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print('Wrote results_summary.csv with', len(rows), 'rows')
