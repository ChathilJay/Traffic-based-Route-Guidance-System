import pandas as pd

site_data = pd.read_pickle('Main/site_data.pkl')

short_sites = 0
valid_sites = 0
for site_id, series in site_data.items():
    volumes = [v for v in series.values() if pd.notna(v)]
    if len(volumes) >= 5:
        valid_sites += 1
        print(f"✅ Site {site_id} has {len(volumes)} data points. Example: {volumes[:5]}")
    else:
        short_sites += 1

print(f"\nTotal SCATS sites: {len(site_data)}")
print(f"Sites with valid data (>=5 points): {valid_sites}")
print(f"Sites skipped due to short data: {short_sites}")
