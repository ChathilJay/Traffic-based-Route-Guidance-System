import pandas as pd
import re

# Load only 'Data' sheet (skip first 6 rows to reach the main table)
SCATS_DATA_PATH = 'Documents/Scats Data October 2006.xls'
df = pd.read_excel(SCATS_DATA_PATH, sheet_name='Data', skiprows=6)

# Rename first two columns
df.rename(columns={df.columns[0]: 'SiteID', df.columns[1]: 'Street'}, inplace=True)

# Keep only columns that are valid time values (e.g., '00:00:00', '23:45:00')
time_columns = [col for col in df.columns if re.match(r'^\\d{2}:\\d{2}:\\d{2}$', str(col))]

# Melt only those time columns
melted = df.melt(id_vars=['SiteID', 'Street'], value_vars=time_columns, var_name='Time', value_name='Volume')

# Assign a placeholder date
melted['Date'] = '2006-10-01'

# Convert to timestamp
melted['Timestamp'] = pd.to_datetime(melted['Date'] + ' ' + melted['Time'], format='%Y-%m-%d %H:%M:%S')

# Drop rows with missing volume
melted.dropna(subset=['Volume'], inplace=True)

# Group into dictionary
site_data = {}
for site_id, group in melted.groupby('SiteID'):
    traffic_series = dict(zip(group['Timestamp'], group['Volume']))
    site_data[site_id] = traffic_series

# Save the processed dictionary
pd.to_pickle(site_data, 'Main/site_data.pkl')

print(f' Successfully processed {len(site_data)} SCATS sites.')
