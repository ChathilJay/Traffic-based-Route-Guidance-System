import pandas as pd

df = pd.read_excel('Documents/Scats Data October 2006.xls', sheet_name='Data', skiprows=6)

df.rename(columns={df.columns[0]: 'SiteID', df.columns[1]: 'Street'}, inplace=True)

# Convert all column headers to string and keep those that look like time
def is_time_like(val):
    try:
        pd.to_datetime(str(val), format='%H:%M:%S')
        return True
    except:
        return False

time_columns = [col for col in df.columns if is_time_like(col)]

print(f"✅ Fixed: Detected {len(time_columns)} time columns.")
print("🕒 First few time columns:", time_columns[:5])

# Melt those time columns
melted = df.melt(id_vars=['SiteID', 'Street'], value_vars=time_columns, var_name='Time', value_name='Volume')
melted['Date'] = '2006-10-01'
melted['Timestamp'] = pd.to_datetime(melted['Date'] + ' ' + melted['Time'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
melted.dropna(subset=['Volume', 'Timestamp'], inplace=True)

print("🧪 Melted sample:\n", melted.head())
