import pandas as pd
import json
import ast
from collections import Counter

from pathlib import Path

# Support relative path resolution
file_path = Path(__file__).resolve().parent / "jan to may police violation_anonymized791b166.csv"
if not file_path.exists():
    file_path = Path("/Users/kartikmanmode/gridlock/data/Phase2/jan to may police violation_anonymized791b166.csv")

# Read only necessary columns to save memory
cols = ['latitude', 'longitude', 'vehicle_type', 'violation_type', 'created_datetime', 'police_station']
df = pd.read_csv(file_path, usecols=cols)

print(f"Total Records: {len(df)}")

# 1. Temporal Analysis (Adjusting to IST)
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
# Convert to IST (UTC+5:30)
df['created_datetime_ist'] = df['created_datetime'] + pd.Timedelta(hours=5, minutes=30)
df['hour_ist'] = df['created_datetime_ist'].dt.hour
df['day_of_week'] = df['created_datetime_ist'].dt.day_name()

print("\nViolations by Hour (IST) - Top 10:")
print(df['hour_ist'].value_counts().sort_index())

# 2. Violation Type Distribution
def parse_violations(x):
    try:
        if pd.isna(x): return []
        return ast.literal_eval(x)
    except:
        return []

df['violation_list'] = df['violation_type'].apply(parse_violations)
all_violations = [item for sublist in df['violation_list'] for item in sublist]
violation_counts = Counter(all_violations)

print("\nTop Violation Types:")
for v, count in violation_counts.most_common(10):
    print(f"{v}: {count}")

# 3. Hotspot Analysis (Top Lat/Long clusters)
# Round lat/long to 4 decimal places (~11 meters) to group nearby violations
df['lat_round'] = df['latitude'].round(4)
df['long_round'] = df['longitude'].round(4)
hotspots = df.groupby(['lat_round', 'long_round']).size().reset_index(name='count').sort_values(by='count', ascending=False)

print("\nTop 10 Hotspots (Lat/Long Clusters):")
print(hotspots.head(10))

# 4. Impact Analysis: High-impact parking types
high_impact_types = ["PARKING IN A MAIN ROAD", "PARKING ON FOOTPATH", "DOUBLE PARKING", "PARKING NEAR ROAD CROSSING"]
df['high_impact'] = df['violation_list'].apply(lambda x: any(v in high_impact_types for v in x))

print(f"\nTotal High-Impact Violations: {df['high_impact'].sum()}")
print(f"Percentage High-Impact: {df['high_impact'].mean() * 100:.2f}%")

# 5. Vehicle Type vs High Impact
print("\nTop Vehicle Types for High-Impact Violations:")
print(df[df['high_impact']]['vehicle_type'].value_counts().head(5))
