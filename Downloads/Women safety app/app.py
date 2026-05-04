"""
SafeMap Bengaluru — Women's Safety Heatmap
Flask Backend with CSV integration + ML Pipeline

Run: python app.py
API: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import random
import json

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'bengaluru_crimes.csv')

# Column name aliases — maps YOUR csv columns to internal names
# Edit these if your CSV has different column names
COLUMN_MAP = {
    'latitude':   ['latitude', 'lat', 'Latitude', 'LAT', 'y'],
    'longitude':  ['longitude', 'lon', 'lng', 'Longitude', 'LON', 'LNG', 'x'],
    'crime_type': ['crime_type', 'crime', 'CrimeType', 'CRIME_TYPE', 'offense', 'Offence', 'category'],
    'severity':   ['severity', 'Severity', 'SEVERITY', 'level', 'Level', 'grade'],
    'time':       ['time', 'Time', 'TIME', 'incident_time', 'reported_time'],
    'date':       ['date', 'Date', 'DATE', 'incident_date'],
    'area':       ['area', 'Area', 'AREA', 'locality', 'Locality', 'location', 'Location', 'place'],
}

CRIME_TYPES = ["Eve Teasing", "Stalking", "Assault", "Robbery", "Harassment", "Kidnapping"]

AREAS = {
    "MG Road":         (12.9756, 77.6097, 0.85),
    "Koramangala":     (12.9352, 77.6245, 0.55),
    "HSR Layout":      (12.9116, 77.6389, 0.45),
    "Whitefield":      (12.9698, 77.7499, 0.60),
    "Marathahalli":    (12.9591, 77.6971, 0.70),
    "Electronic City": (12.8399, 77.6770, 0.50),
    "Jayanagar":       (12.9308, 77.5838, 0.35),
    "Rajajinagar":     (12.9958, 77.5530, 0.65),
    "Yeshwanthpur":    (13.0275, 77.5409, 0.75),
    "KR Puram":        (13.0072, 77.6934, 0.80),
    "Hebbal":          (13.0353, 77.5912, 0.55),
    "Indiranagar":     (12.9784, 77.6408, 0.60),
    "BTM Layout":      (12.9166, 77.6101, 0.50),
    "Banashankari":    (12.9255, 77.5468, 0.40),
    "Yelahanka":       (13.1007, 77.5963, 0.45),
}

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ─── CSV LOADER ───────────────────────────────────────────────────────────────

def resolve_column(df, aliases):
    """Find first matching column from alias list."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def load_csv_data(path):
    """
    Load and normalize crime data from CSV.
    Handles flexible column names and missing fields.
    Returns a cleaned DataFrame.
    """
    print(f"[CSV] Loading from: {path}")
    df = pd.read_csv(path)
    print(f"[CSV] Columns found: {list(df.columns)}")
    print(f"[CSV] Rows loaded: {len(df)}")

    # Resolve column names
    lat_col  = resolve_column(df, COLUMN_MAP['latitude'])
    lon_col  = resolve_column(df, COLUMN_MAP['longitude'])
    type_col = resolve_column(df, COLUMN_MAP['crime_type'])
    sev_col  = resolve_column(df, COLUMN_MAP['severity'])
    time_col = resolve_column(df, COLUMN_MAP['time'])
    area_col = resolve_column(df, COLUMN_MAP['area'])

    if lat_col is None or lon_col is None:
        raise ValueError(f"CSV must have latitude and longitude columns. Found: {list(df.columns)}")

    # Build standardized DataFrame
    result = pd.DataFrame()
    result['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    result['lon'] = pd.to_numeric(df[lon_col], errors='coerce')

    # Crime type
    if type_col:
        result['crime_type'] = df[type_col].fillna('Unknown').astype(str)
    else:
        result['crime_type'] = 'Unknown'

    # Severity (1–3 scale)
    if sev_col:
        result['severity'] = pd.to_numeric(df[sev_col], errors='coerce').fillna(2).clip(1, 3).astype(int)
    else:
        result['severity'] = 2

    # Time → hour
    if time_col:
        def parse_hour(t):
            try:
                return pd.to_datetime(str(t), format='%H:%M').hour
            except:
                try:
                    return pd.to_datetime(str(t)).hour
                except:
                    return random.randint(0, 23)
        result['hour'] = df[time_col].apply(parse_hour)
    else:
        result['hour'] = [random.randint(0, 23) for _ in range(len(df))]

    # Area name
    if area_col:
        result['area'] = df[area_col].fillna('Unknown').astype(str)
    else:
        # Assign area from nearest known lat/lon
        def nearest_area(lat, lon):
            best, best_d = 'Unknown', float('inf')
            for name, (alat, alon, _) in AREAS.items():
                d = (lat - alat)**2 + (lon - alon)**2
                if d < best_d:
                    best, best_d = name, d
            return best
        result['area'] = result.apply(lambda r: nearest_area(r['lat'], r['lon']), axis=1)

    # Derive risk_base from area name
    result['risk_base'] = result['area'].apply(
        lambda a: AREAS.get(a, ('', '', 0.5))[2]
    )

    # Night flag
    result['is_night'] = ((result['hour'] < 6) | (result['hour'] >= 20)).astype(int)

    # Drop rows with invalid coordinates
    result = result.dropna(subset=['lat', 'lon'])
    result = result[
        (result['lat'].between(12.7, 13.2)) &
        (result['lon'].between(77.3, 77.9))
    ]

    print(f"[CSV] Clean rows after validation: {len(result)}")
    return result.reset_index(drop=True)

# ─── SYNTHETIC FALLBACK ───────────────────────────────────────────────────────

def generate_synthetic_data(n=500):
    print("[DATA] No CSV found. Generating synthetic data...")
    records = []
    for _ in range(n):
        area_name = random.choice(list(AREAS.keys()))
        base_lat, base_lon, risk = AREAS[area_name]
        lat = base_lat + random.gauss(0, 0.012)
        lon = base_lon + random.gauss(0, 0.012)
        hour = random.choices(range(24), weights=[
            2,1,1,1,1,2, 3,4,5,5,5,5, 5,5,6,6,7,8, 9,10,9,8,6,4
        ])[0]
        crime_type = random.choices(CRIME_TYPES, weights=[30,20,15,15,15,5])[0]
        severity = random.choices([1,2,3], weights=[50,35,15])[0]
        records.append({
            'area': area_name, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'hour': hour, 'crime_type': crime_type, 'severity': severity,
            'is_night': 1 if hour < 6 or hour >= 20 else 0,
            'risk_base': risk
        })
    return pd.DataFrame(records)

# ─── ML PIPELINE ─────────────────────────────────────────────────────────────

def run_ml_pipeline(df):
    print(f"[ML] Running pipeline on {len(df)} records...")
    n_clusters = min(12, max(3, len(df) // 20))

    coords = df[['lat', 'lon']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(coords_scaled)

    # Cluster stats
    cluster_stats = df.groupby('cluster').agg(
        count=('severity', 'count'),
        avg_severity=('severity', 'mean'),
        night_ratio=('is_night', 'mean'),
        risk_base=('risk_base', 'mean')
    ).reset_index()

    mn, mx = cluster_stats['count'].min(), cluster_stats['count'].max()
    cluster_stats['count_norm'] = (cluster_stats['count'] - mn) / (mx - mn + 1e-9)

    # Logistic Regression risk classification
    X = cluster_stats[['count_norm', 'avg_severity', 'night_ratio', 'risk_base']].values
    composite = (
        0.35 * cluster_stats['count_norm'] +
        0.25 * (cluster_stats['avg_severity'] / 3) +
        0.20 * cluster_stats['night_ratio'] +
        0.20 * cluster_stats['risk_base']
    )
    y = pd.cut(composite, bins=[0, 0.35, 0.60, 1.01], labels=[0, 1, 2]).astype(int)

    try:
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    except TypeError:
        # Older scikit-learn versions may not support multi_class parameter
        lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

    lr.fit(X, y)
    cluster_stats['risk_label'] = lr.predict(X)
    cluster_stats['risk_proba'] = lr.predict_proba(X)[:, 2]

    df = df.merge(cluster_stats[['cluster', 'risk_label', 'risk_proba']], on='cluster', how='left')

    # DBSCAN micro-hotspots
    if len(df) > 10:
        db = DBSCAN(eps=0.008, min_samples=4)
        df['micro_cluster'] = db.fit_predict(coords)
    else:
        df['micro_cluster'] = -1

    print(f"[ML] Done. {n_clusters} clusters, "
          f"{int((cluster_stats['risk_label']==2).sum())} high-risk zones.")
    return df, cluster_stats, kmeans, scaler

# ─── DATA SOURCE INFO ─────────────────────────────────────────────────────────

data_source = 'synthetic'

def load_data():
    global data_source
    if os.path.exists(CSV_PATH):
        try:
            df = load_csv_data(CSV_PATH)
            if len(df) >= 5:
                data_source = 'csv'
                return df
        except Exception as e:
            print(f"[CSV] Error loading CSV: {e}")
            print("[CSV] Falling back to synthetic data.")
    data_source = 'synthetic'
    return generate_synthetic_data()

# ─── STARTUP ──────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  SafeMap Bengaluru — ML Pipeline Starting")
print("="*50)

df_crimes = load_data()
df_crimes, cluster_stats, kmeans_model, scaler_model = run_ml_pipeline(df_crimes)

print(f"\n[READY] Server starting. Data source: {data_source.upper()}")
print(f"[READY] Total incidents: {len(df_crimes)}")
print("="*50 + "\n")

# ─── API ROUTES ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/info')
def info():
    return jsonify({
        'data_source': data_source,
        'csv_path': CSV_PATH,
        'csv_exists': os.path.exists(CSV_PATH),
        'total_records': len(df_crimes),
        'crime_types': df_crimes['crime_type'].unique().tolist(),
        'areas': df_crimes['area'].unique().tolist(),
    })

@app.route('/api/heatmap')
def get_heatmap():
    crime_type = request.args.get('crime_type', 'all')
    time_range = request.args.get('time_range', 'all')

    filtered = df_crimes.copy()
    if crime_type != 'all':
        filtered = filtered[filtered['crime_type'] == crime_type]
    if time_range == 'night':
        filtered = filtered[filtered['is_night'] == 1]
    elif time_range == 'day':
        filtered = filtered[filtered['is_night'] == 0]

    points = []
    for _, row in filtered.iterrows():
        intensity = (row['severity'] / 3) * 0.5 + float(row['risk_proba']) * 0.5
        points.append([row['lat'], row['lon'], round(intensity, 3)])

    return jsonify({'points': points, 'total': len(points)})

@app.route('/api/zones')
def get_zones():
    zones = []
    for _, row in cluster_stats.iterrows():
        cid = int(row['cluster'])
        pts = df_crimes[df_crimes['cluster'] == cid]
        if pts.empty:
            continue
        crime_dist = pts['crime_type'].value_counts().to_dict()
        top_crime = max(crime_dist, key=crime_dist.get) if crime_dist else 'Unknown'
        zones.append({
            'cluster_id': cid,
            'lat': round(float(pts['lat'].mean()), 5),
            'lon': round(float(pts['lon'].mean()), 5),
            'risk_label': RISK_LABELS[int(row['risk_label'])],
            'risk_score': round(float(row['risk_proba']), 3),
            'incident_count': int(row['count']),
            'top_crime': top_crime,
            'night_ratio': round(float(row['night_ratio']), 2),
            'crime_distribution': crime_dist,
        })
    return jsonify({'zones': zones})

@app.route('/api/area-safety')
def area_safety():
    results = []
    for area_name in df_crimes['area'].unique():
        area_data = df_crimes[df_crimes['area'] == area_name]
        if area_data.empty:
            continue
        avg_risk = float(area_data['risk_proba'].mean())
        count = len(area_data)
        night_pct = int(area_data['is_night'].mean() * 100)
        safety_score = round((1 - avg_risk) * 100)

        # Get coordinates
        if area_name in AREAS:
            lat, lon, _ = AREAS[area_name]
        else:
            lat = float(area_data['lat'].mean())
            lon = float(area_data['lon'].mean())

        results.append({
            'area': area_name,
            'lat': lat, 'lon': lon,
            'safety_score': safety_score,
            'risk_level': 'High' if avg_risk > 0.6 else 'Medium' if avg_risk > 0.35 else 'Low',
            'incidents': count,
            'night_crime_pct': night_pct,
            'top_crime': area_data['crime_type'].mode()[0] if not area_data.empty else 'Unknown',
        })

    results.sort(key=lambda x: x['safety_score'])
    return jsonify({'areas': results})

@app.route('/api/stats')
def get_stats():
    total = len(df_crimes)
    high_risk = int((cluster_stats['risk_label'] == 2).sum())
    night_crimes = int(df_crimes['is_night'].sum())
    crime_dist = df_crimes['crime_type'].value_counts().to_dict()
    hourly = df_crimes.groupby('hour').size().to_dict()
    monthly = {}
    if 'date' in df_crimes.columns:
        try:
            df_crimes['month'] = pd.to_datetime(df_crimes['date'], errors='coerce').dt.month
            monthly = df_crimes.groupby('month').size().to_dict()
            monthly = {str(k): int(v) for k, v in monthly.items()}
        except:
            pass

    return jsonify({
        'total_incidents': total,
        'high_risk_zones': high_risk,
        'night_crime_pct': round(night_crimes / total * 100) if total else 0,
        'crime_distribution': {k: int(v) for k, v in crime_dist.items()},
        'hourly_distribution': {str(k): int(v) for k, v in hourly.items()},
        'most_common_crime': df_crimes['crime_type'].mode()[0] if total else 'N/A',
        'data_source': data_source,
        'monthly_distribution': monthly,
    })

@app.route('/api/safe-route')
def safe_route():
    lat = float(request.args.get('lat', 12.9716))
    lon = float(request.args.get('lon', 77.5946))

    point_scaled = scaler_model.transform([[lat, lon]])
    cluster = int(kmeans_model.predict(point_scaled)[0])
    zone_row = cluster_stats[cluster_stats['cluster'] == cluster]

    if zone_row.empty:
        return jsonify({'error': 'No zone data for this point'}), 404

    zone = zone_row.iloc[0]
    risk_score = float(zone['risk_proba'])

    return jsonify({
        'lat': lat, 'lon': lon,
        'cluster': cluster,
        'risk_label': RISK_LABELS[int(zone['risk_label'])],
        'risk_score': round(risk_score, 3),
        'safety_score': round((1 - risk_score) * 100),
        'advice': (
            '⚠️ High-risk zone. Avoid if possible, especially after dark.'
            if risk_score > 0.6 else
            '🟡 Moderate risk. Stay alert and prefer busy routes.'
            if risk_score > 0.35 else
            '✅ Relatively safe zone. Stay aware of surroundings.'
        )
    })

@app.route('/api/crime-types')
def crime_types():
    types = sorted(df_crimes['crime_type'].unique().tolist())
    return jsonify({'crime_types': types})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
