# SafeMap Bengaluru — Women's Safety Heatmap

AI-powered crime heatmap using K-Means + Logistic Regression + DBSCAN.

---

## Project Structure

```
safemap/
├── app.py                  ← Flask backend + ML pipeline
├── requirements.txt        ← Python dependencies
├── data/
│   └── bengaluru_crimes.csv   ← YOUR CRIME DATA GOES HERE
├── templates/
│   └── index.html          ← Frontend (Leaflet heatmap)
└── static/                 ← (optional) CSS/JS assets
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your CSV (or use the sample one provided)
# data/bengaluru_crimes.csv

# 3. Start server
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## CSV Format

Your CSV must have at minimum **latitude** and **longitude** columns.
All other columns are optional — the app auto-detects them.

### Minimum required:
```csv
latitude,longitude
12.9756,77.6097
12.9591,77.6971
```

### Full format (recommended):
```csv
id,date,time,area,latitude,longitude,crime_type,severity,description
1,2024-01-03,21:30,MG Road,12.9756,77.6097,Eve Teasing,2,Verbal harassment
```

### Supported column name variants:
| Field       | Accepted column names |
|-------------|----------------------|
| Latitude    | latitude, lat, Latitude, LAT, y |
| Longitude   | longitude, lon, lng, Longitude, LON, x |
| Crime Type  | crime_type, crime, CrimeType, offense, category |
| Severity    | severity, level, grade (values: 1=low, 2=medium, 3=high) |
| Time        | time, Time, incident_time (format: HH:MM) |
| Date        | date, Date, incident_date |
| Area        | area, Area, locality, location, place |

---

## Real Data Sources

### NCRB (National Crime Records Bureau)
- https://ncrb.gov.in/crime-in-india-year-wise.html
- Download: "Crime in India" → State/City tables

### Bengaluru City Police
- https://bcp.karnataka.gov.in
- Open data portal (if available)

### data.gov.in
- https://data.gov.in → search "crime against women Karnataka"

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| GET | `/api/heatmap` | Heatmap points [lat, lon, intensity] |
| GET | `/api/zones` | ML-classified risk zones |
| GET | `/api/area-safety` | Safety score per area |
| GET | `/api/stats` | Dashboard summary stats |
| GET | `/api/safe-route?lat=&lon=` | Risk score for any coordinate |
| GET | `/api/crime-types` | List of crime types in dataset |
| GET | `/api/info` | Data source info + column detection |

### Filter params for `/api/heatmap`:
```
?crime_type=Eve Teasing&time_range=night
```

---

## ML Pipeline

```
bengaluru_crimes.csv
      │
      ▼
 [CSV Loader]  →  Auto-detects columns, normalizes lat/lon/time/severity
      │
      ▼
 [K-Means n=12]  →  Groups incidents into 12 geospatial hotspot clusters
      │
      ▼
 [Cluster Stats]  →  count, avg_severity, night_ratio, base_risk per cluster
      │
      ▼
 [Logistic Regression]  →  Classifies each cluster: High / Medium / Low risk
      │
      ▼
 [DBSCAN]  →  Finds dense micro-hotspot sub-clusters for pin markers
      │
      ▼
 [REST API]  →  Serves heatmap points + zone data to Leaflet frontend
```

---

## Resume Talking Points

- End-to-end ML pipeline: data ingestion → clustering → classification → API → visualization
- Flexible CSV parser handles 20+ column name variants with graceful fallback
- K-Means (n=12) for geospatial hotspot detection, DBSCAN for micro-cluster identification
- Logistic Regression classifies risk zones using 4 engineered features
- Flask REST API with 7 endpoints; filtered heatmap re-renders dynamically
- Leaflet.js heatmap with crime type + time-of-day filtering
- Social impact domain: women's safety awareness

---

## Future Enhancements
- [ ] Real NCRB data integration
- [ ] Safe route recommendation (Dijkstra avoiding high-risk clusters)
- [ ] Time-series crime forecasting (ARIMA / LSTM)
- [ ] Mobile PWA with live geolocation alerts
- [ ] Emergency SOS with nearest police station routing
- [ ] Admin dashboard for police data upload
