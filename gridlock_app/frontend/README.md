# Gridlock Heatmap: Frontend Application

A production React 18 + Vite web application built with pure JavaScript (JSX), HTML, and CSS according to the **Gridlock Heatmap** product blueprint.

---

## 🚀 Features

1. **MapLibre GL Vector Map Canvas:**
   - Keyless open-source dark vector basemap centered on Bangalore (`12.9500, 77.6200`).
   - Dynamic Geohash-6 bounding polygons with color-coded fills:
     - 🟢 **Green (Low, <35%):** Free flow.
     - 🟡 **Yellow (Medium, 35%–55%):** Moderate commuter flow.
     - 🟠 **Orange (High, 55%–75%):** Heavy congestion.
     - 🔴 **Red / Crimson (Severe, >75%):** Gridlock chokepoint.
2. **15-Minute Playback Slider:**
   - 96 intervals across Day 49 (from 00:00 to 23:45 in 15-minute increments).
   - Play, Pause, and Step controls with speed multipliers (1x, 2x, 4x).
   - Dynamically scrubs through diurnal traffic cycles (Morning Rush 09:00 AM & Evening Gridlock 18:00 PM).
3. **Zone Detail Drawer:**
   - Slides over when clicking any junction.
   - SOTA 93.087% Model Confidence gauge.
   - **Predicted vs. Observed Demand Chart:** Recharts area curve showing Day 48 baseline lag vs. SOTA v47 predictions over 24 hours.
   - Pluviometer Engine context (Core City dampening vs. Outer commuter mode-shift).
4. **Operator Triage Board:**
   - Active violation chokepoints ranked by risk severity.
   - One-Click Officer Dispatch modal (assign officer badge `KA-TP-XXXX`).
5. **Citizen Violation Reporting:**
   - Flag violations (`PARKING`, `SIGNAL`, `SPEEDING`) with severity and description.
6. **Role-Based RBAC Switcher:**
   - Real-time role toggle between `Citizen`, `Operator`, and `Admin`.

---

## 🛠️ Quickstart

### Development Mode:
```bash
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

### Production Build:
```bash
npm run build
npm run preview
```
Proxy routes automatically forward `/api` requests to `http://localhost:8000`.
