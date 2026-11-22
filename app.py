"""
HIEV-AI MURB Live Load & Headroom Prototype
48-hour training baseline + configurable per-charger max kW + live power sharing under the map.

Run:
    pip install flask requests pandas numpy scikit-learn
    python app.py
Open:
    http://127.0.0.1:5000
"""

from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request
from sklearn.linear_model import Ridge

app = Flask(__name__)

# ---- 1. SETTINGS -----------------------------------------------------------
FASTAPI_BASE = "http://127.0.0.1:8000"
PANEL_LIMIT_KW = 250.0
SAFETY_BUFFER_KW = 10.0
SAFE_LIMIT_KW = PANEL_LIMIT_KW - SAFETY_BUFFER_KW  # 240.0 kW
DEFAULT_MAX_CHARGER_KW = 7.2
TRAINING_HOURS = 48
STEP_MINUTES = 30


def add_time_features(frame: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Adds 24h and 12h sine/cosine features so daily morning + evening peaks wrap smoothly."""
    hour_float = frame[col].dt.hour + frame[col].dt.minute / 60.0
    angle_24h = 2 * np.pi * hour_float / 24.0
    angle_12h = 2 * np.pi * hour_float / 12.0
    out = frame.copy()
    out["sin_hour"] = np.sin(angle_24h)
    out["cos_hour"] = np.cos(angle_24h)
    out["sin_12h"] = np.sin(angle_12h)
    out["cos_12h"] = np.cos(angle_12h)
    return out


# ---- 2. FETCH READINGS -----------------------------------------------------
def get_readings() -> tuple[list[dict], bool]:
    """Pulls recent load readings and steps the meter forward when reachable."""
    try:
        requests.get(f"{FASTAPI_BASE}/reading", timeout=1.0)
        resp = requests.get(f"{FASTAPI_BASE}/history?limit=60", timeout=1.5)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            return rows, True
    except Exception:
        pass

    now = datetime.now()
    fallback = []
    for i in range(30, 0, -1):
        t = now - timedelta(seconds=i * 15)
        h = t.hour + t.minute / 60.0
        wave = 0.22 + 0.45 * math.exp(-((h - 8) ** 2) / 6) + 0.72 * math.exp(-((h - 19) ** 2) / 8)
        load = round(min(SAFE_LIMIT_KW, wave * 200 + np.random.uniform(-3, 3)), 1)
        fallback.append({"time": t.isoformat(), "current_load_kw": load})
    return fallback, False


# ---- 3. FORECAST LOAD & EV HEADROOM ----------------------------------------
def forecast_load_and_headroom(raw_readings: list[dict], hours_ahead: int = 24):
    """Trains on 48 hours of history (backfilled + live) to project demand and headroom."""
    df = pd.DataFrame(raw_readings)[["time", "current_load_kw"]].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # 48 hours of 30-minute steps = 96 training intervals
    training_steps = (TRAINING_HOURS * 60) // STEP_MINUTES
    start_t = df["time"].iloc[0]
    pad_rows = []
    for step in range(training_steps, 0, -1):
        pt = start_t - timedelta(minutes=STEP_MINUTES * step)
        h = pt.hour + pt.minute / 60.0
        base = 45.0 + 55.0 * math.exp(-((h - 8) ** 2) / 5.0) + 155.0 * math.exp(-((h - 19) ** 2) / 5.5)
        pad_rows.append({"time": pt, "current_load_kw": round(base, 1)})

    full_df = pd.concat([pd.DataFrame(pad_rows), df], ignore_index=True)
    full_df = add_time_features(full_df, "time")
    full_df["prev_load"] = full_df["current_load_kw"].shift(1)
    full_df["next_load"] = full_df["current_load_kw"].shift(-1)

    train_df = full_df.dropna().copy()
    features = ["prev_load", "sin_hour", "cos_hour", "sin_12h", "cos_12h"]

    model = Ridge(alpha=0.1)
    model.fit(train_df[features], train_df["next_load"])

    recent_live = df.tail(12).copy()
    chart_points = []
    for _, row in recent_live.iterrows():
        load_val = round(float(row["current_load_kw"]), 1)
        headroom_val = round(max(0.0, SAFE_LIMIT_KW - load_val), 1)
        chart_points.append({
            "time_label": row["time"].strftime("%H:%M:%S"),
            "live_load": load_val,
            "forecast_load": None,
            "ev_headroom": headroom_val,
        })

    last_time = recent_live["time"].iloc[-1]
    last_load = round(float(recent_live["current_load_kw"].iloc[-1]), 1)
    chart_points[-1]["forecast_load"] = last_load

    prev_val = last_load
    future_items = []
    future_steps = (hours_ahead * 60) // STEP_MINUTES
    for step in range(1, future_steps + 1):
        ft = last_time + timedelta(minutes=STEP_MINUTES * step)
        h = ft.hour + ft.minute / 60.0
        a24 = 2 * math.pi * h / 24.0
        a12 = 2 * math.pi * h / 12.0
        x_next = pd.DataFrame([{
            "prev_load": prev_val,
            "sin_hour": math.sin(a24),
            "cos_hour": math.cos(a24),
            "sin_12h": math.sin(a12),
            "cos_12h": math.cos(a12),
        }])
        pred = round(float(np.clip(model.predict(x_next)[0], 0.0, PANEL_LIMIT_KW)), 1)
        headroom = round(max(0.0, SAFE_LIMIT_KW - pred), 1)

        chart_points.append({
            "time_label": ft.strftime("%H:%M"),
            "live_load": None,
            "forecast_load": pred,
            "ev_headroom": headroom,
        })
        future_items.append((ft.strftime("%H:%M"), pred, headroom))
        prev_val = pred

    peak_time, peak_load, _ = max(future_items, key=lambda item: item[1])
    min_time, _, min_headroom = min(future_items, key=lambda item: item[2])

    summary = {
        "live_load_kw": last_load,
        "live_headroom_kw": round(max(0.0, SAFE_LIMIT_KW - last_load), 1),
        "peak_load_kw": peak_load,
        "peak_time": peak_time,
        "min_headroom_kw": min_headroom,
        "min_headroom_time": min_time,
        "default_max_charger_kw": DEFAULT_MAX_CHARGER_KW,
    }
    return chart_points, summary


# ---- 4. ROUTES ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE_HTML)


@app.route("/api/status")
def api_status():
    hours = int(request.args.get("hours", 24))
    readings, is_live = get_readings()
    chart_points, summary = forecast_load_and_headroom(readings, hours_ahead=hours)
    return jsonify({
        "is_live": is_live,
        "safe_limit_kw": SAFE_LIMIT_KW,
        "summary": summary,
        "chart": chart_points,
    })


# ---- 5. MINIMAL UI ---------------------------------------------------------
PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>HIEV-AI • MURB Load & EV Headroom</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body class="bg-neutral-950 text-neutral-100 min-h-screen p-6 antialiased">
  <div class="max-w-6xl mx-auto space-y-6">

    <!-- Minimal Header -->
    <header class="flex flex-wrap justify-between items-center gap-4 pb-4 border-b border-neutral-800">
      <div>
        <div class="flex items-center gap-2.5">
          <span class="h-2 w-2 rounded-full bg-emerald-400"></span>
          <h1 class="text-lg font-medium tracking-tight">MURB Building Load & EV Headroom</h1>
          <span id="source-pill" class="text-[11px] px-2 py-0.5 rounded border border-neutral-800 text-neutral-400">Syncing</span>
        </div>
        <p class="text-xs text-neutral-500 mt-1">Waterloo, ON • 250 kW panel capacity • 240 kW safe ceiling • 48h training window</p>
      </div>

      <div class="flex items-center gap-3 text-xs">
        <label class="flex items-center gap-1.5 text-neutral-400 cursor-pointer">
          <input id="auto-sync" type="checkbox" checked class="accent-emerald-400" />
          Auto-sync (15s)
        </label>
        <select id="horizon" onchange="refreshView()" class="bg-neutral-900 border border-neutral-800 rounded-md px-2.5 py-1.5 text-neutral-200 focus:outline-none">
          <option value="8">8h ahead</option>
          <option value="16">16h ahead</option>
          <option value="24" selected>24h ahead</option>
        </select>
        <button onclick="refreshView()" class="border border-neutral-700 hover:border-neutral-500 text-neutral-200 px-3 py-1.5 rounded-md transition">
          Sync now
        </button>
      </div>
    </header>

    <!-- 4 Key Readouts -->
    <section class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="border border-neutral-800 bg-neutral-900/50 p-4 rounded-lg">
        <p class="text-xs text-neutral-400">Current Building Load</p>
        <p id="kpi-load" class="text-2xl font-semibold text-sky-400 mt-1.5">-- kW</p>
        <p class="text-[11px] text-neutral-500 mt-1">Active household & building draw</p>
      </div>

      <div class="border border-neutral-800 bg-neutral-900/50 p-4 rounded-lg">
        <p class="text-xs text-neutral-400">Safe EV Headroom Now</p>
        <p id="kpi-headroom" class="text-2xl font-semibold text-emerald-400 mt-1.5">-- kW</p>
        <p class="text-[11px] text-neutral-500 mt-1">Spare power below safe ceiling</p>
      </div>

      <div class="border border-neutral-800 bg-neutral-900/50 p-4 rounded-lg">
        <p class="text-xs text-neutral-400">Forecast Peak Load</p>
        <p id="kpi-peak" class="text-2xl font-semibold text-amber-400 mt-1.5">-- kW</p>
        <p id="kpi-peak-time" class="text-[11px] text-neutral-500 mt-1">Expected around --:--</p>
      </div>

      <div class="border border-neutral-800 bg-neutral-900/50 p-4 rounded-lg">
        <p class="text-xs text-neutral-400">Tightest EV Headroom</p>
        <p id="kpi-min-hr" class="text-2xl font-semibold text-rose-400 mt-1.5">-- kW</p>
        <p id="kpi-min-hr-time" class="text-[11px] text-neutral-500 mt-1">Around --:--</p>
      </div>
    </section>

    <!-- Chart & Right Column -->
    <section class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <div class="lg:col-span-2 border border-neutral-800 bg-neutral-900/50 p-5 rounded-lg space-y-4">
        <div class="flex justify-between items-baseline">
          <h2 class="text-sm font-medium text-neutral-200">Building Demand & Available EV Headroom</h2>
          <span class="text-[11px] text-neutral-500">15s refresh</span>
        </div>

        <canvas id="loadChart" height="150"></canvas>

        <!-- Plain-English Graph Explanation -->
        <div class="pt-3 border-t border-neutral-800/80 text-xs text-neutral-400 leading-relaxed space-y-1">
          <p>
            <span class="text-neutral-200 font-medium">How to read this graph:</span>
            The <span class="text-sky-400">solid blue line</span> shows what the building is drawing right now, and the
            <span class="text-amber-400">dashed amber line</span> projects how demand rises into the evening peak and eases overnight.
          </p>
          <p>
            The <span class="text-emerald-400">shaded green band</span> is the safe EV power remaining below the
            <span class="text-rose-400">240 kW ceiling</span>—when building demand climbs in the evening, EV headroom narrows so chargers must share or taper power safely.
          </p>
        </div>
      </div>

      <!-- Map + Live Charger Power Split -->
      <div class="space-y-4">
        <div class="border border-neutral-800 bg-neutral-900/50 p-5 rounded-lg space-y-3">
          <h2 class="text-sm font-medium text-neutral-200">Building Location • Waterloo, ON</h2>
          <div id="map" class="h-44 rounded-md overflow-hidden border border-neutral-800"></div>
          <p class="text-[11px] text-neutral-500">
            Multi-unit residential site • 250 kW panel (10 kW safety buffer).
          </p>
        </div>

        <!-- Simple Live Charger Power Allocation -->
        <div class="border border-neutral-800 bg-neutral-900/50 p-5 rounded-lg space-y-3">
          <div class="flex justify-between items-center">
            <h2 class="text-sm font-medium text-neutral-200">Live Power per Charger</h2>
            <span id="charger-mode" class="text-[11px] text-emerald-400">Rated Max</span>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label for="charger-count" class="text-[11px] text-neutral-400 block mb-1">Active chargers</label>
              <input
                id="charger-count"
                type="number"
                min="1"
                max="200"
                value="10"
                oninput="updateChargerSplit()"
                class="w-full bg-neutral-950 border border-neutral-700 rounded-md px-2.5 py-1.5 text-sm text-neutral-100 focus:outline-none focus:border-neutral-500"
              />
            </div>

            <div>
              <label for="max-charger-kw" class="text-[11px] text-neutral-400 block mb-1">Max kW / charger</label>
              <input
                id="max-charger-kw"
                type="number"
                min="0.5"
                max="350"
                step="0.1"
                value="7.2"
                oninput="updateChargerSplit()"
                class="w-full bg-neutral-950 border border-neutral-700 rounded-md px-2.5 py-1.5 text-sm text-neutral-100 focus:outline-none focus:border-neutral-500"
              />
            </div>
          </div>

          <div class="pt-2 border-t border-neutral-800/80 flex justify-between items-baseline">
            <span class="text-xs text-neutral-400">Available now per charger</span>
            <span id="kw-per-charger" class="text-xl font-semibold text-emerald-400">-- kW</span>
          </div>
          <p class="text-[11px] text-neutral-500">
            Divides current safe headroom across active chargers, capped by your entered charger rating.
          </p>
        </div>
      </div>
    </section>

  </div>

  <script>
    const map = L.map("map").setView([43.4643, -80.5204], 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);
    L.marker([43.4643, -80.5204]).addTo(map).bindPopup("Waterloo MURB • 250 kW Panel");

    let latestHeadroomKw = 0;

    function updateChargerSplit() {
      const count = Math.max(1, parseInt(document.getElementById("charger-count").value, 10) || 1);
      const capKw = Math.max(0.1, parseFloat(document.getElementById("max-charger-kw").value) || 7.2);
      const rawShare = latestHeadroomKw / count;
      const allocated = Math.min(capKw, rawShare);

      document.getElementById("kw-per-charger").textContent = allocated.toFixed(1) + " kW";
      const modeEl = document.getElementById("charger-mode");
      if (rawShare >= capKw) {
        modeEl.textContent = "Up to " + capKw.toFixed(1) + " kW";
        modeEl.className = "text-[11px] text-emerald-400";
      } else {
        modeEl.textContent = "Shared / Tapered";
        modeEl.className = "text-[11px] text-amber-400";
      }
    }

    let chart;
    function drawChart(points, safeLimit) {
      const ctx = document.getElementById("loadChart").getContext("2d");
      if (chart) chart.destroy();

      chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: points.map(p => p.time_label),
          datasets: [
            {
              label: "Current Load (kW)",
              data: points.map(p => p.live_load),
              borderColor: "#38bdf8",
              backgroundColor: "#38bdf8",
              borderWidth: 2,
              pointRadius: 2.5,
              tension: 0.25
            },
            {
              label: "Projected Load (kW)",
              data: points.map(p => p.forecast_load),
              borderColor: "#fbbf24",
              borderDash: [5, 5],
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.3
            },
            {
              label: "Safe EV Headroom (kW)",
              data: points.map(p => p.ev_headroom),
              borderColor: "#34d399",
              backgroundColor: "rgba(52, 211, 153, 0.10)",
              fill: true,
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.3
            },
            {
              label: "Safe Ceiling (240 kW)",
              data: points.map(() => safeLimit),
              borderColor: "#f87171",
              borderDash: [3, 3],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          animation: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { labels: { color: "#a3a3a3", boxWidth: 12, font: { size: 11 } } }
          },
          scales: {
            x: {
              ticks: { color: "#737373", maxTicksLimit: 12, font: { size: 10 } },
              grid: { color: "rgba(115, 115, 115, 0.10)" }
            },
            y: {
              min: 0,
              max: 260,
              ticks: { color: "#737373", font: { size: 10 } },
              grid: { color: "rgba(115, 115, 115, 0.12)" },
              title: { display: true, text: "Power (kW)", color: "#a3a3a3", font: { size: 11 } }
            }
          }
        }
      });
    }

    async function refreshView() {
      const hours = document.getElementById("horizon").value;
      const res = await fetch(`/api/status?hours=${encodeURIComponent(hours)}`);
      const data = await res.json();

      document.getElementById("source-pill").textContent = data.is_live ? "LIVE" : "SIMULATED";
      document.getElementById("kpi-load").textContent = data.summary.live_load_kw + " kW";
      document.getElementById("kpi-headroom").textContent = data.summary.live_headroom_kw + " kW";
      document.getElementById("kpi-peak").textContent = data.summary.peak_load_kw + " kW";
      document.getElementById("kpi-peak-time").textContent = "Expected around " + data.summary.peak_time;
      document.getElementById("kpi-min-hr").textContent = data.summary.min_headroom_kw + " kW";
      document.getElementById("kpi-min-hr-time").textContent = "Around " + data.summary.min_headroom_time;

      latestHeadroomKw = data.summary.live_headroom_kw;
      updateChargerSplit();

      drawChart(data.chart, data.safe_limit_kw);
    }

    refreshView();
    setInterval(() => {
      if (document.getElementById("auto-sync").checked) {
        refreshView();
      }
    }, 15000);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)