/**
 * app.js
 * ======
 * Client-side engine for the BMS Digital Twin & SOC Estimation Dashboard.
 * Handles:
 *   - Real-time drive cycle replay loop & timeline scrubbing
 *   - Canvas 2D responsive time-series chart rendering
 *   - Hardware "What-If" parameter slider debounced inference
 *   - Tab navigation & model comparative overlays
 */

(function () {
  "use strict";

  // -------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------
  const state = {
    activeTab: "panelReplayer",
    activePlotModel: "rf", // 'rf', 'dt', 'knn', 'all'
    trips: [],
    selectedTripId: null,
    tripData: null,
    
    // Replay Clock State
    playback: {
      isPlaying: false,
      currentIndex: 0,
      speedMultiplier: 1,
      lastTimestamp: 0,
      accumulatedTime: 0,
      rafId: null
    },

    // Sandbox Input State
    sandbox: {
      voltage: 351.4,
      current: -14.8,
      batteryTemp: 27.3,
      ambientTemp: 22.0,
      debounceTimer: null
    }
  };

  // -------------------------------------------------------------
  // DOM References
  // -------------------------------------------------------------
  const el = {
    // Badges & Telemetry Readouts
    activeModelBadge: document.getElementById("activeModelBadge"),
    engineStatusBadge: document.getElementById("engineStatusBadge"),
    dispTimestampBadge: document.getElementById("dispTimestampBadge"),
    dispSoc: document.getElementById("dispSoc"),
    socBarFill: document.getElementById("socBarFill"),
    dispSocSub: document.getElementById("dispSocSub"),
    socModelPill: document.getElementById("socModelPill"),
    dispVoltage: document.getElementById("dispVoltage"),
    dispPower: document.getElementById("dispPower"),
    dispCurrent: document.getElementById("dispCurrent"),
    dispCurrentState: document.getElementById("dispCurrentState"),
    dispSpeed: document.getElementById("dispSpeed"),
    dispTemp: document.getElementById("dispTemp"),
    dispRange: document.getElementById("dispRange"),

    // Tabs & Panels
    tabs: document.querySelectorAll(".nav-tab"),
    panels: document.querySelectorAll(".panel-section"),

    // Replayer
    selectTrip: document.getElementById("selectTrip"),
    metaDuration: document.getElementById("metaDuration"),
    metaSocRange: document.getElementById("metaSocRange"),
    metaSpeed: document.getElementById("metaSpeed"),
    metaSamples: document.getElementById("metaSamples"),
    btnPlayPause: document.getElementById("btnPlayPause"),
    btnReset: document.getElementById("btnReset"),
    timelineScrubber: document.getElementById("timelineScrubber"),
    dispCurrentTime: document.getElementById("dispCurrentTime"),
    speedButtons: document.querySelectorAll(".btn-speed"),
    toggleChips: document.querySelectorAll(".toggle-chip"),
    legendDt: document.getElementById("legendDt"),
    legendKnn: document.getElementById("legendKnn"),
    statTripRmse: document.getElementById("statTripRmse"),
    statTripMae: document.getElementById("statTripMae"),
    statTripMaxErr: document.getElementById("statTripMaxErr"),
    statInstantErr: document.getElementById("statInstantErr"),
    canvas: document.getElementById("socChart"),

    // Sandbox Sliders & Scores
    sliderVolt: document.getElementById("sliderVolt"),
    sliderCurr: document.getElementById("sliderCurr"),
    sliderBattTemp: document.getElementById("sliderBattTemp"),
    sliderAmbTemp: document.getElementById("sliderAmbTemp"),
    valSliderVolt: document.getElementById("valSliderVolt"),
    valSliderCurr: document.getElementById("valSliderCurr"),
    valSliderBattTemp: document.getElementById("valSliderBattTemp"),
    valSliderAmbTemp: document.getElementById("valSliderAmbTemp"),
    btnResetSandbox: document.getElementById("btnResetSandbox"),
    sandboxRfScore: document.getElementById("sandboxRfScore"),
    sandboxDtScore: document.getElementById("sandboxDtScore"),
    sandboxKnnScore: document.getElementById("sandboxKnnScore"),
    sandboxStateDesc: document.getElementById("sandboxStateDesc")
  };

  // -------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------
  async function init() {
    setupTabNavigation();
    setupModelToggles();
    setupPlaybackControls();
    setupSandboxControls();
    setupCanvasDPI();

    window.addEventListener("resize", () => {
      setupCanvasDPI();
      renderChart();
    });

    await fetchTripCatalog();
    await runSandboxInference();
  }

  // -------------------------------------------------------------
  // Navigation & Model Toggles
  // -------------------------------------------------------------
  function setupTabNavigation() {
    el.tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetId = tab.dataset.target;
        state.activeTab = targetId;

        el.tabs.forEach((t) => t.classList.remove("active"));
        el.panels.forEach((p) => p.classList.remove("active"));

        tab.classList.add("active");
        const panel = document.getElementById(targetId);
        if (panel) panel.classList.add("active");

        if (targetId === "panelReplayer") {
          setupCanvasDPI();
          renderChart();
          if (state.tripData && state.tripData.points.length > 0) {
            updateTelemetryDisplay(state.tripData.points[state.playback.currentIndex]);
          }
        } else if (targetId === "panelSandbox") {
          pausePlayback();
          runSandboxInference();
        }
      });
    });
  }

  function setupModelToggles() {
    el.toggleChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        el.toggleChips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        state.activePlotModel = chip.dataset.model;

        // Toggle legends
        if (state.activePlotModel === "dt" || state.activePlotModel === "all") {
          el.legendDt.style.display = "flex";
        } else {
          el.legendDt.style.display = "none";
        }

        if (state.activePlotModel === "knn" || state.activePlotModel === "all") {
          el.legendKnn.style.display = "flex";
        } else {
          el.legendKnn.style.display = "none";
        }

        const modelNames = {
          rf: "Active: Random Forest (25 Trees)",
          dt: "Active: Decision Tree (Single)",
          knn: "Active: k-Nearest Neighbors (k=5)",
          all: "Active: Multi-Model Overlay"
        };
        el.activeModelBadge.textContent = modelNames[state.activePlotModel] || "Active: Random Forest";
        el.socModelPill.textContent = state.activePlotModel.toUpperCase();

        renderChart();
        if (state.tripData && state.tripData.points.length > 0) {
          updateTelemetryDisplay(state.tripData.points[state.playback.currentIndex]);
        }
      });
    });
  }

  // -------------------------------------------------------------
  // Data Fetching: Trip Catalog & Single Trajectory
  // -------------------------------------------------------------
  async function fetchTripCatalog() {
    try {
      const res = await fetch("/api/trips");
      const data = await res.json();
      if (!data.success || !data.trips.length) {
        throw new Error("No trip datasets returned.");
      }

      state.trips = data.trips;
      el.selectTrip.innerHTML = "";

      state.trips.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.id;
        opt.textContent = `${t.id} — ${t.duration_min} min (${t.initial_soc}% -> ${t.final_soc}% SOC) — ${t.description}`;
        el.selectTrip.appendChild(opt);
      });

      // Default to TripB30 (Highway & Urban profile)
      const defaultTrip = state.trips.find((t) => t.id === "TripB30") || state.trips[0];
      el.selectTrip.value = defaultTrip.id;
      await loadTrip(defaultTrip.id);
    } catch (err) {
      console.error("Failed to fetch trip catalog:", err);
      el.selectTrip.innerHTML = '<option value="">Error loading test trips</option>';
    }
  }

  async function loadTrip(tripId) {
    pausePlayback();
    state.selectedTripId = tripId;
    el.engineStatusBadge.textContent = "Loading Trip...";

    const meta = state.trips.find((t) => t.id === tripId);
    if (meta) {
      el.metaDuration.textContent = `${meta.duration_min} min (${meta.duration_s}s)`;
      el.metaSocRange.textContent = `${meta.initial_soc}% -> ${meta.final_soc}% (-${meta.soc_delta}%)`;
      el.metaSpeed.textContent = `${meta.avg_speed_kmh} km/h`;
      el.metaSamples.textContent = `${meta.samples_count.toLocaleString()} pts`;
    }

    try {
      // Step=5 means 1 point every 0.5 seconds for buttery 60fps playback
      const res = await fetch(`/api/trip/${tripId}?step=5`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error);

      state.tripData = data.data;
      state.playback.currentIndex = 0;
      el.timelineScrubber.max = state.tripData.points.length - 1;
      el.timelineScrubber.value = 0;

      // Update Trip Error Metrics
      const m = state.tripData.trip_metrics[state.activePlotModel] || state.tripData.trip_metrics.rf;
      el.statTripRmse.textContent = `${m.rmse.toFixed(2)}%`;
      el.statTripMae.textContent = `${m.mae.toFixed(2)}%`;
      el.statTripMaxErr.textContent = `${m.max_err.toFixed(2)}%`;

      el.engineStatusBadge.textContent = "Telemetry Ready";
      updateTelemetryDisplay(state.tripData.points[0]);
      renderChart();
    } catch (err) {
      console.error(`Failed to load ${tripId}:`, err);
      el.engineStatusBadge.textContent = "Trip Load Error";
    }
  }

  el.selectTrip.addEventListener("change", (e) => {
    loadTrip(e.target.value);
  });

  // -------------------------------------------------------------
  // Playback Engine & Time Sync
  // -------------------------------------------------------------
  function setupPlaybackControls() {
    el.btnPlayPause.addEventListener("click", togglePlayPause);
    el.btnReset.addEventListener("click", resetPlayback);

    el.timelineScrubber.addEventListener("input", (e) => {
      if (!state.tripData) return;
      state.playback.currentIndex = parseInt(e.target.value, 10);
      updateTelemetryDisplay(state.tripData.points[state.playback.currentIndex]);
      renderChart();
    });

    el.speedButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        el.speedButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.playback.speedMultiplier = parseFloat(btn.dataset.speed);
      });
    });
  }

  function togglePlayPause() {
    if (state.playback.isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }

  function startPlayback() {
    if (!state.tripData || !state.tripData.points.length) return;
    state.playback.isPlaying = true;
    el.btnPlayPause.textContent = "Pause";
    el.btnPlayPause.style.background = "#37534D";
    state.playback.lastTimestamp = performance.now();
    state.playback.accumulatedTime = 0;
    state.playback.rafId = requestAnimationFrame(playbackTick);
  }

  function pausePlayback() {
    state.playback.isPlaying = false;
    el.btnPlayPause.textContent = "Play";
    el.btnPlayPause.style.background = "var(--bms-panel)";
    if (state.playback.rafId) {
      cancelAnimationFrame(state.playback.rafId);
      state.playback.rafId = null;
    }
  }

  function resetPlayback() {
    pausePlayback();
    state.playback.currentIndex = 0;
    el.timelineScrubber.value = 0;
    if (state.tripData && state.tripData.points.length) {
      updateTelemetryDisplay(state.tripData.points[0]);
    }
    renderChart();
  }

  function playbackTick(now) {
    if (!state.playback.isPlaying) return;

    const dt = (now - state.playback.lastTimestamp) / 1000.0;
    state.playback.lastTimestamp = now;

    // Time increment per step is sample_step_sec (e.g. 0.5s)
    const stepDuration = state.tripData.sample_step_sec || 0.5;
    state.playback.accumulatedTime += dt * state.playback.speedMultiplier;

    if (state.playback.accumulatedTime >= stepDuration) {
      const stepsToAdvance = Math.floor(state.playback.accumulatedTime / stepDuration);
      state.playback.accumulatedTime %= stepDuration;

      state.playback.currentIndex += stepsToAdvance;

      if (state.playback.currentIndex >= state.tripData.points.length) {
        state.playback.currentIndex = state.tripData.points.length - 1;
        pausePlayback();
      }

      el.timelineScrubber.value = state.playback.currentIndex;
      updateTelemetryDisplay(state.tripData.points[state.playback.currentIndex]);
      renderChart();
    }

    if (state.playback.isPlaying) {
      state.playback.rafId = requestAnimationFrame(playbackTick);
    }
  }

  // -------------------------------------------------------------
  // Telemetry Dashboard Display Update
  // -------------------------------------------------------------
  function updateTelemetryDisplay(pt) {
    if (!pt) return;

    // Choose which prediction to highlight in the hero card
    let activeSoc = pt.rf_soc;
    if (state.activePlotModel === "dt") activeSoc = pt.dt_soc;
    else if (state.activePlotModel === "knn") activeSoc = pt.knn_soc;

    const actualSoc = pt.act_soc;
    const residual = activeSoc - actualSoc;

    // Estimated SOC
    if (el.dispTimestampBadge) {
      const isStart = Math.floor(pt.t) === 0;
      el.dispTimestampBadge.textContent = isStart ? "t = 00:00 (Start)" : `t = ${formatTime(Math.floor(pt.t))}`;
    }
    el.dispSoc.textContent = `${activeSoc.toFixed(1)}%`;
    el.socBarFill.style.width = `${Math.max(0, Math.min(100, activeSoc))}%`;
    const sign = residual >= 0 ? "+" : "";
    el.dispSocSub.textContent = `Actual: ${actualSoc.toFixed(1)}% • Residual: ${sign}${residual.toFixed(2)}%`;

    // Voltage & Power
    el.dispVoltage.textContent = `${pt.volt.toFixed(1)} V`;
    const powerKw = (pt.volt * pt.curr) / 1000.0;
    el.dispPower.textContent = `Pack Power: ${Math.abs(powerKw).toFixed(2)} kW`;

    // Current & Regeneration state (Sign convention: negative = discharge, positive = regen)
    el.dispCurrent.textContent = `${pt.curr.toFixed(1)} A`;
    if (pt.curr < -15.0) {
      el.dispCurrentState.textContent = "Acceleration Discharge";
      el.dispCurrentState.style.background = "rgba(180, 110, 50, 0.4)";
      el.dispCurrentState.style.color = "var(--bms-canvas)";
    } else if (pt.curr > 5.0) {
      el.dispCurrentState.textContent = "Regenerative Braking";
      el.dispCurrentState.style.background = "rgba(167, 193, 168, 0.4)";
      el.dispCurrentState.style.color = "var(--bms-canvas)";
    } else {
      el.dispCurrentState.textContent = "Nominal Load";
      el.dispCurrentState.style.background = "rgba(167, 193, 168, 0.2)";
      el.dispCurrentState.style.color = "var(--bms-accent)";
    }

    // Velocity
    el.dispSpeed.textContent = `Vehicle Speed: ${pt.v_kmh.toFixed(1)} km/h`;

    // Thermal & Range
    el.dispTemp.textContent = `${pt.b_temp.toFixed(1)} / ${pt.a_temp.toFixed(1)} °C`;
    const estRange = (activeSoc / 100.0) * 150.0;
    el.dispRange.textContent = `Estimated Range: ${estRange.toFixed(1)} km`;

    // Time Scrubber Text
    const currentSeconds = Math.floor(pt.t);
    const totalSeconds = Math.floor(state.tripData.points[state.tripData.points.length - 1].t);
    el.dispCurrentTime.textContent = `${formatTime(currentSeconds)} / ${formatTime(totalSeconds)}`;

    // Instant Error
    el.statInstantErr.textContent = `${Math.abs(residual).toFixed(2)}%`;
  }

  function formatTime(totalSec) {
    const m = Math.floor(totalSec / 60);
    const s = Math.floor(totalSec % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }

  // -------------------------------------------------------------
  // Canvas 2D Responsive Time-Series Chart
  // -------------------------------------------------------------
  let ctx = null;

  function setupCanvasDPI() {
    const canvas = el.canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
  }

  function renderChart() {
    if (!ctx || !state.tripData || !state.tripData.points.length) return;

    const canvas = el.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;

    const padLeft = 45;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Background fill
    ctx.fillStyle = "#E3E5D4";
    ctx.fillRect(0, 0, width, height);

    // Dynamic Y-range (auto-scaled around SOC values or clamped 0-100)
    const points = state.tripData.points;
    let minSoc = 100;
    let maxSoc = 0;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      minSoc = Math.min(minSoc, p.act_soc, p.rf_soc);
      maxSoc = Math.max(maxSoc, p.act_soc, p.rf_soc);
    }
    // Pad Y-axis bounds
    minSoc = Math.max(0, Math.floor(minSoc - 5));
    maxSoc = Math.min(100, Math.ceil(maxSoc + 5));
    if (maxSoc - minSoc < 10) {
      minSoc = Math.max(0, minSoc - 5);
      maxSoc = Math.min(100, maxSoc + 5);
    }

    // Grid lines & Y-axis labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(129, 154, 145, 0.4)";
    ctx.fillStyle = "#496860";
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.textAlign = "right";

    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const val = minSoc + ((maxSoc - minSoc) / ySteps) * i;
      const y = padTop + plotH - (i / ySteps) * plotH;

      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      ctx.fillText(`${Math.round(val)}%`, padLeft - 6, y + 3);
    }

    // X-axis time labels
    ctx.textAlign = "center";
    const xSteps = 6;
    const totalTime = points[points.length - 1].t;
    for (let i = 0; i <= xSteps; i++) {
      const t = (totalTime / xSteps) * i;
      const x = padLeft + (i / xSteps) * plotW;

      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 5);
      ctx.stroke();

      ctx.fillText(formatTime(Math.round(t)), x, padTop + plotH + 18);
    }

    // Coordinate mapping helper
    function getX(index) {
      return padLeft + (index / (points.length - 1)) * plotW;
    }
    function getY(socVal) {
      return padTop + plotH - ((socVal - minSoc) / (maxSoc - minSoc)) * plotH;
    }

    // 1. Draw DT Curve (if active or 'all')
    if (state.activePlotModel === "dt" || state.activePlotModel === "all") {
      ctx.beginPath();
      ctx.strokeStyle = "#966B24";
      ctx.lineWidth = 1.5;
      for (let i = 0; i < points.length; i++) {
        const x = getX(i);
        const y = getY(points[i].dt_soc);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 2. Draw KNN Curve (if active or 'all')
    if (state.activePlotModel === "knn" || state.activePlotModel === "all") {
      ctx.beginPath();
      ctx.strokeStyle = "#486E88";
      ctx.lineWidth = 1.5;
      for (let i = 0; i < points.length; i++) {
        const x = getX(i);
        const y = getY(points[i].knn_soc);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 3. Draw Actual Measured SOC (Ground Truth)
    ctx.beginPath();
    ctx.strokeStyle = "#1F3833";
    ctx.lineWidth = 2.5;
    for (let i = 0; i < points.length; i++) {
      const x = getX(i);
      const y = getY(points[i].act_soc);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 4. Draw RF Predicted SOC (Champion)
    if (state.activePlotModel === "rf" || state.activePlotModel === "all") {
      ctx.beginPath();
      ctx.strokeStyle = "#417367";
      ctx.lineWidth = 2.5;
      for (let i = 0; i < points.length; i++) {
        const x = getX(i);
        const y = getY(points[i].rf_soc);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 5. Draw Playhead Scrubber Line
    const curIdx = state.playback.currentIndex;
    const curX = getX(curIdx);
    ctx.strokeStyle = "#C44536";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(curX, padTop);
    ctx.lineTo(curX, padTop + plotH);
    ctx.stroke();

    // Playhead Marker Dot
    const curY = getY(points[curIdx].act_soc);
    ctx.fillStyle = "#C44536";
    ctx.beginPath();
    ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#FFF";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // -------------------------------------------------------------
  // Hardware "What-If" Sandbox Simulation
  // -------------------------------------------------------------
  function setupSandboxControls() {
    const inputs = [el.sliderVolt, el.sliderCurr, el.sliderBattTemp, el.sliderAmbTemp];
    inputs.forEach((slider) => {
      slider.addEventListener("input", onSandboxInputChange);
    });

    el.btnResetSandbox.addEventListener("click", () => {
      el.sliderVolt.value = 351.4;
      el.sliderCurr.value = -14.8;
      el.sliderBattTemp.value = 27.3;
      el.sliderAmbTemp.value = 22.0;
      onSandboxInputChange();
    });
  }

  function onSandboxInputChange() {
    state.sandbox.voltage = parseFloat(el.sliderVolt.value);
    state.sandbox.current = parseFloat(el.sliderCurr.value);
    state.sandbox.batteryTemp = parseFloat(el.sliderBattTemp.value);
    state.sandbox.ambientTemp = parseFloat(el.sliderAmbTemp.value);

    el.valSliderVolt.textContent = `${state.sandbox.voltage.toFixed(1)} V`;
    el.valSliderCurr.textContent = `${state.sandbox.current.toFixed(1)} A`;
    el.valSliderBattTemp.textContent = `${state.sandbox.batteryTemp.toFixed(1)} °C`;
    el.valSliderAmbTemp.textContent = `${state.sandbox.ambientTemp.toFixed(1)} °C`;

    // Dynamic electrical operating state description
    const pKw = (state.sandbox.voltage * state.sandbox.current) / 1000.0;
    if (state.sandbox.current > 0.5) {
      el.sandboxStateDesc.textContent = `Regenerative braking active • Battery recovering kinetic energy at ${Math.abs(pKw).toFixed(1)} kW.`;
    } else if (state.sandbox.current < -15.0) {
      el.sandboxStateDesc.textContent = `Acceleration discharge • Delivering ${Math.abs(pKw).toFixed(1)} kW to drivetrain.`;
    } else {
      el.sandboxStateDesc.textContent = `Cruising / Low load • Nominal draw ${Math.abs(pKw).toFixed(1)} kW.`;
    }

    clearTimeout(state.sandbox.debounceTimer);
    state.sandbox.debounceTimer = setTimeout(runSandboxInference, 60);
  }

  async function runSandboxInference() {
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          voltage: state.sandbox.voltage,
          current: state.sandbox.current,
          battery_temp: state.sandbox.batteryTemp,
          ambient_temp: state.sandbox.ambientTemp
        })
      });

      const data = await res.json();
      if (!data.success) return;

      const p = data.predictions;
      const t = data.telemetry;

      el.sandboxRfScore.textContent = `${p.rf.toFixed(2)}%`;
      el.sandboxDtScore.textContent = `${p.dt.toFixed(2)}%`;
      el.sandboxKnnScore.textContent = `${p.knn.toFixed(2)}%`;

      // If in sandbox tab, also update the top telemetry cards
      if (state.activeTab === "panelSandbox") {
        if (el.dispTimestampBadge) {
          el.dispTimestampBadge.textContent = "Manual Slider Bench";
        }
        el.dispSoc.textContent = `${p.rf.toFixed(1)}%`;
        el.socBarFill.style.width = `${p.rf.toFixed(1)}%`;
        el.dispSocSub.textContent = `RF: ${p.rf.toFixed(1)}% | DT: ${p.dt.toFixed(1)}% | KNN: ${p.knn.toFixed(1)}%`;

        el.dispVoltage.textContent = `${state.sandbox.voltage.toFixed(1)} V`;
        el.dispPower.textContent = `Pack Power: ${Math.abs(t.power_kw).toFixed(2)} kW`;
        el.dispCurrent.textContent = `${state.sandbox.current.toFixed(1)} A`;
        el.dispCurrentState.textContent = t.is_regenerating ? "Regeneration" : "Discharge";
        el.dispSpeed.textContent = "Vehicle Speed: Static Bench";
        el.dispTemp.textContent = `${state.sandbox.batteryTemp.toFixed(1)} / ${state.sandbox.ambientTemp.toFixed(1)} °C`;
        el.dispRange.textContent = `Estimated Range: ${t.estimated_range_km.toFixed(1)} km`;
      }
    } catch (err) {
      console.error("Sandbox inference failed:", err);
    }
  }

  // -------------------------------------------------------------
  // Start on DOM Ready
  // -------------------------------------------------------------
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
