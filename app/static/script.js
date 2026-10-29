/* ==========================================================================
   Smart Refrigerator Temperature Anomaly Detection - Frontend JavaScript
   ========================================================================== */

let tempChart = null;
let scoreChart = null;
let autoSimInterval = null;
let isAutoSimulating = false;
let currentStepIndex = 0;

document.addEventListener("DOMContentLoaded", () => {
    // Initial Data Fetch
    initCharts();
    fetchCurrentReading();
    fetchTimeSeriesData();
    fetchModelComparison();

    // Setup Event Listeners
    document.getElementById("refreshBtn").addEventListener("click", () => {
        fetchCurrentReading();
    });

    document.getElementById("toggleAutoBtn").addEventListener("click", toggleAutoSimulation);
});

/**
 * Fetches current temperature reading from Flask REST API.
 * Updates summary cards, status badges, alert banners, and the Asia/Kolkata date/time timestamp.
 */
async function fetchCurrentReading(stepIndex = null) {
    try {
        let url = "/api/current";
        if (stepIndex !== null) {
            url += `?index=${stepIndex}`;
        }
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            console.warn("API Warning:", data.error);
            return;
        }

        currentStepIndex = data.simulation_index;

        // Update Summary Cards
        document.getElementById("currentTemp").textContent = `${data.temperature.toFixed(2)} °C`;
        document.getElementById("anomalyScore").textContent = data.anomaly_score.toFixed(6);
        document.getElementById("scoreThreshold").textContent = `Threshold: ${data.threshold.toFixed(6)}`;
        
        // Update Last Updated with current Asia/Kolkata date & time
        document.getElementById("lastUpdated").textContent = data.timestamp || data.current_timestamp;

        // Update Status Badge & Warning Banner
        const statusEl = document.getElementById("currentStatus");
        const alertBanner = document.getElementById("alertBanner");
        const alertMessage = document.getElementById("alertMessage");

        if (data.is_anomaly || data.status.toUpperCase().includes("ANOMALY")) {
            statusEl.textContent = "ANOMALY DETECTED";
            statusEl.className = "status-badge status-anomaly";
            alertBanner.classList.remove("hidden");
            alertMessage.textContent = data.alert_message || "⚠ Temperature anomaly detected. Please check the refrigerator.";
        } else {
            statusEl.textContent = "NORMAL";
            statusEl.className = "status-badge status-normal";
            alertBanner.classList.add("hidden");
        }

    } catch (err) {
        console.error("Failed to fetch current reading:", err);
    }
}

/**
 * Fetches time-series data and updates Chart.js graphs & recent logs table.
 */
async function fetchTimeSeriesData() {
    try {
        const response = await fetch("/api/temperature-data?limit=300");
        const data = await response.json();

        if (data.error) return;

        updateCharts(data);
        updateRecentReadingsTable(data);
    } catch (err) {
        console.error("Failed to fetch time-series data:", err);
    }
}

/**
 * Initializes empty Chart.js graphs.
 */
function initCharts() {
    const ctxTemp = document.getElementById("tempChart").getContext("2d");
    tempChart = new Chart(ctxTemp, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Temperature (°C)",
                    data: [],
                    borderColor: "#0284c7",
                    backgroundColor: "rgba(2, 132, 199, 0.05)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: "Upper Safe Limit (5°C)",
                    data: [],
                    borderColor: "#f97316",
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: "Lower Safe Limit (2°C)",
                    data: [],
                    borderColor: "#06b6d4",
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxTicksLimit: 8, font: { size: 10 } } },
                y: { title: { display: true, text: "Temperature (°C)" } }
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });

    const ctxScore = document.getElementById("scoreChart").getContext("2d");
    scoreChart = new Chart(ctxScore, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Reconstruction Error (MSE)",
                    data: [],
                    borderColor: "#64748b",
                    borderWidth: 1.2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: "Anomaly Threshold",
                    data: [],
                    borderColor: "#ef4444",
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxTicksLimit: 8, font: { size: 10 } } },
                y: { title: { display: true, text: "Reconstruction Error (MSE)" } }
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });
}

/**
 * Updates Chart.js graphs.
 */
function updateCharts(data) {
    if (!tempChart || !scoreChart) return;

    const timestamps = data.timestamps;
    const temps = data.temperatures;
    const scores = data.anomaly_scores;
    const thresholds = data.thresholds;

    tempChart.data.labels = timestamps;
    tempChart.data.datasets[0].data = temps;
    tempChart.data.datasets[1].data = new Array(temps.length).fill(5.0);
    tempChart.data.datasets[2].data = new Array(temps.length).fill(2.0);
    tempChart.update("none");

    scoreChart.data.labels = timestamps;
    scoreChart.data.datasets[0].data = scores;
    scoreChart.data.datasets[1].data = thresholds;
    scoreChart.update("none");
}

/**
 * Populates recent temperature log table (keeping historical simulated timestamps).
 */
function updateRecentReadingsTable(data) {
    const tbody = document.getElementById("recentReadingsBody");
    tbody.innerHTML = "";

    const count = data.timestamps.length;
    const limit = Math.min(count, 12);

    for (let i = count - 1; i >= count - limit; i--) {
        const tr = document.createElement("tr");

        const statusStr = data.statuses[i] || "Normal";
        const isAnomaly = (statusStr.toUpperCase().includes("ANOMALY"));
        const statusBadge = isAnomaly 
            ? `<span class="status-badge status-anomaly">ANOMALY</span>`
            : `<span class="status-badge status-normal">NORMAL</span>`;

        tr.innerHTML = `
            <td>${data.timestamps[i]}</td>
            <td><strong>${data.temperatures[i].toFixed(2)} °C</strong></td>
            <td>${statusBadge}</td>
            <td>${data.anomaly_scores[i].toFixed(6)}</td>
        `;
        tbody.appendChild(tr);
    }
}

/**
 * Fetches actual model evaluation comparison results from /api/model-results.
 * Displays Precision, Recall, F1-Score, and Status.
 */
async function fetchModelComparison() {
    const tbody = document.getElementById("comparisonTableBody");
    try {
        const response = await fetch("/api/model-results");
        const data = await response.json();

        if (data.status === "not_available" || !data.models || data.models.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center">Model results file not available.</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        data.models.forEach(m => {
            const tr = document.createElement("tr");
            
            const precision = (typeof m.Precision === 'number') ? m.Precision.toFixed(4) : m.Precision;
            const recall = (typeof m.Recall === 'number') ? m.Recall.toFixed(4) : m.Recall;
            const f1 = (typeof m["F1-score"] === 'number') ? m["F1-score"].toFixed(4) : m["F1-score"];
            const anomalies = m["Detected Anomalies"];
            const status = m.Status || "Evaluated";

            tr.innerHTML = `
                <td><strong>${m.Model}</strong></td>
                <td>${precision}</td>
                <td>${recall}</td>
                <td>${f1}</td>
                <td>${anomalies}</td>
                <td><span class="chart-badge">${status}</span></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error("Failed to fetch model comparison:", err);
        tbody.innerHTML = `<tr><td colspan="6" class="text-center">Error loading model comparison data.</td></tr>`;
    }
}

/**
 * Toggles auto-simulation playback.
 */
function toggleAutoSimulation() {
    const btn = document.getElementById("toggleAutoBtn");
    
    if (isAutoSimulating) {
        clearInterval(autoSimInterval);
        isAutoSimulating = false;
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Start Auto Simulation`;
        btn.className = "btn btn-secondary";
    } else {
        isAutoSimulating = true;
        btn.innerHTML = `<i class="fa-solid fa-pause"></i> Pause Simulation`;
        btn.className = "btn btn-primary";
        
        autoSimInterval = setInterval(() => {
            fetchCurrentReading();
        }, 1500);
    }
}
