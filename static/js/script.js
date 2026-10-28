/**
 * script.js
 * Student Marks Prediction System — Frontend Interactions
 * - Interactive Chart.js Study Hours vs Marks Scatter + Regression Plot
 * - Asynchronous AJAX Prediction Form with Client-Side Validation
 * - Mobile Sidebar Navigation Toggle
 * - Real-time Dataset Search & Filtering
 */

document.addEventListener("DOMContentLoaded", function () {
    initSidebar();
    initChart();
    initPredictionForm();
    initDatasetSearch();
    initSmoothScroll();
});

/* --------------------------------------------------------------------------
   1. Mobile Sidebar Navigation
   -------------------------------------------------------------------------- */
function initSidebar() {
    const sidebar = document.getElementById("appSidebar");
    const overlay = document.getElementById("sidebarOverlay");
    const openBtn = document.getElementById("mobileToggleBtn");
    const closeBtn = document.getElementById("sidebarCloseBtn");

    if (!sidebar) return;

    function openSidebar() {
        sidebar.classList.add("is-open");
        if (overlay) overlay.classList.add("is-open");
        document.body.style.overflow = "hidden";
    }

    function closeSidebar() {
        sidebar.classList.remove("is-open");
        if (overlay) overlay.classList.remove("is-open");
        document.body.style.overflow = "";
    }

    if (openBtn) openBtn.addEventListener("click", openSidebar);
    if (closeBtn) closeBtn.addEventListener("click", closeSidebar);
    if (overlay) overlay.addEventListener("click", closeSidebar);
}

/* --------------------------------------------------------------------------
   2. Interactive Chart.js (Study Hours vs Marks)
   -------------------------------------------------------------------------- */
function initChart() {
    const canvas = document.getElementById("hoursVsMarksChart");
    if (!canvas) return;

    const payloadElement = document.getElementById("chart-data-payload");
    if (!payloadElement) return;

    let chartData = { scatter: [], line: {}, equation: "" };
    try {
        chartData = JSON.parse(payloadElement.textContent);
    } catch (e) {
        console.error("Failed to parse chart data payload:", e);
        return;
    }

    const scatterPoints = chartData.scatter || [];
    const regressionLine = chartData.line || {};

    // Generate line points across the domain
    let lineData = [];
    if (regressionLine.x_min !== undefined && regressionLine.x_max !== undefined) {
        lineData = [
            { x: regressionLine.x_min, y: regressionLine.y_at_x_min },
            { x: regressionLine.x_max, y: regressionLine.y_at_x_max }
        ];
    }

    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Actual Student Marks",
                    data: scatterPoints,
                    backgroundColor: "rgba(40, 120, 240, 0.75)",
                    borderColor: "#142B50",
                    borderWidth: 1.2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointHoverBackgroundColor: "#2878F0",
                    pointHoverBorderColor: "#FFFFFF",
                    order: 2
                },
                {
                    label: "Linear Regression Line",
                    data: lineData,
                    type: "line",
                    borderColor: "#10B981",
                    backgroundColor: "transparent",
                    borderWidth: 2.8,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false,
                    tension: 0,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 900,
                easing: "easeOutQuart"
            },
            interaction: {
                mode: "nearest",
                intersect: false
            },
            plugins: {
                legend: {
                    display: false // Using custom styled HTML legend below chart
                },
                tooltip: {
                    backgroundColor: "#142B50",
                    titleColor: "#FFFFFF",
                    bodyColor: "#CBD5E1",
                    padding: 10,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        title: function (items) {
                            if (!items.length) return "";
                            const item = items[0];
                            return item.datasetIndex === 0 ? "Student Record" : "Regression Estimate";
                        },
                        label: function (context) {
                            const x = context.parsed.x;
                            const y = context.parsed.y;
                            if (context.datasetIndex === 0) {
                                return `Study: ${x} hrs/day ➔ Exam: ${y} Marks`;
                            } else {
                                return `Trend: ${y.toFixed(1)} Marks at ${x} hrs`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: "linear",
                    position: "bottom",
                    title: {
                        display: true,
                        text: "Study Hours (per day)",
                        color: "#172B4D",
                        font: {
                            family: "'Plus Jakarta Sans', sans-serif",
                            weight: "600",
                            size: 12
                        },
                        padding: { top: 8 }
                    },
                    min: 0,
                    grid: {
                        color: "#EDF2F7",
                        drawBorder: false
                    },
                    ticks: {
                        color: "#64748B",
                        font: { size: 11 }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Exam Marks (out of 100)",
                        color: "#172B4D",
                        font: {
                            family: "'Plus Jakarta Sans', sans-serif",
                            weight: "600",
                            size: 12
                        },
                        padding: { bottom: 8 }
                    },
                    min: 0,
                    max: 105,
                    grid: {
                        color: "#EDF2F7",
                        drawBorder: false
                    },
                    ticks: {
                        color: "#64748B",
                        font: { size: 11 },
                        stepSize: 20
                    }
                }
            }
        }
    });
}

/* --------------------------------------------------------------------------
   3. Marks Prediction Form (AJAX & Client-Side Validation)
   -------------------------------------------------------------------------- */
function initPredictionForm() {
    const form = document.getElementById("predictionForm");
    const input = document.getElementById("studyHoursInput");
    const predictBtn = document.getElementById("predictBtn");
    const btnText = document.getElementById("btnText");
    const clientErrorMsg = document.getElementById("clientErrorMsg");
    const resultCard = document.getElementById("predictionResultCard");
    const errorCard = document.getElementById("predictionErrorCard");
    const serverErrorMsg = document.getElementById("serverErrorMsg");

    if (!form || !input) return;

    function showClientError(message) {
        if (clientErrorMsg) {
            clientErrorMsg.textContent = message;
            clientErrorMsg.style.display = "block";
        }
        input.focus();
    }

    function clearClientError() {
        if (clientErrorMsg) {
            clientErrorMsg.textContent = "";
            clientErrorMsg.style.display = "none";
        }
    }

    input.addEventListener("input", clearClientError);

    form.addEventListener("submit", async function (e) {
        e.preventDefault();
        clearClientError();

        if (errorCard) errorCard.style.display = "none";

        const rawVal = input.value.trim();

        // 1. Validation: Empty Check
        if (rawVal === "") {
            showClientError("Please enter the number of study hours.");
            return;
        }

        const hours = parseFloat(rawVal);

        // 2. Validation: Numeric Check
        if (isNaN(hours)) {
            showClientError("Please enter a valid numeric value for study hours.");
            return;
        }

        // 3. Validation: Non-negative Check
        if (hours < 0) {
            showClientError("Study hours cannot be negative. Please enter a value >= 0.");
            return;
        }

        // 4. Validation: Realistic Day Check (24 hours max)
        if (hours > 24) {
            showClientError("Study hours cannot exceed 24 hours in a single day.");
            return;
        }

        // Trigger Loading State
        predictBtn.classList.add("is-loading");
        btnText.textContent = "Calculating...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ study_hours: hours })
            });

            const data = await response.json();

            // Reset Loading State
            predictBtn.classList.remove("is-loading");
            btnText.textContent = "Predict Marks";

            if (!response.ok || !data.success) {
                // Show Server Error Card
                const msg = data.error || "Prediction request could not be processed.";
                if (serverErrorMsg) serverErrorMsg.textContent = msg;
                if (errorCard) errorCard.style.display = "flex";
                if (resultCard) resultCard.style.display = "none";
                return;
            }

            // Populate Prediction Result Card
            document.getElementById("resultMarksValue").textContent = data.predicted_marks;
            document.getElementById("resultHoursVal").textContent = `${data.study_hours} hrs/day`;
            document.getElementById("resultGradeVal").textContent = data.grade;
            document.getElementById("resultAdviceVal").textContent = data.advice;

            // Animate Progress Bar
            const bar = document.getElementById("resultProgressBar");
            if (bar) {
                bar.style.width = "0%";
                setTimeout(() => {
                    bar.style.width = `${Math.min(100, Math.max(0, data.predicted_marks))}%`;
                }, 80);
            }

            // Show Result Card
            if (resultCard) {
                resultCard.style.display = "block";
                resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }

        } catch (err) {
            predictBtn.classList.remove("is-loading");
            btnText.textContent = "Predict Marks";

            if (serverErrorMsg) serverErrorMsg.textContent = "Network error. Please check your connection and try again.";
            if (errorCard) errorCard.style.display = "flex";
            if (resultCard) resultCard.style.display = "none";
        }
    });
}

/* --------------------------------------------------------------------------
   4. Real-time Dataset Search & Filtering (Data Overview Page)
   -------------------------------------------------------------------------- */
function initDatasetSearch() {
    const searchInput = document.getElementById("datasetSearchInput");
    const table = document.getElementById("fullDatasetTable");
    const countDisplay = document.getElementById("filteredRowCount");

    if (!searchInput || !table) return;

    searchInput.addEventListener("input", function () {
        const query = searchInput.value.toLowerCase().trim();
        const rows = table.querySelectorAll("tbody tr");
        let visibleCount = 0;

        rows.forEach(function (row) {
            const text = row.textContent.toLowerCase();
            if (text.includes(query)) {
                row.style.display = "";
                visibleCount++;
            } else {
                row.style.display = "none";
            }
        });

        if (countDisplay) {
            if (query === "") {
                countDisplay.textContent = `Showing all ${rows.length} records`;
            } else {
                countDisplay.textContent = `Showing ${visibleCount} of ${rows.length} matching records`;
            }
        }
    });
}

/* --------------------------------------------------------------------------
   5. Smooth Scroll & Predict Link Focusing
   -------------------------------------------------------------------------- */
function initSmoothScroll() {
    const predictLink = document.getElementById("navPredictLink");
    if (!predictLink) return;

    predictLink.addEventListener("click", function (e) {
        const target = document.getElementById("predict-panel");
        const input = document.getElementById("studyHoursInput");
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: "smooth" });
            if (input) {
                setTimeout(() => input.focus(), 400);
            }
        }
    });
}
