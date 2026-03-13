function animateBlocks() {
    if (typeof gsap === "undefined") {
        return;
    }

    gsap.to(".reveal-up", {
        opacity: 1,
        y: 0,
        duration: 0.8,
        stagger: 0.12,
        ease: "power3.out"
    });

    gsap.to(".reveal-scale", {
        opacity: 1,
        scale: 1,
        duration: 0.8,
        stagger: 0.12,
        ease: "power3.out"
    });
}

function presetRevealState() {
    document.querySelectorAll(".reveal-up").forEach((item) => {
        item.style.transform = "translateY(24px)";
    });
    document.querySelectorAll(".reveal-scale").forEach((item) => {
        item.style.transform = "scale(0.96)";
    });
}

function initUploadPreview() {
    const input = document.getElementById("image");
    const preview = document.getElementById("imagePreview");
    const placeholder = document.getElementById("previewPlaceholder");
    const form = document.getElementById("uploadForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    if (!input || !preview || !placeholder) {
        return;
    }

    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) {
            preview.classList.add("hidden");
            placeholder.classList.remove("hidden");
            return;
        }

        const reader = new FileReader();
        reader.onload = (loadEvent) => {
            preview.src = loadEvent.target.result;
            preview.classList.remove("hidden");
            placeholder.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    });

    if (form && loadingOverlay) {
        form.addEventListener("submit", () => {
            loadingOverlay.classList.remove("hidden");
        });
    }
}

function buildBarChart(canvasId, labels, values, chartLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") {
        return;
    }

    new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: chartLabel,
                data: values,
                borderRadius: 12,
                backgroundColor: ["#2c8f5b", "#c46a2f"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function initResultChart() {
    if (!window.resultData) {
        return;
    }
    const labels = Object.keys(window.resultData.probabilities);
    const values = Object.values(window.resultData.probabilities);
    buildBarChart("resultProbabilityChart", labels, values, "Probability %");
}

function initDashboardCharts() {
    if (!window.dashboardData) {
        return;
    }

    const labels = ["Real Image", "AI Generated Image"];
    const values = [window.dashboardData.real_count || 0, window.dashboardData.fake_count || 0];
    buildBarChart("dashboardDistributionChart", labels, values, "Predictions");
    buildBarChart("landingDistributionChart", labels, values, "Predictions");
}

document.addEventListener("DOMContentLoaded", () => {
    presetRevealState();
    animateBlocks();
    initUploadPreview();
    initResultChart();
    initDashboardCharts();
});
