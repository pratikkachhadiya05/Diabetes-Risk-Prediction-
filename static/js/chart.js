// Function to create risk gauge chart
function createRiskGauge(riskPercentage) {
    const ctx = document.getElementById('riskGaugeChart');
    if (!ctx) return;

    if (window.riskGaugeChart instanceof Chart) {
        window.riskGaugeChart.destroy();
    }

    const riskLevel = riskPercentage;
    const safeLevel = 100 - riskLevel;

    window.riskGaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Diabetes Risk', 'Safe Zone'],
            datasets: [{
                data: [riskLevel, safeLevel],
                backgroundColor: [
                    riskLevel > 70 ? '#ff4757' :
                    riskLevel > 40 ? '#ffa502' :
                    '#2ed573',
                    '#dfe6e9'
                ],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        },
        plugins: [{
            id: 'centerText',
            afterDraw(chart) {
                const { ctx, width, height } = chart;

                ctx.save();
                ctx.font = 'bold 24px Arial';
                ctx.fillStyle = '#333';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                ctx.fillText(
                    riskLevel + '%',
                    width / 2,
                    height / 2
                );

                ctx.restore();
            }
        }]
    });
}

// Function to create probability comparison chart
function createProbabilityChart(probClass0, probClass1) {
    const ctx = document.getElementById('probabilityChart');
    if (!ctx) return;

    if (window.probabilityChart instanceof Chart) {
        window.probabilityChart.destroy();
    }

    window.probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['No Diabetes', 'Diabetes'],
            datasets: [{
                label: 'Probability (%)',
                data: [probClass0, probClass1],
                backgroundColor: [
                    '#51cf66',
                    '#ff6b6b'
                ],
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Function to create health metrics comparison chart
function createHealthMetricsChart(userValues, normalRanges) {
    const ctx = document.getElementById('healthMetricsChart');
    if (!ctx) return;

    if (window.healthMetricsChart instanceof Chart) {
        window.healthMetricsChart.destroy();
    }

    window.healthMetricsChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'BMI',
                'HbA1c',
                'Blood Glucose',
                'Age Impact',
                'Lifestyle'
            ],
            datasets: [
                {
                    label: 'Your Values',
                    data: userValues,
                    backgroundColor: 'rgba(102,126,234,0.2)',
                    borderColor: 'rgba(102,126,234,1)',
                    borderWidth: 2
                },
                {
                    label: 'Normal Range',
                    data: normalRanges,
                    backgroundColor: 'rgba(81,207,102,0.2)',
                    borderColor: 'rgba(81,207,102,1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Normalize values
function normalizeValue(value, min, max, type) {
    if (type === 'bmi') {
        if (value < 18.5) return 30;
        if (value < 25) return 20;
        if (value < 30) return 50;
        return 80;
    }

    if (type === 'hba1c') {
        if (value < 5.7) return 20;
        if (value < 6.5) return 50;
        return 80;
    }

    if (type === 'glucose') {
        if (value < 100) return 20;
        if (value < 126) return 50;
        return 80;
    }

    return Math.min(100, (value / max) * 100);
}

// Scroll Animation
document.addEventListener('DOMContentLoaded', function() {

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    });

    document.querySelectorAll(
        '.feature-card, .info-card, .stat-item'
    ).forEach((el) => {
        observer.observe(el);
    });

});