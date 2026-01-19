let currentTab = 'clothes';
let trendChart = null;
let clothesItems = [];
let medicineItems = [];
let areas = [];

// Data configuration
const DATA_CONFIG = {
    clothes: {
        items: ['shirts', 'pants', 'jackets', 'sarees', 'dresses', 'coats'],
        icons: {
            shirts: '👔',
            pants: '👖', 
            jackets: '🧥',
            sarees: '🥻',
            dresses: '👗',
            coats: '🧥'
        }
    },
    medicine: {
        items: ['antibiotics', 'painkillers', 'antacids', 'vitamins', 'antihistamines', 'insulin'],
        icons: {
            antibiotics: '💊',
            painkillers: '🩹',
            antacids: '🥛',
            vitamins: '🌟',
            antihistamines: '🤧',
            insulin: '💉'
        }
    },
    areas: [
        'Dhaka North', 'Dhaka South', 'Gazipur', 'Chittagong City',
        'Cox\'s Bazar', 'Khulna City', 'Rajshahi City', 'Sylhet City',
        'Rangpur City', 'Barisal City'
    ]
};

// Initialize dropdowns when page loads
document.addEventListener('DOMContentLoaded', function() {
    populateDropdowns();
});

function populateDropdowns() {
    // Populate clothes items
    const clothesSelect = document.getElementById('clothes-item');
    clothesSelect.innerHTML = '';
    DATA_CONFIG.clothes.items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = `${DATA_CONFIG.clothes.icons[item]} ${item.charAt(0).toUpperCase() + item.slice(1)}`;
        clothesSelect.appendChild(option);
    });
    
    // Populate medicine items
    const medicineSelect = document.getElementById('medicine-item');
    medicineSelect.innerHTML = '';
    DATA_CONFIG.medicine.items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = `${DATA_CONFIG.medicine.icons[item]} ${item.charAt(0).toUpperCase() + item.slice(1)}`;
        medicineSelect.appendChild(option);
    });
    
    // Populate areas for both dropdowns
    ['clothes-area', 'medicine-area'].forEach(selectId => {
        const select = document.getElementById(selectId);
        select.innerHTML = '';
        DATA_CONFIG.areas.forEach(area => {
            const option = document.createElement('option');
            option.value = area;
            option.textContent = area;
            select.appendChild(option);
        });
    });
}

function switchTab(tab) {
    currentTab = tab;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    // Show/hide content
    document.getElementById('clothes-content').style.display = tab === 'clothes' ? 'block' : 'none';
    document.getElementById('medicine-content').style.display = tab === 'medicine' ? 'block' : 'none';
    
    // Hide results
    hideResults();
}

function hideResults() {
    document.getElementById('result-card').classList.remove('show');
    document.getElementById('chart-container').classList.remove('show');
    document.getElementById('error-message').style.display = 'none';
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
    hideResults();
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    hideLoading();
}

function predictDemand(type) {
    showLoading();
    
    const year = document.getElementById(`${type}-year`).value;
    const item = document.getElementById(`${type}-item`).value;
    const area = document.getElementById(`${type}-area`).value;
    
    // Simulate API call with setTimeout
    setTimeout(() => {
        try {
            const result = simulatePrediction(type, year, item, area);
            displayPredictionResult(result, type);
            hideLoading();
        } catch (error) {
            showError('Error making prediction. Please try again.');
        }
    }, 1500);
}

function showTrend(type) {
    showLoading();
    
    const item = document.getElementById(`${type}-item`).value;
    const area = document.getElementById(`${type}-area`).value;
    
    // Simulate API call with setTimeout
    setTimeout(() => {
        try {
            const trendData = simulateTrendData(type, item, area);
            displayTrendChart(trendData);
            hideLoading();
        } catch (error) {
            showError('Error generating trend chart. Please try again.');
        }
    }, 2000);
}

function simulatePrediction(type, year, item, area) {
    // Simulate prediction logic
    const baseValue = Math.random() * 50 + 20;
    const yearFactor = (parseInt(year) - 2020) * 0.1;
    const predicted_demand = Math.round((baseValue + yearFactor) * 10) / 10;
    
    return {
        item: item,
        area: area,
        year: parseInt(year),
        predicted_demand: predicted_demand,
        population: Math.floor(Math.random() * 500000 + 200000),
        density: Math.floor(Math.random() * 3000 + 1000)
    };
}

function simulateTrendData(type, item, area) {
    const years = [];
    const actual = [];
    const predicted = [];
    
    for (let year = 2010; year <= 2025; year++) {
        years.push(year);
        const baseActual = Math.random() * 40 + 15;
        const basePredicted = baseActual + (Math.random() - 0.5) * 10;
        actual.push(Math.round(baseActual * 10) / 10);
        predicted.push(Math.round(basePredicted * 10) / 10);
    }
    
    return {
        years: years,
        actual: actual,
        predicted: predicted,
        item: item,
        area: area
    };
}

function displayPredictionResult(result, type) {
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultGrid = document.getElementById('result-grid');
    const accuracyGrid = document.getElementById('accuracy-grid');
    
    // Set icon and title
    resultIcon.textContent = type === 'clothes' ? '👕' : '💊';
    resultTitle.textContent = `${type === 'clothes' ? 'Clothes' : 'Medicine'} Prediction Result`;
    
    // Populate result grid
    resultGrid.innerHTML = `
        <div class="result-item">
            <div class="label">${type === 'clothes' ? 'Item' : 'Medicine'}</div>
            <div class="value">${result.item.charAt(0).toUpperCase() + result.item.slice(1)}</div>
        </div>
        <div class="result-item">
            <div class="label">Area</div>
            <div class="value">${result.area}</div>
        </div>
        <div class="result-item">
            <div class="label">Year</div>
            <div class="value">${result.year}</div>
        </div>
        <div class="result-item">
            <div class="label">Predicted Demand</div>
            <div class="value">${result.predicted_demand} units</div>
        </div>
        <div class="result-item">
            <div class="label">Population</div>
            <div class="value">${result.population.toLocaleString()}</div>
        </div>
        <div class="result-item">
            <div class="label">Density</div>
            <div class="value">${result.density.toLocaleString()}/km²</div>
        </div>
    `;
    
    // Populate accuracy grid with dynamic MAE and RMSE
    const accuracy = type === 'clothes' ? {mae: 15.2, rmse: 18.7} : {mae: 12.4, rmse: 16.1}; // These would come from actual model calculations
    accuracyGrid.innerHTML = `
        <div class="accuracy-item">
            <div class="label">Model Performance</div>
            <div class="value">Average MAE: ${accuracy.mae} units</div>
        </div>
        <div class="accuracy-item">
            <div class="label"></div>
            <div class="value">Average RMSE: ${accuracy.rmse} units</div>
        </div>
    `;
    
    resultCard.classList.add('show');
}

function displayTrendChart(data) {
    const chartContainer = document.getElementById('chart-container');
    const chartTitle = document.getElementById('chart-title');
    const ctx = document.getElementById('trendChart').getContext('2d');
    
    chartTitle.textContent = `${data.item.charAt(0).toUpperCase() + data.item.slice(1)} Demand: Actual vs Predicted in ${data.area} (2010-2025)`;
    
    // Destroy existing chart
    if (trendChart) {
        trendChart.destroy();
    }
    
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.years,
            datasets: [{
                label: 'Actual',
                data: data.actual,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7,
                tension: 0.1
            }, {
                label: 'Predicted',
                data: data.predicted,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointStyle: 'rect',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Demand (units)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
    
    chartContainer.classList.add('show');
}