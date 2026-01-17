// Globális változók
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadLabel = document.getElementById('upload-label');
const imagePreview = document.getElementById('image-preview');
const predictButton = document.getElementById('predict-button');
const resultsArea = document.getElementById('results-area');
const maskImage = document.getElementById('mask-image');
const chartContainer = document.getElementById('stats-chart');
let statsChart;

// -- 1. Fájlfeltöltés Kezelése --
fileInput.addEventListener('change', handleFile);
uploadArea.addEventListener('dragover', (e) => e.preventDefault());
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileInput.files = e.dataTransfer.files;
    handleFile();
});

function handleFile() {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadLabel.style.display = 'none';
            predictButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

// -- 2. API Hívás (Predikció) Kezelése --
predictButton.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) return;

    predictButton.disabled = true;
    predictButton.textContent = 'Analyzing... 🧠'; // <-- Angol
    resultsArea.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert(`An error occurred during analysis: ${error.message}`); // <-- Angol
    })
    .finally(() => {
        predictButton.disabled = false;
        predictButton.textContent = 'Start Analysis'; // <-- Angol
    });
});

// -- 3. Eredmények Megjelenítése --
function displayResults(data) {
    const ctx = chartContainer.getContext('2d');
    const labels = Object.keys(data.statistics);
    const values = Object.values(data.statistics);

    if (statsChart) {
        statsChart.destroy();
    }

    statsChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: 'Distribution',
                data: values,
                backgroundColor: [
                    'rgba(0, 0, 255, 0.7)',
                    'rgba(255, 0, 0, 0.7)',
                    'rgba(0, 255, 0, 0.7)',
                    'rgba(255, 255, 0, 0.7)'
                ],
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'white'
                    }
                }
            }
        }
    });

    maskImage.src = data.mask_image;
    resultsArea.style.display = 'grid';
}