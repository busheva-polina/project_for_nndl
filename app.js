import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.trainingData = null;
        this.isTraining = false;
        
        this.initializeEventListeners();
        this.setupCharts();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('stopTraining').addEventListener('click', () => this.stopTraining());
    }

    setupCharts() {
        this.lossChart = this.createChart('lossChart', 'Training vs Validation Loss', ['Loss', 'Validation Loss']);
        this.predictionChart = this.createChart('predictionChart', 'WTI Price Prediction', ['Actual', 'Predicted']);
    }

    createChart(canvasId, title, labels) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: labels.map((label, index) => ({
                    label: label,
                    borderColor: index === 0 ? '#0066cc' : '#00cc66',
                    backgroundColor: index === 0 ? 'rgba(0, 102, 204, 0.1)' : 'rgba(0, 204, 102, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    data: []
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: '#333' },
                        ticks: { color: '#fff' }
                    },
                    y: {
                        grid: { color: '#333' },
                        ticks: { color: '#fff' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    },
                    title: {
                        display: true,
                        text: title,
                        color: '#fff'
                    }
                }
            }
        });
    }

    async loadData() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.updateStatus('fileStatus', 'Please select a CSV file', 'error');
            return;
        }

        try {
            this.updateStatus('fileStatus', 'Loading CSV file...', 'info');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('fileStatus', 'CSV file loaded successfully!', 'success');
        } catch (error) {
            this.updateStatus('fileStatus', `Error loading CSV: ${error.message}`, 'error');
            console.error('Load error:', error);
        }
    }

    async trainModel() {
        if (this.isTraining) {
            this.updateStatus('trainingStatus', 'Training already in progress', 'error');
            return;
        }

        if (!this.dataLoader.data) {
            this.updateStatus('trainingStatus', 'Please load CSV data first', 'error');
            return;
        }

        try {
            this.isTraining = true;
            document.getElementById('trainModel').disabled = true;
            document.getElementById('stopTraining').disabled = false;
            
            const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            const batchSize = parseInt(document.getElementById('batchSize').value);

            this.updateStatus('trainingStatus', 'Preparing data...', 'info');
            
            // Dispose previous tensors if they exist
            if (this.trainingData) {
                Object.values(this.trainingData).forEach(tensor => {
                    if (tensor && tensor.dispose) tensor.dispose();
                });
            }

            this.trainingData = this.dataLoader.prepareData(sequenceLength);
            
            this.updateStatus('trainingStatus', 'Building model...', 'info');
            this.model.buildModel(sequenceLength, this.trainingData.featureNames.length);

            this.updateStatus('trainingStatus', 'Starting training...', 'info');
            
            const history = await this.model.trainModel(
                this.trainingData.X_train,
                this.trainingData.y_train,
                this.trainingData.X_test,
                this.trainingData.y_test,
                epochs,
                batchSize,
                {
                    onEpochEnd: (epoch, logs, history) => {
                        const progress = ((epoch + 1) / epochs) * 100;
                        document.getElementById('trainingProgress').style.width = `${progress}%`;
                        
                        this.updateStatus('trainingStatus', 
                            `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}`, 
                            'info'
                        );
                        
                        this.updateLossChart(history);
                    },
                    onBatchEnd: (batch, batchCount, logs) => {
                        // Optional: Update progress more frequently
                    }
                }
            );

            await this.makePredictions();
            this.updateStatus('trainingStatus', 'Training completed successfully!', 'success');

        } catch (error) {
            this.updateStatus('trainingStatus', `Training error: ${error.message}`, 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainModel').disabled = false;
            document.getElementById('stopTraining').disabled = true;
            document.getElementById('trainingProgress').style.width = '0%';
        }
    }

    updateLossChart(history) {
        this.lossChart.data.labels = history.epochs;
        this.lossChart.data.datasets[0].data = history.loss;
        this.lossChart.data.datasets[1].data = history.val_loss;
        this.lossChart.update();
    }

    async makePredictions() {
        try {
            const predictions = await this.model.predict(this.trainingData.X_test);
            const predictionsData = await predictions.data();
            const actualData = await this.trainingData.y_test.data();
            
            predictions.dispose();

            // Update prediction chart
            this.predictionChart.data.labels = Array.from({length: actualData.length}, (_, i) => i + 1);
            this.predictionChart.data.datasets[0].data = Array.from(actualData);
            this.predictionChart.data.datasets[1].data = Array.from(predictionsData);
            this.predictionChart.update();

            // Calculate and display RMSE
            const squaredErrors = actualData.map((actual, i) => Math.pow(actual - predictionsData[i], 2));
            const mse = squaredErrors.reduce((sum, error) => sum + error, 0) / squaredErrors.length;
            const rmse = Math.sqrt(mse);

            this.updateStatus('resultsStatus', 
                `Prediction completed - RMSE: ${rmse.toFixed(4)}`, 
                'success'
            );

        } catch (error) {
            this.updateStatus('resultsStatus', `Prediction error: ${error.message}`, 'error');
            console.error('Prediction error:', error);
        }
    }

    stopTraining() {
        if (this.isTraining) {
            this.model.stopTraining();
            this.updateStatus('trainingStatus', 'Training stopped by user', 'info');
        }
    }

    updateStatus(elementId, message, type) {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = `status ${type}`;
    }

    dispose() {
        this.model.dispose();
        this.dataLoader.dispose();
        if (this.trainingData) {
            Object.values(this.trainingData).forEach(tensor => {
                if (tensor && tensor.dispose) tensor.dispose();
            });
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
