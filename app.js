class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.trainingData = null;
        this.isTraining = false;
        
        this.initializeUI();
    }

    initializeUI() {
        // File input handler
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Train button handler
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });

        // Initialize charts
        this.initializeCharts();
    }

    initializeCharts() {
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        this.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        borderColor: 'rgb(75, 192, 192)',
                        data: [],
                        fill: false
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: 'rgb(255, 99, 132)',
                        data: [],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Training vs Validation Loss (MSE)'
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        type: 'logarithmic'
                    }
                }
            }
        });
    }

    async handleFileUpload(file) {
        try {
            this.updateStatus('Loading CSV file...');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('CSV loaded successfully. Click "Train Model" to start training.');
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`, 'error');
        }
    }

    async startTraining() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            this.updateStatus('Preprocessing data...');
            
            // Preprocess data
            this.trainingData = this.dataLoader.preprocessData();
            
            // Build model
            const inputShape = [this.dataLoader.sequenceLength, this.trainingData.featureNames.length];
            this.model = new LSTMModel(inputShape);
            this.model.buildModel();
            
            // Set up training callbacks
            this.model.onEpochEnd = (epoch, logs) => {
                this.updateTrainingProgress(epoch, logs);
            };

            this.updateStatus('Starting model training...');
            
            // Train model
            await this.model.train(
                this.trainingData.X_train,
                this.trainingData.y_train,
                this.trainingData.X_test,
                this.trainingData.y_test,
                100,
                32
            );

            this.updateStatus('Training completed!');
            
            // Make predictions
            await this.makePredictions();
            
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, 'error');
        } finally {
            this.isTraining = false;
        }
    }

    updateTrainingProgress(epoch, logs) {
        const progress = document.getElementById('trainingProgress');
        const status = document.getElementById('trainingStatus');
        
        if (progress && status) {
            const percent = ((epoch + 1) / 100) * 100;
            progress.value = percent;
            status.textContent = `Epoch ${epoch + 1}/100 - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}`;
        }

        // Update loss chart
        this.updateLossChart();
    }

    updateLossChart() {
        if (!this.model) return;

        const history = this.model.getTrainingHistory();
        const epochs = history.loss.map((_, i) => i + 1);
        
        this.lossChart.data.labels = epochs;
        this.lossChart.data.datasets[0].data = history.loss;
        this.lossChart.data.datasets[1].data = history.val_loss;
        this.lossChart.update();
    }

    async makePredictions() {
        try {
            if (!this.model || !this.trainingData) return;

            this.updateStatus('Making predictions...');
            
            const predictions = this.model.predict(this.trainingData.X_test);
            const actual = this.trainingData.y_test;
            
            // Calculate accuracy metrics
            const predData = await predictions.data();
            const actualData = await actual.data();
            
            const mse = this.calculateMSE(predData, actualData);
            const rmse = Math.sqrt(mse);
            
            this.updateStatus(`Prediction completed - RMSE: ${rmse.toFixed(4)}`);
            
        } catch (error) {
            console.error('Prediction error:', error);
        }
    }

    calculateMSE(predictions, actual) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.pow(predictions[i] - actual[i], 2);
        }
        return sum / predictions.length;
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
        console.log(`Status: ${message}`);
    }

    dispose() {
        if (this.dataLoader) {
            this.dataLoader.dispose();
        }
        if (this.model) {
            this.model.dispose();
        }
        if (this.trainingData) {
            this.trainingData.X_train.dispose();
            this.trainingData.y_train.dispose();
            this.trainingData.X_test.dispose();
            this.trainingData.y_test.dispose();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StockPredictionApp();
});
