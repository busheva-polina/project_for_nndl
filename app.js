class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.lossChart = null;
        this.predictionChart = null;
        this.isTraining = false;
        
        this.initializeEventListeners();
        this.initializeCharts();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('predict').addEventListener('click', () => this.makePredictions());
        document.getElementById('saveModel').addEventListener('click', () => this.saveModel());
        document.getElementById('loadModel').addEventListener('click', () => this.loadModel());
    }

    initializeCharts() {
        // Loss chart
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
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                }
            }
        });

        // Prediction chart
        const predictionCtx = document.getElementById('predictionChart').getContext('2d');
        this.predictionChart = new Chart(predictionCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Actual WTI Prices',
                        borderColor: 'rgb(54, 162, 235)',
                        data: [],
                        fill: false
                    },
                    {
                        label: 'Predicted WTI Prices',
                        borderColor: 'rgb(255, 99, 132)',
                        data: [],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'WTI Price'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    async loadData() {
        const fileInput = document.getElementById('csvFile');
        if (!fileInput.files.length) {
            alert('Please select a CSV file first');
            return;
        }

        try {
            this.showMessage('Loading CSV data...');
            await this.dataLoader.loadCSV(fileInput.files[0]);
            this.dataLoader.prepareData();
            this.showMessage('Data loaded and prepared successfully!');
        } catch (error) {
            this.showMessage(`Error loading data: ${error.message}`, true);
            console.error(error);
        }
    }

    async trainModel() {
        if (!this.dataLoader.X_train) {
            alert('Please load data first');
            return;
        }

        if (this.isTraining) {
            alert('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showMessage('Building and training model...');
            
            if (!this.model.model) {
                this.model.buildModel();
            }

            const progressBar = document.getElementById('trainingProgress');
            const progressText = document.getElementById('progressText');

            await this.model.train(
                this.dataLoader.X_train, 
                this.dataLoader.y_train, 
                this.dataLoader.X_test, 
                this.dataLoader.y_test,
                50, // epochs
                32,  // batchSize
                (progress, epoch, logs) => {
                    progressBar.value = progress;
                    progressText.textContent = `${Math.round(progress)}%`;
                    this.showMessage(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
                    this.updateLossChart();
                }
            );

            this.showMessage('Training completed!');
        } catch (error) {
            this.showMessage(`Training error: ${error.message}`, true);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    async makePredictions() {
        if (!this.model.model || !this.dataLoader.X_test) {
            alert('Please train the model first');
            return;
        }

        try {
            this.showMessage('Making predictions...');
            
            const predictions = await this.model.predict(this.dataLoader.X_test);
            const predData = await predictions.data();
            const actualData = await this.dataLoader.y_test.data();

            // Denormalize predictions
            const denormPredictions = this.dataLoader.denormalize(Array.from(predData));
            const denormActual = this.dataLoader.denormalize(Array.from(actualData));

            this.updatePredictionChart(denormActual, denormPredictions);
            this.showMessage('Predictions completed!');

            // Calculate RMSE
            const rmse = this.calculateRMSE(denormActual, denormPredictions);
            this.showMessage(`RMSE: ${rmse.toFixed(2)}`);

            predictions.dispose();
        } catch (error) {
            this.showMessage(`Prediction error: ${error.message}`, true);
            console.error(error);
        }
    }

    calculateRMSE(actual, predicted) {
        let sum = 0;
        for (let i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - predicted[i], 2);
        }
        return Math.sqrt(sum / actual.length);
    }

    updateLossChart() {
        const history = this.model.getTrainingHistory();
        const epochs = history.loss.map((_, i) => i + 1);
        
        this.lossChart.data.labels = epochs;
        this.lossChart.data.datasets[0].data = history.loss;
        this.lossChart.data.datasets[1].data = history.valLoss;
        this.lossChart.update();
    }

    updatePredictionChart(actual, predicted) {
        const timePoints = actual.map((_, i) => i + 1);
        
        this.predictionChart.data.labels = timePoints;
        this.predictionChart.data.datasets[0].data = actual;
        this.predictionChart.data.datasets[1].data = predicted;
        this.predictionChart.update();
    }

    async saveModel() {
        try {
            await this.model.saveModel();
            this.showMessage('Model saved successfully!');
        } catch (error) {
            this.showMessage(`Error saving model: ${error.message}`, true);
            console.error(error);
        }
    }

    async loadModel() {
        try {
            const success = await this.model.loadModel();
            if (success) {
                this.showMessage('Model loaded successfully!');
            } else {
                this.showMessage('No saved model found. Please train a model first.', true);
            }
        } catch (error) {
            this.showMessage(`Error loading model: ${error.message}`, true);
            console.error(error);
        }
    }

    showMessage(message, isError = false) {
        const infoDiv = document.getElementById('trainingInfo');
        infoDiv.textContent = message;
        infoDiv.style.color = isError ? 'red' : 'green';
        console.log(message);
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        if (this.lossChart) this.lossChart.destroy();
        if (this.predictionChart) this.predictionChart.destroy();
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new StockPredictionApp();
});

// Clean up when leaving the page
window.addEventListener('beforeunload', () => {
    if (app) {
        app.dispose();
    }
    tf.disposeVariables();
});

// Export for global access if needed
window.StockPredictionApp = StockPredictionApp;
