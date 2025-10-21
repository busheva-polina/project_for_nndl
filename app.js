class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isTraining = false;
        this.normalization = null;
        this.chart = null;
        this.lossChart = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // File input handler
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        // Train button handler
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });

        // Initialize charts
        this.initializeCharts();
    }

    initializeCharts() {
        // Prediction chart
        const ctx = document.getElementById('predictionChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Actual WTI',
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'Predicted WTI',
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        display: true,
                        title: {
                            display: true,
                            text: 'Time Steps'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'WTI Price (Normalized)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });

        // Loss chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        this.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Training Loss',
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        pointRadius: 1
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 2,
                        pointRadius: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        type: 'logarithmic',
                        display: true,
                        title: {
                            display: true,
                            text: 'Loss (MSE)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateStatus('Loading CSV data...', 'info');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('CSV loaded successfully!', 'success');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading CSV: ${error.message}`, 'error');
            console.error('File loading error:', error);
        }
    }

    async startTraining() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            this.updateStatus('Preparing data...', 'info');
            document.getElementById('trainBtn').disabled = true;

            // Prepare sequences
            const { X_train, y_train, X_test, y_test, normalization, featureNames } = 
                this.dataLoader.prepareSequences();
            
            this.normalization = normalization;

            this.updateStatus('Building model...', 'info');
            
            // Build model
            this.model.buildModel([X_train.shape[1], X_train.shape[2]], featureNames);
            
            // Display feature importance
            this.displayFeatureImportance();

            // Set up training callback
            this.model.onEpochEnd = (epoch, logs) => {
                this.updateTrainingProgress(epoch, logs);
            };

            this.updateStatus('Training started...', 'info');
            
            // Train model
            await this.model.train(X_train, y_train, X_test, y_test, 40);

            this.updateStatus('Training completed!', 'success');
            
            // Make predictions
            this.makePredictions(X_test, y_test);

            // Cleanup tensors
            tf.dispose([X_train, y_train, X_test, y_test]);

        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainBtn').disabled = false;
        }
    }

    displayFeatureImportance() {
        const importance = this.model.getFeatureImportance();
        const featureList = document.getElementById('featureImportance');
        
        featureList.innerHTML = importance.map(([feature, score]) => 
            `<li>${feature}: ${score.toFixed(2)}</li>`
        ).join('');
    }

    updateTrainingProgress(epoch, logs) {
        const progress = ((epoch + 1) / 40) * 100;
        document.getElementById('trainingProgress').style.width = `${progress}%`;
        document.getElementById('trainingProgress').textContent = `${Math.round(progress)}%`;
        
        document.getElementById('currentEpoch').textContent = epoch + 1;
        document.getElementById('currentLoss').textContent = logs.loss.toFixed(6);
        document.getElementById('currentValLoss').textContent = logs.val_loss.toFixed(6);

        // Update loss chart
        const history = this.model.getTrainingHistory();
        this.updateLossChart(history);
    }

    updateLossChart(history) {
        const epochs = history.loss.map((_, i) => i + 1);
        
        this.lossChart.data.labels = epochs;
        this.lossChart.data.datasets[0].data = history.loss;
        this.lossChart.data.datasets[1].data = history.val_loss;
        this.lossChart.update();
    }

    async makePredictions(X_test, y_test) {
        try {
            const predictions = this.model.predict(X_test);
            const actualData = await y_test.data();
            const predictedData = await predictions.data();

            // Update prediction chart
            const timeSteps = Array.from({ length: actualData.length }, (_, i) => i);
            
            this.chart.data.labels = timeSteps;
            this.chart.data.datasets[0].data = actualData;
            this.chart.data.datasets[1].data = Array.from(predictedData);
            this.chart.update();

            // Calculate and display accuracy metrics
            this.displayAccuracyMetrics(actualData, Array.from(predictedData));

            // Cleanup
            predictions.dispose();
        } catch (error) {
            console.error('Prediction error:', error);
        }
    }

    displayAccuracyMetrics(actual, predicted) {
        const mse = tf.tidy(() => {
            const actualTensor = tf.tensor1d(actual);
            const predictedTensor = tf.tensor1d(predicted);
            return tf.metrics.meanSquaredError(actualTensor, predictedTensor).dataSync()[0];
        });

        const rmse = Math.sqrt(mse);
        
        document.getElementById('finalMSE').textContent = mse.toFixed(6);
        document.getElementById('finalRMSE').textContent = rmse.toFixed(6);
    }

    updateStatus(message, type) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
    }

    dispose() {
        this.model.dispose();
        this.dataLoader.dispose();
        if (this.chart) {
            this.chart.destroy();
        }
        if (this.lossChart) {
            this.lossChart.destroy();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StockPredictionApp();
});
