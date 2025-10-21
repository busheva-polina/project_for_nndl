class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.trainingData = null;
        this.isTraining = false;
        this.currentEpoch = 0;
        
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
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointBackgroundColor: 'rgb(59, 130, 246)',
                        pointBorderColor: 'rgb(59, 130, 246)',
                        pointHoverBackgroundColor: 'rgb(59, 130, 246)',
                        pointHoverBorderColor: 'white',
                        pointHoverBorderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        data: []
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: 'rgb(239, 68, 68)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointBackgroundColor: 'rgb(239, 68, 68)',
                        pointBorderColor: 'rgb(239, 68, 68)',
                        pointHoverBackgroundColor: 'rgb(239, 68, 68)',
                        pointHoverBorderColor: 'white',
                        pointHoverBorderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        data: []
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300,
                    easing: 'easeInOutCubic'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Progress - Loss Curves',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        color: '#1f2937',
                        padding: 20
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: {
                            size: 12
                        },
                        bodyFont: {
                            size: 12
                        },
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch',
                            font: {
                                size: 12,
                                weight: 'bold'
                            },
                            color: '#6b7280'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#6b7280',
                            font: {
                                size: 11
                            }
                        }
                    },
                    y: {
                        display: true,
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Mean Squared Error (Log Scale)',
                            font: {
                                size: 12,
                                weight: 'bold'
                            },
                            color: '#6b7280'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#6b7280',
                            font: {
                                size: 11
                            },
                            callback: function(value) {
                                return value.toExponential(2);
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                elements: {
                    line: {
                        tension: 0.4
                    }
                }
            }
        });

        // Initialize prediction chart
        const predictionCtx = document.getElementById('predictionChart').getContext('2d');
        this.predictionChart = new Chart(predictionCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Actual WTI Price',
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        data: []
                    },
                    {
                        label: 'Predicted WTI Price',
                        borderColor: 'rgb(139, 92, 246)',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: true,
                        data: []
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'WTI Price Prediction vs Actual',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        color: '#1f2937',
                        padding: 20
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Normalized Price'
                        }
                    }
                }
            }
        });
    }

    async handleFileUpload(file) {
        try {
            this.updateStatus('Loading CSV file...', 'info');
            this.showLoadingSpinner(true);
            await this.dataLoader.loadCSV(file);
            this.updateStatus('CSV loaded successfully. Click "Train Model" to start training.', 'success');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`, 'error');
        } finally {
            this.showLoadingSpinner(false);
        }
    }

    showLoadingSpinner(show) {
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
    }

    async startTraining() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            this.currentEpoch = 0;
            document.getElementById('trainBtn').disabled = true;
            this.updateStatus('Preprocessing data...', 'info');
            
            // Reset charts
            this.lossChart.data.labels = [];
            this.lossChart.data.datasets[0].data = [];
            this.lossChart.data.datasets[1].data = [];
            this.lossChart.update();

            // Preprocess data
            this.trainingData = this.dataLoader.preprocessData();
            
            // Build model
            const inputShape = [this.dataLoader.sequenceLength, this.trainingData.featureNames.length];
            this.model = new LSTMModel(inputShape);
            this.model.buildModel();
            
            // Set up training callbacks
            this.model.onEpochEnd = (epoch, logs) => {
                this.currentEpoch = epoch;
                this.updateTrainingProgress(epoch, logs);
            };

            this.updateStatus('Starting model training...', 'info');
            
            // Train model
            await this.model.train(
                this.trainingData.X_train,
                this.trainingData.y_train,
                this.trainingData.X_test,
                this.trainingData.y_test,
                100,
                32
            );

            this.updateStatus('Training completed! Making predictions...', 'success');
            
            // Make predictions
            await this.makePredictions();
            
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, 'error');
        } finally {
            this.isTraining = false;
            document.getElementById('trainBtn').disabled = false;
        }
    }

    updateTrainingProgress(epoch, logs) {
        const progress = document.getElementById('trainingProgress');
        const status = document.getElementById('trainingStatus');
        const progressText = document.getElementById('progressText');
        
        if (progress && status) {
            const percent = ((epoch + 1) / 100) * 100;
            progress.value = percent;
            status.textContent = `Epoch ${epoch + 1}/100 - Loss: ${logs.loss.toFixed(6)}, Val Loss: ${logs.val_loss.toFixed(6)}`;
        }

        if (progressText) {
            progressText.textContent = `${epoch + 1}/100 epochs`;
        }

        // Update loss chart with smooth animation
        this.updateLossChart(epoch, logs);
    }

    updateLossChart(epoch, logs) {
        if (!this.model) return;

        const history = this.model.getTrainingHistory();
        
        // Add new data point
        this.lossChart.data.labels.push(epoch + 1);
        this.lossChart.data.datasets[0].data.push(logs.loss);
        this.lossChart.data.datasets[1].data.push(logs.val_loss);
        
        // Limit data points for performance (show last 50 points or all if less than 50)
        if (this.lossChart.data.labels.length > 50) {
            this.lossChart.data.labels = this.lossChart.data.labels.slice(-50);
            this.lossChart.data.datasets[0].data = this.lossChart.data.datasets[0].data.slice(-50);
            this.lossChart.data.datasets[1].data = this.lossChart.data.datasets[1].data.slice(-50);
        }
        
        this.lossChart.update('active');
    }

    async makePredictions() {
        try {
            if (!this.model || !this.trainingData) return;

            this.updateStatus('Making predictions and generating charts...', 'info');
            
            const predictions = this.model.predict(this.trainingData.X_test);
            const actual = this.trainingData.y_test;
            
            // Get data for charts
            const predData = await predictions.data();
            const actualData = await actual.data();
            
            // Update prediction chart
            this.updatePredictionChart(actualData, predData);
            
            // Calculate accuracy metrics
            const mse = this.calculateMSE(predData, actualData);
            const rmse = Math.sqrt(mse);
            
            this.updateStatus(`Training completed! Final RMSE: ${rmse.toFixed(6)}`, 'success');
            
            // Show metrics
            this.showMetrics(mse, rmse);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.updateStatus('Error generating predictions', 'error');
        }
    }

    updatePredictionChart(actualData, predData) {
        // Create time labels for x-axis
        const timeLabels = Array.from({length: Math.min(actualData.length, 100)}, (_, i) => `T+${i + 1}`);
        
        // Limit data points for better visualization
        const displayActual = actualData.slice(0, 100);
        const displayPred = predData.slice(0, 100);
        
        this.predictionChart.data.labels = timeLabels;
        this.predictionChart.data.datasets[0].data = displayActual;
        this.predictionChart.data.datasets[1].data = displayPred;
        this.predictionChart.update();
    }

    showMetrics(mse, rmse) {
        const metricsDiv = document.getElementById('metrics');
        if (metricsDiv) {
            metricsDiv.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${mse.toExponential(4)}</div>
                        <div class="metric-label">Mean Squared Error</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${rmse.toFixed(6)}</div>
                        <div class="metric-label">Root Mean Squared Error</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${this.currentEpoch + 1}</div>
                        <div class="metric-label">Training Epochs</div>
                    </div>
                </div>
            `;
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
