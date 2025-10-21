import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';
import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.trainingData = null;
        this.isTraining = false;
        this.chart = null;
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

        // Save/Load button handlers
        document.getElementById('saveModel').addEventListener('click', () => {
            this.saveModel();
        });

        document.getElementById('loadModel').addEventListener('click', () => {
            this.loadModel();
        });

        // Initialize chart
        this.initializeChart();
    }

    async handleFileUpload(file) {
        try {
            this.updateStatus('Loading CSV file...');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('CSV loaded successfully. Ready to train.');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`, 'error');
        }
    }

    async startTraining() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            this.updateStatus('Preparing training data...');
            
            // Prepare sequences
            this.trainingData = this.dataLoader.prepareSequences();
            
            const { X_train, y_train, X_test, y_test, trainSize, testSize } = this.trainingData;
            
            this.updateStatus(`Training on ${trainSize} samples, testing on ${testSize} samples`);
            
            // Build model
            this.model = new GRUModel(
                this.dataLoader.sequenceLength,
                this.dataLoader.featureColumns.length
            );
            this.model.buildModel();

            // Setup training callbacks
            const trainingCallbacks = {
                onEpochEnd: (epoch, logs) => {
                    this.updateTrainingProgress(epoch, logs);
                    this.updateChart(logs);
                },
                onTrainEnd: () => {
                    this.trainingComplete();
                }
            };

            // Start training
            const epochs = parseInt(document.getElementById('epochs').value) || 50;
            await this.model.train(X_train, y_train, X_test, y_test, epochs, trainingCallbacks);

        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, 'error');
            this.isTraining = false;
        }
    }

    updateTrainingProgress(epoch, logs) {
        const status = `Epoch ${epoch + 1} - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}`;
        this.updateStatus(status);
        
        // Update progress bar if exists
        const progressBar = document.getElementById('trainingProgress');
        if (progressBar) {
            const epochsTotal = parseInt(document.getElementById('epochs').value) || 50;
            const progress = ((epoch + 1) / epochsTotal) * 100;
            progressBar.value = progress;
        }
    }

    updateChart(logs) {
        if (!this.chart) return;

        // Add data points to chart
        this.chart.data.labels.push(this.chart.data.labels.length + 1);
        this.chart.data.datasets[0].data.push(logs.loss);
        this.chart.data.datasets[1].data.push(logs.val_loss);
        
        // Limit data points to prevent memory issues
        if (this.chart.data.labels.length > 100) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
            this.chart.data.datasets[1].data.shift();
        }
        
        this.chart.update('none');
    }

    initializeChart() {
        const ctx = document.getElementById('trainingChart').getContext('2d');
        this.chart = new Chart(ctx, {
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

    trainingComplete() {
        this.isTraining = false;
        this.updateStatus('Training completed!');
        
        // Enable prediction
        document.getElementById('predictBtn').disabled = false;
        
        // Show model summary
        if (this.model && this.model.model) {
            console.log('Model summary:');
            this.model.model.summary();
        }
    }

    async saveModel() {
        if (!this.model) {
            this.updateStatus('No model to save', 'error');
            return;
        }
        
        try {
            await this.model.saveModel();
            this.updateStatus('Model saved successfully!');
        } catch (error) {
            this.updateStatus(`Error saving model: ${error.message}`, 'error');
        }
    }

    async loadModel() {
        try {
            this.model = new GRUModel();
            const success = await this.model.loadModel();
            
            if (success) {
                this.updateStatus('Model loaded successfully!');
                document.getElementById('predictBtn').disabled = false;
            } else {
                this.updateStatus('No saved model found', 'warning');
            }
        } catch (error) {
            this.updateStatus(`Error loading model: ${error.message}`, 'error');
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        
        // Reset classes
        statusElement.className = 'status';
        if (type === 'error') {
            statusElement.classList.add('error');
        } else if (type === 'warning') {
            statusElement.classList.add('warning');
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
