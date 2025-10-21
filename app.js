import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.trainingData = null;
        this.isTraining = false;
        this.lossChart = null;
        
        this.initializeEventListeners();
        this.setupCharts();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('stopTraining').addEventListener('click', () => this.stopTraining());
        document.getElementById('saveModel').addEventListener('click', () => this.saveModel());
        document.getElementById('loadModel').addEventListener('click', () => this.loadModel());
    }

    setupCharts() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        this.lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        borderColor: '#0066cc',
                        backgroundColor: 'rgba(0, 102, 204, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        data: []
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: '#00cc66',
                        backgroundColor: 'rgba(0, 204, 102, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        data: []
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: '#fff'
                        },
                        grid: { color: '#333' },
                        ticks: { color: '#fff' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss (Binary Crossentropy)',
                            color: '#fff'
                        },
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
                        text: 'Training Progress',
                        color: '#fff',
                        font: { size: 16 }
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
            this.updateStatus('fileStatus', 'CSV file loaded successfully! Ready for training.', 'success');
            document.getElementById('trainModel').disabled = false;
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
            this.setTrainingUIState(true);
            
            const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            const batchSize = parseInt(document.getElementById('batchSize').value);
            const learningRate = parseFloat(document.getElementById('learningRate').value);

            this.updateStatus('trainingStatus', 'Preparing data...', 'info');
            
            if (this.trainingData) {
                this.disposeTrainingData();
            }

            this.trainingData = this.dataLoader.prepareData(sequenceLength);
            
            this.updateStatus('trainingStatus', 'Building model...', 'info');
            this.model.buildModel(
                sequenceLength, 
                this.trainingData.featureNames.length, 
                this.trainingData.featureNames.length, 
                learningRate
            );

            this.updateStatus('trainingStatus', 'Starting training...', 'info');
            this.lossChart.data.labels = [];
            this.lossChart.data.datasets[0].data = [];
            this.lossChart.data.datasets[1].data = [];
            this.lossChart.update();
            
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
                            `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Acc: ${logs.acc?.toFixed(4)}`, 
                            'info'
                        );
                        
                        this.updateLossChart(history);
                    }
                }
            );

            await this.model.saveWeights();
            document.getElementById('saveModel').disabled = false;
            document.getElementById('loadModel').disabled = false;
            
            this.updateStatus('trainingStatus', 
                `Training completed! Final Loss: ${history.loss[history.loss.length-1].toFixed(4)}, Val Loss: ${history.val_loss[history.val_loss.length-1].toFixed(4)}`, 
                'success'
            );

        } catch (error) {
            this.updateStatus('trainingStatus', `Training error: ${error.message}`, 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            this.setTrainingUIState(false);
            document.getElementById('trainingProgress').style.width = '0%';
        }
    }

    updateLossChart(history) {
        this.lossChart.data.labels = history.epochs;
        this.lossChart.data.datasets[0].data = history.loss;
        this.lossChart.data.datasets[1].data = history.val_loss;
        this.lossChart.update();
    }

    async saveModel() {
        try {
            this.updateStatus('resultsStatus', 'Saving model...', 'info');
            await this.model.saveModel();
            this.updateStatus('resultsStatus', 'Model saved successfully!', 'success');
        } catch (error) {
            this.updateStatus('resultsStatus', `Error saving model: ${error.message}`, 'error');
        }
    }

    async loadModel() {
        try {
            this.updateStatus('resultsStatus', 'Loading model...', 'info');
            const success = await this.model.loadModel();
            if (success) {
                this.updateStatus('resultsStatus', 'Model loaded successfully from browser storage!', 'success');
                document.getElementById('saveModel').disabled = false;
            } else {
                this.updateStatus('resultsStatus', 'No saved model found. Please train a model first.', 'error');
            }
        } catch (error) {
            this.updateStatus('resultsStatus', `Error loading model: ${error.message}`, 'error');
        }
    }

    setTrainingUIState(training) {
        document.getElementById('trainModel').disabled = training;
        document.getElementById('stopTraining').disabled = !training;
        document.getElementById('loadData').disabled = training;
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

    disposeTrainingData() {
        if (this.trainingData) {
            Object.values(this.trainingData).forEach(tensor => {
                if (tensor && tensor.dispose && tensor instanceof tf.Tensor) {
                    tensor.dispose();
                }
            });
            this.trainingData = null;
        }
    }

    dispose() {
        this.model.dispose();
        this.dataLoader.dispose();
        this.disposeTrainingData();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
