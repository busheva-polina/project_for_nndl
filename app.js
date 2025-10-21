class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.predictor = new LSTMStockPredictor();
        this.isTraining = false;
        this.trainingLogs = [];
        
        this.initializeUI();
    }

    initializeUI() {
        // File input handler
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const statusDiv = document.getElementById('status');
        const progressDiv = document.getElementById('progress');

        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        trainBtn.addEventListener('click', () => {
            this.startTraining();
        });
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateStatus('Loading CSV data...');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('Data loaded successfully. Ready to train.');
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`, 'error');
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.trainingLogs = [];
            this.updateStatus('Preparing data...');
            
            // Prepare data
            const { X_train, y_train, X_test, y_test, featureNames } = this.dataLoader.prepareData();
            
            this.updateStatus(`Data prepared. Training samples: ${X_train.shape[0]}, Test samples: ${X_test.shape[0]}`);
            
            // Build model
            this.updateStatus('Building LSTM model...');
            this.predictor.buildModel([X_train.shape[1], X_train.shape[2]]);
            
            // Set up training callbacks
            this.predictor.onEpochEnd = (epoch, logs) => {
                this.trainingLogs.push({
                    epoch: epoch + 1,
                    loss: logs.loss,
                    val_loss: logs.val_loss
                });
                
                this.updateProgress(epoch + 1, 40, logs);
                this.plotTrainingHistory();
            };
            
            // Train model
            this.updateStatus('Starting training...');
            await this.predictor.trainModel(X_train, y_train, X_test, y_test, 40);
            
            // Make predictions
            this.updateStatus('Making predictions...');
            const predictions = await this.predictor.predict(X_test);
            
            // Visualize results
            this.plotPredictions(y_test, predictions, this.dataLoader.scalers.WTI);
            
            this.updateStatus('Training completed successfully!');
            
            // Clean up tensors
            X_train.dispose();
            y_train.dispose();
            X_test.dispose();
            y_test.dispose();
            predictions.dispose();
            
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`, 'error');
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    updateStatus(message, type = 'info') {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }

    updateProgress(epoch, totalEpochs, logs) {
        const progressDiv = document.getElementById('progress');
        const percent = (epoch / totalEpochs) * 100;
        progressDiv.innerHTML = `
            <div>Epoch ${epoch}/${totalEpochs}</div>
            <div>Loss: ${logs.loss.toFixed(4)}</div>
            <div>Val Loss: ${logs.val_loss.toFixed(4)}</div>
            <progress value="${percent}" max="100"></progress>
        `;
    }

    plotTrainingHistory() {
        if (this.trainingLogs.length === 0) return;

        const losses = this.trainingLogs.map(log => log.loss);
        const valLosses = this.trainingLogs.map(log => log.val_loss);
        const epochs = this.trainingLogs.map(log => log.epoch);

        const trace1 = {
            x: epochs,
            y: losses,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: { color: 'blue' }
        };

        const trace2 = {
            x: epochs,
            y: valLosses,
            type: 'scatter',
            mode: 'lines',
            name: 'Validation Loss',
            line: { color: 'red' }
        };

        const layout = {
            title: 'Training and Validation Loss',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Mean Squared Error' }
        };

        Plotly.newPlot('lossChart', [trace1, trace2], layout);
    }

    plotPredictions(actual, predicted, scaler) {
        // Convert tensors to arrays
        const actualData = actual.dataSync();
        const predictedData = predicted.dataSync();
        
        // Unscale values
        const actualUnscaled = this.dataLoader.unscaleValues(Array.from(actualData), 'WTI');
        const predictedUnscaled = this.dataLoader.unscaleValues(Array.from(predictedData), 'WTI');
        
        const indices = Array.from({length: actualUnscaled.length}, (_, i) => i);

        const trace1 = {
            x: indices,
            y: actualUnscaled,
            type: 'scatter',
            mode: 'lines',
            name: 'Actual WTI',
            line: { color: 'blue' }
        };

        const trace2 = {
            x: indices,
            y: predictedUnscaled,
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted WTI',
            line: { color: 'red' }
        };

        const layout = {
            title: 'WTI Crude Oil Price Prediction',
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Price' }
        };

        Plotly.newPlot('predictionChart', [trace1, trace2], layout);
    }

    dispose() {
        this.dataLoader.dispose();
        this.predictor.dispose();
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new StockPredictionApp();
});
