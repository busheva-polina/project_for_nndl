class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.predictor = new LSTMStockPredictor();
        this.isTraining = false;
        this.trainingLogs = [];
        
        this.initializeUI();
    }

    initializeUI() {
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const featuresList = document.getElementById('featuresList');

        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        trainBtn.addEventListener('click', () => {
            this.startTraining();
        });

        const featureNames = [
            'WTI (Target)',
            'Gold Futures', 
            'US Dollar Index Futures', 
            'US 10 Year Bond Yield', 
            'S&P 500', 
            'Dow Jones Utility Average'
        ];
        
        featuresList.innerHTML = featureNames.map(feature => 
            `<div class="feature-item">${feature}</div>`
        ).join('');
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateStatus('Loading CSV data...', 'info');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('Data loaded successfully. Ready to train.', 'success');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading file: ${error.message}`, 'error');
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.trainingLogs = [];
            this.updateStatus('Preparing data...', 'info');
            document.getElementById('trainBtn').disabled = true;
            
            const { X_train, y_train, X_test, y_test, featureNames } = this.dataLoader.prepareData();
            
            this.updateStatus(`Data prepared. Training samples: ${X_train.shape[0]}, Test samples: ${X_test.shape[0]}`, 'info');
            
            this.updateStatus('Building LSTM model...', 'info');
            this.predictor.buildModel([X_train.shape[1], X_train.shape[2]]);
            
            this.predictor.onEpochEnd = (epoch, logs) => {
                this.trainingLogs.push({
                    epoch: epoch + 1,
                    loss: logs.loss,
                    val_loss: logs.val_loss
                });
                
                this.updateProgress(epoch + 1, 40, logs);
                this.plotTrainingHistory();
            };
            
            this.updateStatus('Starting training...', 'info');
            await this.predictor.trainModel(X_train, y_train, X_test, y_test, 40);
            
            this.updateStatus('Making predictions...', 'info');
            const predictions = await this.predictor.predict(X_test);
            
            this.plotPredictions(y_test, predictions, this.dataLoader.scalers.WTI);
            
            this.updateStatus('Training completed successfully!', 'success');
            
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
            document.getElementById('trainBtn').disabled = false;
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
            <div class="progress-info">
                <span>Epoch ${epoch}/${totalEpochs}</span>
                <span>Loss: ${logs.loss ? logs.loss.toFixed(4) : 'N/A'}</span>
                <span>Val Loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}</span>
            </div>
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
            line: { color: '#00a8ff', width: 3 }
        };

        const trace2 = {
            x: epochs,
            y: valLosses,
            type: 'scatter',
            mode: 'lines',
            name: 'Validation Loss',
            line: { color: '#e84118', width: 3 }
        };

        const layout = {
            title: 'Training and Validation Loss',
            xaxis: { 
                title: 'Epoch',
                gridcolor: '#2f3640',
                color: '#f5f6fa'
            },
            yaxis: { 
                title: 'Mean Squared Error',
                gridcolor: '#2f3640',
                color: '#f5f6fa'
            },
            paper_bgcolor: '#1e272e',
            plot_bgcolor: '#1e272e',
            font: { color: '#f5f6fa' },
            legend: { font: { color: '#f5f6fa' } }
        };

        Plotly.react('lossChart', [trace1, trace2], layout);
    }

    plotPredictions(actual, predicted, scaler) {
        const actualData = actual.dataSync();
        const predictedData = predicted.dataSync();
        
        const actualUnscaled = this.dataLoader.unscaleValues(Array.from(actualData), 'WTI');
        const predictedUnscaled = this.dataLoader.unscaleValues(Array.from(predictedData), 'WTI');
        
        const indices = Array.from({length: actualUnscaled.length}, (_, i) => i);

        const trace1 = {
            x: indices,
            y: actualUnscaled,
            type: 'scatter',
            mode: 'lines',
            name: 'Actual WTI',
            line: { color: '#00a8ff', width: 2 }
        };

        const trace2 = {
            x: indices,
            y: predictedUnscaled,
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted WTI',
            line: { color: '#fbc531', width: 2 }
        };

        const layout = {
            title: 'WTI Crude Oil Price - Actual vs Predicted',
            xaxis: { 
                title: 'Time Steps',
                gridcolor: '#2f3640',
                color: '#f5f6fa'
            },
            yaxis: { 
                title: 'Price (USD)',
                gridcolor: '#2f3640',
                color: '#f5f6fa'
            },
            paper_bgcolor: '#1e272e',
            plot_bgcolor: '#1e272e',
            font: { color: '#f5f6fa' },
            legend: { font: { color: '#f5f6fa' } }
        };

        Plotly.react('predictionChart', [trace1, trace2], layout);
    }

    dispose() {
        this.dataLoader.dispose();
        this.predictor.dispose();
    }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new StockPredictionApp();
});
