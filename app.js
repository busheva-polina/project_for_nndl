import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';
import { DataLoader } from './data-loader.js';
import { LSTMModel } from './lstm.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new LSTMModel();
        this.isTraining = false;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        
        this.initializeUI();
        this.setupEventListeners();
    }

    initializeUI() {
        // Create main container with rose theme
        document.body.style.cssText = `
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ffe6e6, #ffb6c1);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        `;

        const container = document.createElement('div');
        container.innerHTML = `
            <div style="max-width: 1200px; margin: 0 auto;">
                <h1 style="color: #c44569; text-align: center; margin-bottom: 30px;">
                    📈 WTI Oil Price Prediction (LSTM)
                </h1>
                
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(199, 21, 133, 0.1); margin-bottom: 20px;">
                    <h3 style="color: #c44569; margin-top: 0;">📁 Data Upload</h3>
                    <input type="file" id="csvFile" accept=".csv" style="margin: 10px 0; padding: 8px; border: 2px dashed #ff69b4; border-radius: 8px; width: 100%;">
                    <div id="fileInfo" style="color: #666; font-size: 14px; margin: 5px 0;"></div>
                </div>

                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(199, 21, 133, 0.1); margin-bottom: 20px;">
                    <h3 style="color: #c44569; margin-top: 0;">⚙️ Training Controls</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                        <div>
                            <label style="color: #c44569; font-weight: bold;">Epochs:</label>
                            <input type="number" id="epochs" value="50" min="1" max="200" style="width: 100%; padding: 8px; border: 1px solid #ff69b4; border-radius: 5px;">
                        </div>
                        <div>
                            <label style="color: #c44569; font-weight: bold;">Sequence Length:</label>
                            <input type="number" id="sequenceLength" value="30" min="10" max="60" style="width: 100%; padding: 8px; border: 1px solid #ff69b4; border-radius: 5px;">
                        </div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <button id="trainBtn" style="flex: 1; background: #ff69b4; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold;">
                            🚀 Train Model
                        </button>
                        <button id="loadBtn" style="flex: 1; background: #9370db; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold;">
                            📥 Load Model
                        </button>
                        <button id="resetBtn" style="flex: 1; background: #ff4757; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold;">
                            🔄 Reset
                        </button>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(199, 21, 133, 0.1);">
                        <h3 style="color: #c44569; margin-top: 0;">📊 Training Progress</h3>
                        <div id="progressInfo" style="color: #666; font-size: 14px; margin-bottom: 15px;">
                            No training started yet
                        </div>
                        <div id="lossChart" style="height: 300px; background: #f8f9fa; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999;">
                            Loss chart will appear here
                        </div>
                    </div>

                    <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(199, 21, 133, 0.1);">
                        <h3 style="color: #c44569; margin-top: 0;">📈 Model Performance</h3>
                        <div id="performanceInfo" style="color: #666; font-size: 14px;">
                            Train RMSE: -<br>
                            Test RMSE: -<br>
                            Last Epoch: -
                        </div>
                        <div id="predictionChart" style="height: 300px; background: #f8f9fa; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999;">
                            Prediction visualization will appear here
                        </div>
                    </div>
                </div>

                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(199, 21, 133, 0.1);">
                    <h3 style="color: #c44569; margin-top: 0;">📋 Console Log</h3>
                    <div id="console" style="height: 150px; overflow-y: auto; background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px;">
                        Welcome to WTI Oil Price Prediction App!
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(container);
        
        // Initialize charts
        this.initializeCharts();
    }

    initializeCharts() {
        // Placeholder for chart initialization
        // In a real implementation, you would use Chart.js or similar
        this.lossChart = null;
        this.predictionChart = null;
    }

    setupEventListeners() {
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e);
        });

        document.getElementById('trainBtn').addEventListener('click', () => {
            this.startTraining();
        });

        document.getElementById('loadBtn').addEventListener('click', () => {
            this.loadModel();
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetApp();
        });
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.log(`Loading CSV file: ${file.name}`);
        
        try {
            await this.dataLoader.loadCSV(file);
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.textContent = `✅ Loaded ${this.dataLoader.data.length} records, ${this.dataLoader.trainData.sequences.length} training sequences, ${this.dataLoader.testData.sequences.length} test sequences`;
            
            this.log('Data loaded and normalized successfully');
        } catch (error) {
            this.log(`❌ Error loading file: ${error.message}`);
        }
    }

    async startTraining() {
        if (this.isTraining) {
            this.log('Training already in progress');
            return;
        }

        if (this.dataLoader.trainData.sequences.length === 0) {
            this.log('❌ Please load CSV data first');
            return;
        }

        this.isTraining = true;
        this.currentEpoch = 0;
        this.totalEpochs = parseInt(document.getElementById('epochs').value) || 50;

        this.log(`Starting training for ${this.totalEpochs} epochs...`);

        try {
            // Get tensors
            const { X: X_train, y: y_train } = this.dataLoader.getTrainTensors();
            const { X: X_test, y: y_test } = this.dataLoader.getTestTensors();

            // Build model
            this.model.buildModel();

            // Train model
            await this.model.train(X_train, y_train, X_test, y_test, this.totalEpochs, {
                onEpochEnd: (epoch, logs, history) => {
                    this.currentEpoch = epoch + 1;
                    this.updateProgress(epoch + 1, this.totalEpochs, logs);
                    this.updateLossChart(history);
                },
                onTrainEnd: () => {
                    this.trainingComplete(X_test, y_test);
                }
            });

            // Cleanup tensors
            X_train.dispose();
            y_train.dispose();
            X_test.dispose();
            y_test.dispose();

        } catch (error) {
            this.log(`❌ Training error: ${error.message}`);
            this.isTraining = false;
        }
    }

    updateProgress(currentEpoch, totalEpochs, logs) {
        const progressInfo = document.getElementById('progressInfo');
        const progress = ((currentEpoch / totalEpochs) * 100).toFixed(1);
        
        progressInfo.innerHTML = `
            <div style="color: #c44569; font-weight: bold;">
                Epoch: ${currentEpoch}/${totalEpochs} (${progress}%)
            </div>
            <div style="font-size: 12px; color: #666;">
                Loss: ${logs.loss.toFixed(6)} | Val Loss: ${logs.val_loss.toFixed(6)}
            </div>
        `;
    }

    updateLossChart(history) {
        const lossChart = document.getElementById('lossChart');
        
        // Simple text-based chart update
        // In production, integrate with Chart.js for proper visualization
        if (history.loss.length > 0) {
            const latestLoss = history.loss[history.loss.length - 1];
            const latestValLoss = history.valLoss[history.valLoss.length - 1];
            
            lossChart.innerHTML = `
                <div style="text-align: center; width: 100%;">
                    <div style="color: #c44569; font-weight: bold; margin-bottom: 10px;">
                        Training Progress
                    </div>
                    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
                        <div style="color: #ff6b6b;">Train Loss: ${latestLoss.toFixed(6)}</div>
                        <div style="color: #4834d4;">Val Loss: ${latestValLoss.toFixed(6)}</div>
                    </div>
                    <div style="background: linear-gradient(90deg, #ff6b6b ${(1 - Math.min(latestLoss, 1)) * 100}%, #eee 0%); 
                                height: 20px; border-radius: 10px; margin: 5px 0;"></div>
                    <div style="background: linear-gradient(90deg, #4834d4 ${(1 - Math.min(latestValLoss, 1)) * 100}%, #eee 0%); 
                                height: 20px; border-radius: 10px; margin: 5px 0;"></div>
                    <div style="font-size: 12px; color: #666; margin-top: 10px;">
                        Epoch ${history.epochs.length} of ${this.totalEpochs}
                    </div>
                </div>
            `;
        }
    }

    async trainingComplete(X_test, y_test) {
        this.isTraining = false;
        this.log('✅ Training completed!');
        
        try {
            // Calculate final metrics
            const predictions = await this.model.predict(X_test);
            const testLoss = tf.losses.meanSquaredError(y_test, predictions).dataSync()[0];
            const testRMSE = Math.sqrt(testLoss);
            
            // Update performance info
            const performanceInfo = document.getElementById('performanceInfo');
            performanceInfo.innerHTML = `
                Train RMSE: Calculating...<br>
                Test RMSE: ${testRMSE.toFixed(6)}<br>
                Last Epoch: ${this.totalEpochs}
            `;

            // Save model
            await this.model.saveModel();
            this.log('Model saved to browser storage');

            // Cleanup
            predictions.dispose();
            
        } catch (error) {
            this.log(`❌ Error in training completion: ${error.message}`);
        }
    }

    async loadModel() {
        this.log('Loading saved model...');
        
        try {
            const success = await this.model.loadModel();
            if (success) {
                this.log('✅ Model loaded successfully');
            } else {
                this.log('❌ No saved model found. Train a model first.');
            }
        } catch (error) {
            this.log(`❌ Error loading model: ${error.message}`);
        }
    }

    resetApp() {
        this.model.dispose();
        this.dataLoader.dispose();
        this.model = new LSTMModel();
        this.isTraining = false;
        
        document.getElementById('progressInfo').textContent = 'No training started yet';
        document.getElementById('performanceInfo').innerHTML = 'Train RMSE: -<br>Test RMSE: -<br>Last Epoch: -';
        document.getElementById('lossChart').innerHTML = 'Loss chart will appear here';
        document.getElementById('predictionChart').innerHTML = 'Prediction visualization will appear here';
        document.getElementById('console').textContent = 'Welcome to WTI Oil Price Prediction App!';
        
        this.log('App reset successfully');
    }

    log(message) {
        const console = document.getElementById('console');
        const timestamp = new Date().toLocaleTimeString();
        console.innerHTML += `[${timestamp}] ${message}\n`;
        console.scrollTop = console.scrollHeight;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new StockPredictionApp();
});
