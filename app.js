import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { DataLoader } from './data-loader.js';
import { LSTMModel } from './lstm.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model = null;
    this.isTraining = false;
    this.chart = null;
    
    this.initializeUI();
    this.setupEventListeners();
  }

  initializeUI() {
    // Create main container with rose theme
    document.body.style.cssText = `
      margin: 0; 
      padding: 20px; 
      font-family: Arial, sans-serif; 
      background: linear-gradient(135deg, #ffe4e6, #fecdd3);
      min-height: 100vh;
    `;

    const container = document.createElement('div');
    container.innerHTML = `
      <div style="max-width: 1200px; margin: 0 auto;">
        <h1 style="color: #be123c; text-align: center; margin-bottom: 30px;">
          📈 WTI Oil Price Prediction (LSTM)
        </h1>
        
        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
          <h3 style="color: #be123c; margin-top: 0;">Upload Data</h3>
          <input type="file" id="csvFile" accept=".csv" style="margin-bottom: 15px;">
          <div id="fileInfo" style="color: #666; font-size: 14px;"></div>
        </div>

        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
          <h3 style="color: #be123c; margin-top: 0;">Model Controls</h3>
          <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;">
            <button id="buildModelBtn" style="background: #f43f5e; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">
              Build Model
            </button>
            <button id="trainBtn" style="background: #ec4899; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;" disabled>
              Train Model
            </button>
            <button id="loadModelBtn" style="background: #db2777; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">
              Load Saved Model
            </button>
          </div>
          <div style="color: #666; font-size: 14px;">
            Epochs: <input type="number" id="epochs" value="50" min="1" max="200" style="width: 80px; margin-right: 15px;">
            Batch Size: <input type="number" id="batchSize" value="32" min="1" max="128" style="width: 80px;">
          </div>
        </div>

        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
          <h3 style="color: #be123c; margin-top: 0;">Training Progress</h3>
          <div id="trainingInfo" style="color: #666; margin-bottom: 15px;">Ready to train...</div>
          <div id="lossChartContainer">
            <canvas id="lossChart" width="400" height="200"></canvas>
          </div>
        </div>

        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
          <h3 style="color: #be123c; margin-top: 0;">Predictions</h3>
          <div id="predictions" style="color: #666;">No predictions yet.</div>
        </div>
      </div>
    `;
    
    document.body.appendChild(container);
  }

  setupEventListeners() {
    document.getElementById('csvFile').addEventListener('change', (e) => {
      this.handleFileUpload(e);
    });

    document.getElementById('buildModelBtn').addEventListener('click', () => {
      this.buildModel();
    });

    document.getElementById('trainBtn').addEventListener('click', () => {
      this.trainModel();
    });

    document.getElementById('loadModelBtn').addEventListener('click', () => {
      this.loadSavedModel();
    });
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      this.updateTrainingInfo('Loading CSV data...');
      await this.dataLoader.loadCSV(file);
      
      const fileInfo = `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      document.getElementById('fileInfo').textContent = fileInfo;
      
      this.updateTrainingInfo('CSV data loaded successfully!');
      
      // Enable build model button
      document.getElementById('buildModelBtn').disabled = false;
      
    } catch (error) {
      this.updateTrainingInfo(`Error loading file: ${error.message}`);
      console.error('File loading error:', error);
    }
  }

  async buildModel() {
    try {
      this.updateTrainingInfo('Preparing sequences...');
      
      const sequences = this.dataLoader.prepareSequences();
      this.sequences = sequences;
      
      this.model = new LSTMModel(30, 3); // 30 days, 3 features
      this.model.buildModel();
      
      // Set up epoch callback for UI updates
      this.model.setEpochCallback((epoch, trainLoss, valLoss) => {
        this.updateTrainingProgress(epoch, trainLoss, valLoss);
      });

      document.getElementById('trainBtn').disabled = false;
      this.updateTrainingInfo(`Model built! Training data: ${sequences.trainSize} sequences, Test data: ${sequences.testSize} sequences`);
      
    } catch (error) {
      this.updateTrainingInfo(`Error building model: ${error.message}`);
      console.error('Model building error:', error);
    }
  }

  async trainModel() {
    if (this.isTraining) return;
    
    this.isTraining = true;
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('buildModelBtn').disabled = true;
    
    try {
      const epochs = parseInt(document.getElementById('epochs').value) || 50;
      const batchSize = parseInt(document.getElementById('batchSize').value) || 32;
      
      this.updateTrainingInfo(`Starting training for ${epochs} epochs...`);
      this.initializeChart();
      
      await this.model.train(
        this.sequences.X_train, 
        this.sequences.y_train, 
        this.sequences.X_test, 
        this.sequences.y_test,
        epochs,
        batchSize
      );
      
      // Save model after training
      await this.model.saveModel();
      this.updateTrainingInfo('Training completed! Model saved.');
      
    } catch (error) {
      this.updateTrainingInfo(`Training error: ${error.message}`);
      console.error('Training error:', error);
    } finally {
      this.isTraining = false;
      document.getElementById('trainBtn').disabled = false;
      document.getElementById('buildModelBtn').disabled = false;
    }
  }

  async loadSavedModel() {
    try {
      this.updateTrainingInfo('Loading saved model...');
      
      if (!this.model) {
        this.model = new LSTMModel(30, 3);
      }
      
      const loaded = await this.model.loadModel();
      if (loaded) {
        this.updateTrainingInfo('Saved model loaded successfully!');
      } else {
        this.updateTrainingInfo('No saved model found. Please train a new model.');
      }
    } catch (error) {
      this.updateTrainingInfo(`Error loading model: ${error.message}`);
      console.error('Model loading error:', error);
    }
  }

  initializeChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    if (this.chart) {
      this.chart.destroy();
    }
    
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Training Loss (RMSE)',
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            data: []
          },
          {
            label: 'Validation Loss (RMSE)',
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            data: []
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
              text: 'RMSE Loss'
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
  }

  updateTrainingProgress(epoch, trainLoss, valLoss) {
    const trainingInfo = `Epoch ${epoch + 1}: Train RMSE = ${trainLoss.toFixed(4)}, Val RMSE = ${valLoss.toFixed(4)}`;
    document.getElementById('trainingInfo').textContent = trainingInfo;
    
    // Update chart
    if (this.chart) {
      this.chart.data.labels.push(epoch + 1);
      this.chart.data.datasets[0].data.push(trainLoss);
      this.chart.data.datasets[1].data.push(valLoss);
      this.chart.update();
    }
  }

  updateTrainingInfo(message) {
    document.getElementById('trainingInfo').textContent = message;
  }

  dispose() {
    if (this.dataLoader) {
      this.dataLoader.dispose();
    }
    if (this.model) {
      this.model.dispose();
    }
    if (this.chart) {
      this.chart.destroy();
    }
  }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new StockPredictionApp();
});

// Cleanup when page unloads
window.addEventListener('beforeunload', () => {
  if (app) {
    app.dispose();
  }
});
