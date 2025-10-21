import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

export class LSTMModel {
  constructor(sequenceLength, featureCount) {
    this.sequenceLength = sequenceLength;
    this.featureCount = featureCount;
    this.model = null;
    this.trainingHistory = {
      loss: [],
      valLoss: []
    };
  }

  buildModel() {
    this.model = tf.sequential();
    
    // First LSTM layer
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: true,
      inputShape: [this.sequenceLength, this.featureCount]
    }));
    
    // Second LSTM layer
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: false
    }));
    
    // Dense layers
    this.model.add(tf.layers.dense({ units: 25, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1 }));
    
    // Compile model with RMSE as loss (using MSE and then taking sqrt)
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
    
    return this.model;
  }

  async train(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32) {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    // Clear previous history
    this.trainingHistory = { loss: [], valLoss: [] };

    const history = await this.model.fit(X_train, y_train, {
      epochs: epochs,
      batchSize: batchSize,
      validationData: [X_test, y_test],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          // Calculate RMSE from MSE
          const trainRMSE = Math.sqrt(logs.loss);
          const valRMSE = Math.sqrt(logs.val_loss);
          
          this.trainingHistory.loss.push(trainRMSE);
          this.trainingHistory.valLoss.push(valRMSE);
          
          // Update UI if callback provided
          if (this.onEpochEndCallback) {
            this.onEpochEndCallback(epoch, trainRMSE, valRMSE);
          }
          
          // Memory cleanup
          await tf.nextFrame();
        }
      }
    });

    return history;
  }

  setEpochCallback(callback) {
    this.onEpochEndCallback = callback;
  }

  async predict(X) {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }
    return this.model.predict(X);
  }

  async saveModel() {
    if (!this.model) {
      throw new Error('No model to save.');
    }
    
    const saveResult = await this.model.save('indexeddb://wti-prediction-model');
    return saveResult;
  }

  async loadModel() {
    try {
      this.model = await tf.loadLayersModel('indexeddb://wti-prediction-model');
      return true;
    } catch (error) {
      console.warn('No saved model found:', error);
      return false;
    }
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }

  getTrainingHistory() {
    return this.trainingHistory;
  }
}
