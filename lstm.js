import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';

export class LSTMModel {
    constructor(sequenceLength = 30, featureCount = 3) {
        this.sequenceLength = sequenceLength;
        this.featureCount = featureCount;
        this.model = null;
        this.trainingHistory = {
            loss: [],
            valLoss: [],
            epochs: []
        };
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.lstm({
                    units: 50,
                    returnSequences: true,
                    inputShape: [this.sequenceLength, this.featureCount]
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.lstm({
                    units: 50,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 25, activation: 'relu' }),
                tf.layers.dense({ units: 1 })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        console.log('LSTM model built successfully');
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 100, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        const onEpochEnd = (epoch, logs) => {
            this.trainingHistory.loss.push(logs.loss);
            this.trainingHistory.valLoss.push(logs.val_loss);
            this.trainingHistory.epochs.push(epoch + 1);

            if (callbacks.onEpochEnd) {
                callbacks.onEpochEnd(epoch, logs, this.trainingHistory);
            }
        };

        const history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: onEpochEnd.bind(this),
                onTrainBegin: callbacks.onTrainBegin,
                onTrainEnd: callbacks.onTrainEnd
            },
            verbose: 0
        });

        return history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }
        return this.model.predict(X);
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }

        const saveResult = await this.model.save('indexeddb://wti-prediction-model');
        console.log('Model saved successfully');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://wti-prediction-model');
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.log('No saved model found:', error.message);
            return false;
        }
    }

    getTrainingHistory() {
        return this.trainingHistory;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    summary() {
        if (this.model) {
            this.model.summary();
        }
    }
}
