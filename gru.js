import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';

export class GRUModel {
    constructor(sequenceLength, numFeatures) {
        this.sequenceLength = sequenceLength;
        this.numFeatures = numFeatures;
        this.model = null;
        this.trainingHistory = null;
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: [this.sequenceLength, this.numFeatures]
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 1 })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 100, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel first.');
        }

        const onEpochEnd = callbacks.onEpochEnd || (() => {});
        const onTrainEnd = callbacks.onTrainEnd || (() => {});

        this.trainingHistory = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    onEpochEnd(epoch, logs);
                    await tf.nextFrame(); // Prevent UI blocking
                },
                onTrainEnd: onTrainEnd
            }
        });

        return this.trainingHistory;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel first.');
        }
        return this.model.predict(X);
    }

    async saveModel(name = 'gru-model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save(`indexeddb://${name}`);
        return saveResult;
    }

    async loadModel(name = 'gru-model') {
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            // Update model configuration
            this.sequenceLength = this.model.inputs[0].shape[1];
            this.numFeatures = this.model.inputs[0].shape[2];
            return true;
        } catch (error) {
            console.warn('Failed to load model:', error);
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
