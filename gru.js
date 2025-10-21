class GRUModel {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.trainingStopRequested = false;
    }

    buildModel(sequenceLength, numFeatures, learningRate = 0.001) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: [sequenceLength, numFeatures]
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
            optimizer: tf.train.adam(learningRate),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        console.log('GRU model built successfully');
        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs, batchSize, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.isTraining = true;
        this.trainingStopRequested = false;
        
        const history = {
            loss: [],
            val_loss: [],
            epochs: []
        };

        const batchCount = Math.ceil(X_train.shape[0] / batchSize);
        
        try {
            await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        if (this.trainingStopRequested) {
                            this.model.stopTraining = true;
                            this.isTraining = false;
                            return;
                        }

                        history.loss.push(logs.loss);
                        history.val_loss.push(logs.val_loss);
                        history.epochs.push(epoch + 1);

                        if (callbacks.onEpochEnd) {
                            callbacks.onEpochEnd(epoch, logs, history);
                        }

                        // Force memory cleanup every few epochs
                        if (epoch % 5 === 0) {
                            await tf.nextFrame();
                        }
                    },
                    onBatchEnd: async (batch, logs) => {
                        if (callbacks.onBatchEnd) {
                            callbacks.onBatchEnd(batch, batchCount, logs);
                        }
                    }
                },
                shuffle: true,
                validationSplit: 0
            });
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        } finally {
            this.isTraining = false;
        }

        return history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built or trained');
        }
        return this.model.predict(X);
    }

    stopTraining() {
        this.trainingStopRequested = true;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }

    summary() {
        if (this.model) {
            this.model.summary();
        }
    }
}

export default GRUModel;
