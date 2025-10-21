class GRUModel {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.trainingStopRequested = false;
        this.history = {
            loss: [],
            val_loss: [],
            epochs: []
        };
    }

    buildModel(sequenceLength, numFeatures, numOutputs, learningRate = 0.001) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: [sequenceLength, numFeatures],
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01})
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.gru({
                    units: 32,
                    returnSequences: false,
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01})
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.dense({ 
                    units: 16, 
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01})
                }),
                tf.layers.batchNormalization(),
                
                tf.layers.dense({ 
                    units: numOutputs, 
                    activation: 'sigmoid'
                })
            ]
        });

        const optimizer = tf.train.adam(learningRate);
        
        this.model.compile({
            optimizer: optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'mse']
        });

        console.log('Multi-output GRU model built successfully');
        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs, batchSize, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.isTraining = true;
        this.trainingStopRequested = false;
        this.history = { loss: [], val_loss: [], epochs: [] };
        
        const batchCount = Math.ceil(X_train.shape[0] / batchSize);
        
        try {
            await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                callbacks: {
                    onEpochBegin: async (epoch) => {
                        if (this.trainingStopRequested) {
                            this.model.stopTraining = true;
                            this.isTraining = false;
                            return;
                        }
                    },
                    onEpochEnd: async (epoch, logs) => {
                        if (this.trainingStopRequested) {
                            this.model.stopTraining = true;
                            this.isTraining = false;
                            return;
                        }

                        this.history.loss.push(logs.loss);
                        this.history.val_loss.push(logs.val_loss);
                        this.history.epochs.push(epoch + 1);

                        if (callbacks.onEpochEnd) {
                            callbacks.onEpochEnd(epoch, logs, this.history);
                        }

                        if (epoch % 3 === 0) {
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

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built or trained');
        }
        return this.model.predict(X);
    }

    async evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not built or trained');
        }
        return this.model.evaluate(X_test, y_test);
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('downloads://multi-stock-gru-model');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://multi-stock-gru-model');
            console.log('Model loaded successfully from IndexedDB');
            return true;
        } catch (error) {
            console.log('No saved model found in IndexedDB');
            return false;
        }
    }

    async saveWeights() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const modelData = await this.model.save('indexeddb://multi-stock-gru-model');
        return modelData;
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
