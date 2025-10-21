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
                        if (callbacks.onBatchEnd
