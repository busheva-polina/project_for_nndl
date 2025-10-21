class GRUModel {
    constructor(sequenceLength = 30, featureCount = 3) {
        this.sequenceLength = sequenceLength;
        this.featureCount = featureCount;
        this.model = null;
        this.trainingHistory = {
            loss: [],
            valLoss: []
        };
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 50,
                    returnSequences: true,
                    inputShape: [this.sequenceLength, this.featureCount]
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({
                    units: 50,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 25, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'linear' })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        console.log('GRU Model built successfully');
        this.model.summary();
    }

    async train(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32, progressCallback = null) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.trainingHistory = { loss: [], valLoss: [] };

        await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    this.trainingHistory.loss.push(logs.loss);
                    this.trainingHistory.valLoss.push(logs.val_loss);
                    
                    if (progressCallback) {
                        const progress = ((epoch + 1) / epochs) * 100;
                        progressCallback(progress, epoch + 1, logs);
                    }

                    // Prevent memory leaks
                    await tf.nextFrame();
                }
            }
        });

        console.log('Training completed');
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
        
        const saveResult = await this.model.save('indexeddb://gru-stock-model');
        console.log('Model saved successfully');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://gru-stock-model');
            console.log('Model loaded successfully');
            
            // Update sequence length and feature count from loaded model
            const inputShape = this.model.layers[0].batchInputShape;
            this.sequenceLength = inputShape[1];
            this.featureCount = inputShape[2];
            
            return true;
        } catch (error) {
            console.warn('No saved model found or error loading model:', error);
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

// Export for use in other modules
window.GRUModel = GRUModel;
