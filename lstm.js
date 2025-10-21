class LSTMStockPredictor {
    constructor() {
        this.model = null;
        this.history = null;
    }

    buildModel(inputShape, units = 64, layers = 5) {
        this.model = tf.sequential();
        
        // First LSTM layer
        this.model.add(tf.layers.lstm({
            units: units,
            returnSequences: layers > 1,
            inputShape: inputShape
        }));
        
        // Additional LSTM layers
        for (let i = 1; i < layers - 1; i++) {
            this.model.add(tf.layers.lstm({
                units: units,
                returnSequences: i < layers - 2
            }));
        }
        
        // Final Dense layer
        this.model.add(tf.layers.dense({ units: 1 }));
        
        // Compile model
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs = 40) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
                    
                    // Update UI if callback is provided
                    if (this.onEpochEnd) {
                        this.onEpochEnd(epoch, logs);
                    }
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        return this.model.predict(X);
    }

    getTrainingHistory() {
        return this.history;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
