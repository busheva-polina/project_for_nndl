class LSTMModel {
    constructor(inputShape) {
        this.model = null;
        this.inputShape = inputShape;
        this.history = {
            loss: [],
            val_loss: []
        };
    }

    buildModel() {
        this.model = tf.sequential();
        
        // First LSTM layer
        this.model.add(tf.layers.lstm({
            units: 50,
            returnSequences: true,
            inputShape: this.inputShape
        }));
        
        // Second LSTM layer
        this.model.add(tf.layers.lstm({
            units: 50,
            returnSequences: false
        }));
        
        // Dense layers
        this.model.add(tf.layers.dense({units: 25, activation: 'relu'}));
        this.model.add(tf.layers.dense({units: 1, activation: 'linear'}));

        // Compile model
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
        this.history = { loss: [], val_loss: [] };

        await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.history.loss.push(logs.loss);
                    this.history.val_loss.push(logs.val_loss);
                    
                    // Update progress
                    if (typeof this.onEpochEnd === 'function') {
                        this.onEpochEnd(epoch, logs);
                    }
                }
            }
        });
    }

    predict(X) {
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
            this.model = null;
        }
    }
}
