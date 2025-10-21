class LSTMStockPredictor {
    constructor() {
        this.model = null;
        this.history = null;
        this.onEpochEnd = null;
    }

    buildModel(inputShape) {
        this.model = tf.sequential();
        
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            inputShape: inputShape
        }));
        
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true
        }));
        
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: false
        }));
        
        this.model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: 1 }));
        
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        console.log('Model built successfully');
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
