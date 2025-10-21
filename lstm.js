class LSTMStockPredictor {
    constructor() {
        this.model = null;
        this.history = null;
        this.onEpochEnd = null;
    }

    buildModel(inputShape) {
        this.model = tf.sequential();
        
        // First LSTM layer with input dropout
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            inputShape: inputShape,
            dropout: 0.2,
            recurrentDropout: 0.2
        }));
        
        // Second LSTM layer
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            dropout: 0.2,
            recurrentDropout: 0.2
        }));
        
        // Third LSTM layer
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: false,
            dropout: 0.1,
            recurrentDropout: 0.1
        }));
        
        // Dense layers with regularization
        this.model.add(tf.layers.dense({ 
            units: 32, 
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001})
        }));
        
        this.model.add(tf.layers.dropout({rate: 0.3}));
        
        this.model.add(tf.layers.dense({ 
            units: 16, 
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001})
        }));
        
        // Output layer
        this.model.add(tf.layers.dense({ units: 1 }));
        
        // Use a more stable optimizer configuration
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        console.log('Model built successfully');
        console.log('Model summary:');
        this.model.summary();
        
        return this.model;
    }

    async trainModel(X_train, y_train, X_test, y_test, epochs = 40) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        // Validate tensors before training
        this.validateTensors(X_train, y_train, X_test, y_test);

        console.log('Starting training...');
        console.log(`Training data - X: ${X_train.shape}, y: ${y_train.shape}`);
        console.log(`Validation data - X: ${X_test.shape}, y: ${y_test.shape}`);

        try {
            this.history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: 32,
                validationData: [X_test, y_test],
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        // Add small epsilon to prevent NaN
                        if (logs.loss !== undefined) logs.loss = Math.max(1e-8, logs.loss);
                        if (logs.val_loss !== undefined) logs.val_loss = Math.max(1e-8, logs.val_loss);
                        
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(6)}, val_loss = ${logs.val_loss.toFixed(6)}`);
                        
                        if (this.onEpochEnd) {
                            this.onEpochEnd(epoch, logs);
                        }
                    }
                }
            });

            return this.history;
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        }
    }

    validateTensors(...tensors) {
        tensors.forEach((tensor, index) => {
            if (!tensor || tensor.isDisposed) {
                throw new Error(`Tensor at index ${index} is invalid or disposed`);
            }
            
            const data = tensor.dataSync();
            if (data.some(val => !isFinite(val))) {
                throw new Error(`Tensor at index ${index} contains NaN or infinite values`);
            }
        });
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not trained');
        }
        
        // Validate input tensor
        this.validateTensors(X);
        
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
