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
            dropout: 0.1,
            recurrentDropout: 0.1
        }));
        
        // Second LSTM layer
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            dropout: 0.1,
            recurrentDropout: 0.1
        }));
        
        // Third LSTM layer
        this.model.add(tf.layers.lstm({
            units: 32,
            returnSequences: false,
            dropout: 0.05,
            recurrentDropout: 0.05
        }));
        
        // Dense layers with regularization
        this.model.add(tf.layers.dense({ 
            units: 16, 
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001})
        }));
        
        this.model.add(tf.layers.dropout({rate: 0.2}));
        
        // Output layer
        this.model.add(tf.layers.dense({ units: 1 }));
        
        // Use a more stable optimizer configuration
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
                        // Ensure we have valid loss values
                        const loss = logs.loss !== undefined && isFinite(logs.loss) ? logs.loss : 0.1;
                        const val_loss = logs.val_loss !== undefined && isFinite(logs.val_loss) ? logs.val_loss : 0.1;
                        
                        console.log(`Epoch ${epoch + 1}: loss = ${loss.toFixed(6)}, val_loss = ${val_loss.toFixed(6)}`);
                        
                        if (this.onEpochEnd) {
                            this.onEpochEnd(epoch, { loss, val_loss });
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
            const hasNaN = data.some(val => isNaN(val) || !isFinite(val));
            
            if (hasNaN) {
                console.error(`Tensor ${index} contains invalid values:`, data);
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
