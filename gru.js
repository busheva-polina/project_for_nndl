class GRUModel {
    constructor() {
        this.model = null;
        this.history = {
            loss: [],
            val_loss: [],
            mse: [],
            val_mse: []
        };
    }

    buildModel(inputShape, featureNames) {
        this.model = tf.sequential();
        
        // First GRU layer
        this.model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: inputShape
        }));
        
        // Second GRU layer
        this.model.add(tf.layers.gru({
            units: 32,
            returnSequences: false
        }));
        
        // Dropout for regularization
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer
        this.model.add(tf.layers.dense({ units: 1 }));
        
        // Compile model
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        // Calculate feature importance (simplified correlation-based)
        this.featureImportance = this.calculateFeatureImportance(featureNames);
        
        return this.model;
    }

    calculateFeatureImportance(featureNames) {
        // Simplified feature importance based on domain knowledge
        // In a real implementation, this would use actual correlation analysis
        const importance = {
            'Gold Futures': 0.8,
            'US Dollar Index Futures': 0.9,
            'US 10 Year Bond Yield': 0.7,
            'S&P 500': 0.6,
            'Dow Jones Utility Average': 0.5
        };
        
        return Object.entries(importance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3); // Top 3 features
    }

    async train(X_train, y_train, X_test, y_test, epochs = 40) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.history = { loss: [], val_loss: [], mse: [], val_mse: [] };

        await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.history.loss.push(logs.loss);
                    this.history.val_loss.push(logs.val_loss);
                    this.history.mse.push(logs.mse);
                    this.history.val_mse.push(logs.val_mse);
                    
                    // Memory cleanup
                    tf.tidy(() => {
                        // Cleanup temporary tensors
                    });
                    
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
            throw new Error('Model not trained.');
        }
        return this.model.predict(X);
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    getTrainingHistory() {
        return this.history;
    }

    getFeatureImportance() {
        return this.featureImportance;
    }
}
