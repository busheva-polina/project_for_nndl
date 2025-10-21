class LSTMModel {
    constructor(inputShape) {
        this.model = null;
        this.inputShape = inputShape;
        this.history = {
            loss: [],
            val_loss: []
        };
        this.bestValLoss = Infinity;
        this.patienceCounter = 0;
        this.maxPatience = 5;
    }

    buildModel() {
        this.model = tf.sequential();
        
        // First LSTM layer with dropout and recurrent dropout
        this.model.add(tf.layers.lstm({
            units: 64,
            returnSequences: true,
            inputShape: this.inputShape,
            dropout: 0.2,
            recurrentDropout: 0.2
        }));
        
        // Second LSTM layer with dropout
        this.model.add(tf.layers.lstm({
            units: 32,
            returnSequences: false,
            dropout: 0.1
        }));
        
        // Batch normalization for better convergence
        this.model.add(tf.layers.batchNormalization());
        
        // Dense layers with regularization
        this.model.add(tf.layers.dense({
            units: 16, 
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001})
        }));
        
        this.model.add(tf.layers.dropout({rate: 0.3}));
        
        this.model.add(tf.layers.dense({
            units: 1, 
            activation: 'linear'
        }));

        // Compile model with lower learning rate
        this.model.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 20, batchSize = 32) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        // Clear previous history
        this.history = { loss: [], val_loss: [] };
        this.bestValLoss = Infinity;
        this.patienceCounter = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            const history = await this.model.fit(X_train, y_train, {
                epochs: 1,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                verbose: 0
            });

            const loss = history.history.loss[0];
            const val_loss = history.history.val_loss[0];
            
            this.history.loss.push(loss);
            this.history.val_loss.push(val_loss);
            
            // Early stopping check
            if (val_loss < this.bestValLoss) {
                this.bestValLoss = val_loss;
                this.patienceCounter = 0;
            } else {
                this.patienceCounter++;
            }
            
            // Update progress
            if (typeof this.onEpochEnd === 'function') {
                this.onEpochEnd(epoch, { loss, val_loss });
            }
            
            // Early stopping
            if (this.patienceCounter >= this.maxPatience) {
                console.log(`Early stopping at epoch ${epoch + 1}`);
                break;
            }
        }
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
