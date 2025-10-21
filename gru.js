class GRUModel {
    constructor() {
        this.model = null;
        this.history = {
            loss: [],
            val_loss: []
        };
    }

    buildModel(inputShape) {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 50,
                    returnSequences: true,
                    inputShape: inputShape
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.gru({
                    units: 25,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 1 })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 30) {
        if (!this.model) {
            throw new Error('Model not built. Call buildModel() first.');
        }

        this.history = { loss: [], val_loss: [] };

        await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.history.loss.push(logs.loss);
                    this.history.val_loss.push(logs.val_loss);
                    
                    // Memory cleanup during training
                    tf.engine().startScope();
                    tf.engine().endScope();
                    
                    if (typeof this.onEpochEnd === 'function') {
                        this.onEpochEnd(epoch, logs);
                    }
                },
                onTrainEnd: () => {
                    // Final memory cleanup
                    tf.tidy(() => {});
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
            this.model = null;
        }
    }

    getTrainingHistory() {
        return this.history;
    }
}
