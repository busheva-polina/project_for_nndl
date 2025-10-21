class DataLoader {
    constructor() {
        this.data = null;
        this.features = ['WTI', 'US Dollar Index Futures', 'Gold Futures'];
        this.target = 'WTI';
        this.trainTestSplit = 0.8;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(`CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`));
                        return;
                    }
                    
                    if (results.data.length === 0) {
                        reject(new Error('CSV file is empty or could not be parsed'));
                        return;
                    }
                    
                    this.data = this.preprocessData(results.data);
                    resolve(this.data);
                },
                error: (error) => {
                    reject(new Error(`Failed to parse CSV: ${error.message}`));
                }
            });
        });
    }

    preprocessData(rawData) {
        // Remove rows with missing values in our selected features
        const cleanData = rawData.filter(row => {
            return this.features.every(feature => 
                row[feature] !== null && 
                row[feature] !== undefined && 
                !isNaN(row[feature])
            );
        });

        if (cleanData.length === 0) {
            throw new Error('No valid data remaining after cleaning');
        }

        console.log(`Original data: ${rawData.length} rows, Clean data: ${cleanData.length} rows`);
        return cleanData;
    }

    createSequences(data, sequenceLength) {
        const sequences = [];
        const targets = [];

        for (let i = sequenceLength; i < data.length; i++) {
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const featureVector = this.features.map(feature => data[j][feature]);
                sequence.push(featureVector);
            }
            
            sequences.push(sequence);
            targets.push(data[i][this.target]);
        }

        return { sequences, targets };
    }

    prepareData(sequenceLength = 30) {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data available. Please load CSV file first.');
        }

        if (sequenceLength >= this.data.length) {
            throw new Error('Sequence length is too large for the available data');
        }

        const { sequences, targets } = this.createSequences(this.data, sequenceLength);
        
        // Split chronologically (first 80% train, last 20% test)
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequences.slice(0, splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_test = targets.slice(splitIndex);

        // Convert to tensors
        const X_train_tensor = tf.tensor3d(X_train, [X_train.length, sequenceLength, this.features.length]);
        const y_train_tensor = tf.tensor1d(y_train);
        const X_test_tensor = tf.tensor3d(X_test, [X_test.length, sequenceLength, this.features.length]);
        const y_test_tensor = tf.tensor1d(y_test);

        console.log(`Data prepared: ${X_train.length} training samples, ${X_test.length} test samples`);
        console.log(`Tensor shapes - X_train: ${X_train_tensor.shape}, y_train: ${y_train_tensor.shape}`);

        return {
            X_train: X_train_tensor,
            y_train: y_train_tensor,
            X_test: X_test_tensor,
            y_test: y_test_tensor,
            featureNames: this.features,
            targetName: this.target
        };
    }

    dispose() {
        if (this.data) {
            this.data = null;
        }
    }
}

export default DataLoader;
