class DataLoader {
    constructor() {
        this.data = null;
        this.features = ['WTI', 'Gold Futures', 'US Dollar Index Futures'];
        this.sequenceLength = 30;
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

                    console.log('CSV headers:', Object.keys(results.data[0]));
                    console.log('First few rows:', results.data.slice(0, 3));
                    
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
        const cleanData = rawData.filter(row => {
            return this.features.every(feature => {
                const value = row[feature];
                return value !== null && 
                       value !== undefined && 
                       !isNaN(value) &&
                       typeof value === 'number';
            });
        });

        if (cleanData.length === 0) {
            throw new Error('No valid data remaining after cleaning. Check CSV headers match: WTI, Gold Futures, US Dollar Index Futures');
        }

        console.log(`Original data: ${rawData.length} rows, Clean data: ${cleanData.length} rows`);
        console.log('Sample cleaned data:', cleanData.slice(0, 3));
        return cleanData;
    }

    prepareData(sequenceLength = 30) {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data available. Please load CSV file first.');
        }

        if (sequenceLength >= this.data.length) {
            throw new Error('Sequence length is too large for the available data');
        }

        const sequences = [];
        const targets = [];

        for (let i = sequenceLength; i < this.data.length - 1; i++) {
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const featureVector = this.features.map(feature => {
                    const value = this.data[j][feature];
                    if (value === null || value === undefined || isNaN(value)) {
                        throw new Error(`Invalid data at row ${j}, feature ${feature}`);
                    }
                    return value;
                });
                sequence.push(featureVector);
            }
            
            sequences.push(sequence);
            
            const nextDayPrices = this.features.map(feature => this.data[i + 1][feature]);
            const currentPrices = this.features.map(feature => this.data[i][feature]);
            const returns = nextDayPrices.map((price, idx) => (price - currentPrices[idx]) / currentPrices[idx]);
            const binaryLabels = returns.map(ret => ret > 0 ? 1 : 0);
            
            targets.push(binaryLabels);
        }

        if (sequences.length === 0) {
            throw new Error('No sequences created. Check data length and sequence length.');
        }

        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequences.slice(0, splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_test = targets.slice(splitIndex);

        console.log(`Creating tensors: ${X_train.length} training, ${X_test.length} test samples`);
        
        const X_train_tensor = tf.tensor3d(X_train, [X_train.length, sequenceLength, this.features.length]);
        const y_train_tensor = tf.tensor2d(y_train, [y_train.length, this.features.length]);
        const X_test_tensor = tf.tensor3d(X_test, [X_test.length, sequenceLength, this.features.length]);
        const y_test_tensor = tf.tensor2d(y_test, [y_test.length, this.features.length]);

        console.log(`Data prepared: ${X_train.length} training samples, ${X_test.length} test samples`);
        console.log(`X_train shape: ${X_train_tensor.shape}, y_train shape: ${y_train_tensor.shape}`);

        return {
            X_train: X_train_tensor,
            y_train: y_train_tensor,
            X_test: X_test_tensor,
            y_test: y_test_tensor,
            featureNames: this.features,
            stockSymbols: this.features
        };
    }

    dispose() {
        if (this.data) {
            this.data = null;
        }
    }
}

export default DataLoader;
