class DataLoader {
    constructor() {
        this.data = null;
        this.featureColumns = [];
        this.targetColumn = 'WTI';
        this.sequenceLength = 30;
        this.trainTestSplit = 0.8;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            if (!file) {
                reject(new Error('No file provided'));
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csvData = e.target.result;
                    this.parseCSV(csvData);
                    resolve(this.data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Identify feature columns (all except date and target)
        this.featureColumns = headers.filter(h => h !== 'Date' && h !== this.targetColumn);
        
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                row[header] = parseFloat(values[index]) || 0;
            });
            rows.push(row);
        }
        
        this.data = rows;
    }

    preprocessData() {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data loaded');
        }

        // Extract features and target
        const features = this.featureColumns.map(col => 
            this.data.map(row => row[col])
        );
        
        const target = this.data.map(row => row[this.targetColumn]);

        // Normalize data
        const normalizedFeatures = this.normalizeFeatures(features);
        const normalizedTarget = this.normalizeArray(target);

        // Create sequences
        return this.createSequences(normalizedFeatures, normalizedTarget);
    }

    normalizeFeatures(features) {
        return features.map(featureArray => this.normalizeArray(featureArray));
    }

    normalizeArray(array) {
        const min = Math.min(...array);
        const max = Math.max(...array);
        return array.map(val => (val - min) / (max - min));
    }

    createSequences(normalizedFeatures, normalizedTarget) {
        const sequences = [];
        const targets = [];
        
        const totalSamples = normalizedTarget.length - this.sequenceLength;
        
        for (let i = 0; i < totalSamples; i++) {
            const sequence = [];
            for (let j = 0; j < this.sequenceLength; j++) {
                const timeStep = [];
                normalizedFeatures.forEach(feature => {
                    timeStep.push(feature[i + j]);
                });
                sequence.push(timeStep);
            }
            sequences.push(sequence);
            targets.push(normalizedTarget[i + this.sequenceLength]);
        }

        // Split into train/test
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequences.slice(0, splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_test = targets.slice(splitIndex);

        return {
            X_train: tf.tensor3d(X_train),
            y_train: tf.tensor1d(y_train),
            X_test: tf.tensor3d(X_test),
            y_test: tf.tensor1d(y_test),
            featureNames: this.featureColumns
        };
    }

    dispose() {
        if (this.data) {
            this.data = null;
        }
    }
}
