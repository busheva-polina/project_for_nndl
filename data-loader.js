class DataLoader {
    constructor() {
        this.data = null;
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.scalers = {};
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
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
        const headers = lines[0].split(';').map(h => h.trim());
        
        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            const values = line.split(';');
            const row = {};
            
            // Handle European decimal format (comma as decimal separator)
            for (let j = 0; j < headers.length; j++) {
                let value = values[j].trim();
                if (value === '#N/A' || value === '' || value === undefined) {
                    value = NaN;
                } else {
                    value = parseFloat(value.replace(',', '.'));
                }
                row[headers[j]] = value;
            }
            
            // Only add rows with valid data
            if (!isNaN(row.WTI) && !isNaN(row.GOLD) && !isNaN(row['US DOLLAR INDEX'])) {
                parsedData.push(row);
            }
        }

        this.data = parsedData;
        console.log(`Loaded ${this.data.length} valid records`);
    }

    prepareData(sequenceLength = 30, testSplit = 0.2) {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data loaded. Please load CSV file first.');
        }

        // Extract features: WTI, GOLD, US DOLLAR INDEX
        const features = ['WTI', 'GOLD', 'US DOLLAR INDEX'];
        const featureData = features.map(feature => 
            this.data.map(row => row[feature]).filter(val => !isNaN(val))
        );

        // Normalize data
        const normalizedData = this.normalizeData(featureData);

        // Create sequences
        const sequences = [];
        const targets = [];

        for (let i = 0; i < normalizedData[0].length - sequenceLength; i++) {
            const sequence = [];
            for (let j = 0; j < sequenceLength; j++) {
                const timeStep = [];
                for (let k = 0; k < features.length; k++) {
                    timeStep.push(normalizedData[k][i + j]);
                }
                sequence.push(timeStep);
            }
            sequences.push(sequence);
            
            // Target: next day's WTI price
            targets.push(normalizedData[0][i + sequenceLength]);
        }

        // Split data chronologically
        const splitIndex = Math.floor(sequences.length * (1 - testSplit));
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor1d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor1d(targets.slice(splitIndex));

        console.log(`Training sequences: ${this.X_train.shape[0]}`);
        console.log(`Test sequences: ${this.X_test.shape[0]}`);
    }

    normalizeData(featureData) {
        const normalized = [];
        this.scalers = {};

        for (let i = 0; i < featureData.length; i++) {
            const values = featureData[i];
            const min = Math.min(...values);
            const max = Math.max(...values);
            
            this.scalers[i] = { min, max };
            normalized.push(values.map(val => (val - min) / (max - min)));
        }

        return normalized;
    }

    denormalize(normalizedValues, featureIndex = 0) {
        const scaler = this.scalers[featureIndex];
        return normalizedValues.map(val => val * (scaler.max - scaler.min) + scaler.min);
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}

// Export for use in other modules
window.DataLoader = DataLoader;
