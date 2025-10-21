class DataLoader {
    constructor() {
        this.data = null;
        this.features = null;
        this.target = 'WTI';
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
        
        // Validate required columns
        const requiredColumns = ['Date', 'WTI'];
        requiredColumns.forEach(col => {
            if (!headers.includes(col)) {
                throw new Error(`Missing required column: ${col}`);
            }
        });

        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                const value = parseFloat(values[index]);
                row[header] = isNaN(value) ? values[index] : value;
            });
            parsedData.push(row);
        }

        this.data = parsedData;
        this.features = headers.filter(h => h !== 'Date' && h !== this.target);
    }

    prepareSequences() {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data loaded');
        }

        const featureColumns = this.features;
        const targetColumn = this.target;

        // Convert to tensors
        const values = this.data.map(row => 
            featureColumns.map(col => row[col] || 0)
        );
        const targets = this.data.map(row => row[targetColumn] || 0);

        // Normalize data
        const { normalized: normalizedValues, min: valueMin, max: valueMax } = 
            this.minMaxNormalize(values);
        const { normalized: normalizedTargets, min: targetMin, max: targetMax } = 
            this.minMaxNormalize(targets);

        // Create sequences
        const sequences = [];
        const labels = [];

        for (let i = this.sequenceLength; i < normalizedValues.length; i++) {
            sequences.push(normalizedValues.slice(i - this.sequenceLength, i));
            labels.push(normalizedTargets[i]);
        }

        // Split into train/test
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequences.slice(0, splitIndex);
        const y_train = labels.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_test = labels.slice(splitIndex);

        return {
            X_train: tf.tensor3d(X_train),
            y_train: tf.tensor1d(y_train),
            X_test: tf.tensor3d(X_test),
            y_test: tf.tensor1d(y_test),
            normalization: { valueMin, valueMax, targetMin, targetMax },
            featureNames: featureColumns
        };
    }

    minMaxNormalize(data) {
        if (Array.isArray(data[0])) {
            // 2D array (features)
            const mins = [];
            const maxs = [];
            const normalized = [];
            
            for (let col = 0; col < data[0].length; col++) {
                const column = data.map(row => row[col]);
                mins.push(Math.min(...column));
                maxs.push(Math.max(...column));
            }

            for (let i = 0; i < data.length; i++) {
                const normalizedRow = [];
                for (let col = 0; col < data[i].length; col++) {
                    const min = mins[col];
                    const max = maxs[col];
                    normalizedRow.push((data[i][col] - min) / (max - min));
                }
                normalized.push(normalizedRow);
            }

            return { normalized, min: mins, max: maxs };
        } else {
            // 1D array (target)
            const min = Math.min(...data);
            const max = Math.max(...data);
            const normalized = data.map(val => (val - min) / (max - min));
            return { normalized, min, max };
        }
    }

    dispose() {
        // Cleanup if needed
    }
}
