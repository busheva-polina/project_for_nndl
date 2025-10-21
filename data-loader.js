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
        if (!headers.includes('Date') || !headers.includes('WTI')) {
            throw new Error('CSV must contain Date and WTI columns');
        }

        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                const value = parseFloat(values[index]);
                row[header] = isNaN(value) ? values[index] : value;
            });
            
            // Only add rows with valid WTI data
            if (row[this.target] !== undefined && !isNaN(row[this.target])) {
                parsedData.push(row);
            }
        }

        if (parsedData.length === 0) {
            throw new Error('No valid data found in CSV');
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

        // Extract values and handle missing data
        const values = this.data.map(row => 
            featureColumns.map(col => row[col] || 0)
        );
        const targets = this.data.map(row => row[targetColumn]);

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

        // Split chronologically
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequences.slice(0, splitIndex);
        const y_train = labels.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_test = labels.slice(splitIndex);

        // Convert to tensors
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
            
            // Calculate min/max for each feature
            for (let col = 0; col < data[0].length; col++) {
                const column = data.map(row => row[col]);
                mins.push(Math.min(...column));
                maxs.push(Math.max(...column));
            }

            // Normalize each feature
            for (let i = 0; i < data.length; i++) {
                const normalizedRow = [];
                for (let col = 0; col < data[i].length; col++) {
                    const min = mins[col];
                    const max = maxs[col];
                    const range = max - min;
                    normalizedRow.push(range === 0 ? 0 : (data[i][col] - min) / range);
                }
                normalized.push(normalizedRow);
            }

            return { normalized, min: mins, max: maxs };
        } else {
            // 1D array (target)
            const min = Math.min(...data);
            const max = Math.max(...data);
            const range = max - min;
            const normalized = data.map(val => range === 0 ? 0 : (val - min) / range);
            return { normalized, min, max };
        }
    }

    dispose() {
        this.data = null;
        this.features = null;
    }
}
