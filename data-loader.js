class DataLoader {
    constructor() {
        this.data = null;
        this.trainSize = 0.8;
        this.sequenceLength = 30;
        this.scalers = {};
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
                    const csv = e.target.result;
                    this.data = this.parseCSV(csv);
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
        
        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;
            
            const row = {};
            headers.forEach((header, index) => {
                const value = parseFloat(values[index]);
                row[header] = isNaN(value) ? 0 : value;
            });
            data.push(row);
        }
        
        return data;
    }

    prepareData() {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data loaded');
        }

        // Remove any rows with invalid data
        const cleanData = this.data.filter(row => {
            return Object.values(row).every(val => !isNaN(val) && isFinite(val));
        });

        const featureColumns = [
            'WTI', 'Gold Futures', 'US Dollar Index Futures', 
            'US 10 Year Bond Yield', 'S&P 500', 'Dow Jones Utility Average'
        ];
        
        const features = cleanData.map(row => featureColumns.map(col => row[col]));
        const target = cleanData.map(row => row['WTI']);

        // Robust scaling with epsilon to avoid division by zero
        const scaledFeatures = this.robustScaleFeatures(features, featureColumns);
        const scaledTarget = this.robustScaleValues(target, 'WTI');

        const { X, y } = this.createSequences(scaledFeatures, scaledTarget);
        
        const splitIndex = Math.floor(X.shape[0] * this.trainSize);
        
        const X_train = X.slice(0, splitIndex);
        const y_train = y.slice(0, splitIndex);
        const X_test = X.slice(splitIndex);
        const y_test = y.slice(splitIndex);

        console.log(`Data shapes - X_train: ${X_train.shape}, y_train: ${y_train.shape}`);

        return {
            X_train, y_train, X_test, y_test,
            featureNames: featureColumns,
            scalers: this.scalers,
            originalData: cleanData
        };
    }

    robustScaleFeatures(features, columnNames) {
        const scaled = [];
        for (let i = 0; i < features[0].length; i++) {
            const column = features.map(row => row[i]);
            const scaler = this.createRobustScaler(column);
            this.scalers[columnNames[i]] = scaler;
            
            const scaledColumn = this.robustScaleValues(column, columnNames[i]);
            scaled.push(scaledColumn);
        }

        return scaled[0].map((_, i) => scaled.map(col => col[i]));
    }

    createRobustScaler(values) {
        // Use robust scaling with median and IQR to handle outliers
        const sorted = [...values].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        
        // Use standard deviation as fallback if IQR is too small
        const range = iqr > 0 ? iqr : this.calculateStdDev(values);
        const epsilon = 1e-8; // Prevent division by zero
        
        return { 
            median, 
            range: range || 1,
            epsilon,
            min: Math.min(...values),
            max: Math.max(...values)
        };
    }

    calculateStdDev(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squareDiffs = values.map(value => Math.pow(value - mean, 2));
        const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
        return Math.sqrt(avgSquareDiff);
    }

    robustScaleValues(values, columnName) {
        const scaler = this.scalers[columnName] || this.createRobustScaler(values);
        if (!this.scalers[columnName]) {
            this.scalers[columnName] = scaler;
        }
        
        return values.map(v => {
            const scaled = (v - scaler.median) / (scaler.range + scaler.epsilon);
            // Clip extreme values to prevent NaN
            return Math.max(-10, Math.min(10, scaled));
        });
    }

    unscaleValues(scaledValues, columnName) {
        const scaler = this.scalers[columnName];
        if (!scaler) return scaledValues;
        
        return scaledValues.map(v => v * (scaler.range + scaler.epsilon) + scaler.median);
    }

    createSequences(features, target) {
        const X = [];
        const y = [];
        
        for (let i = this.sequenceLength; i < features.length; i++) {
            X.push(features.slice(i - this.sequenceLength, i));
            y.push(target[i]);
        }
        
        return {
            X: tf.tensor3d(X),
            y: tf.tensor1d(y)
        };
    }

    dispose() {
        this.data = null;
        this.scalers = {};
    }
}
