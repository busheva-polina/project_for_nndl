class DataLoader {
    constructor() {
        this.data = null;
        this.trainSize = 0.8;
        this.sequenceLength = 20;
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

        const featureColumns = [
            'WTI', 'Gold Futures', 'US Dollar Index Futures', 
            'US 10 Year Bond Yield', 'S&P 500', 'Dow Jones Utility Average'
        ];
        
        const features = this.data.map(row => featureColumns.map(col => row[col]));
        const target = this.data.map(row => row['WTI']);

        const scaledFeatures = this.scaleFeatures(features, featureColumns);
        const scaledTarget = this.scaleValues(target, 'WTI');

        const { X, y } = this.createSequences(scaledFeatures, scaledTarget);
        
        const splitIndex = Math.floor(X.shape[0] * this.trainSize);
        
        const X_train = X.slice(0, splitIndex);
        const y_train = y.slice(0, splitIndex);
        const X_test = X.slice(splitIndex);
        const y_test = y.slice(splitIndex);

        return {
            X_train, y_train, X_test, y_test,
            featureNames: featureColumns,
            scalers: this.scalers
        };
    }

    scaleFeatures(features, columnNames) {
        const scaled = [];
        for (let i = 0; i < features[0].length; i++) {
            const column = features.map(row => row[i]);
            const scaler = this.createScaler(column);
            this.scalers[columnNames[i]] = scaler;
            
            const scaledColumn = this.scaleValues(column, columnNames[i]);
            scaled.push(scaledColumn);
        }

        return scaled[0].map((_, i) => scaled.map(col => col[i]));
    }

    createScaler(values) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        return { min, max, range: max - min || 1 };
    }

    scaleValues(values, columnName) {
        const scaler = this.scalers[columnName] || this.createScaler(values);
        if (!this.scalers[columnName]) {
            this.scalers[columnName] = scaler;
        }
        
        return values.map(v => (v - scaler.min) / scaler.range);
    }

    unscaleValues(scaledValues, columnName) {
        const scaler = this.scalers[columnName];
        if (!scaler) return scaledValues;
        
        return scaledValues.map(v => v * scaler.range + scaler.min);
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
