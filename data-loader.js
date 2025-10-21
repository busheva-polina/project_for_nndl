import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';

export class DataLoader {
    constructor() {
        this.data = null;
        this.sequenceLength = 30;
        this.featureColumns = ['WTI', 'US Dollar Index Futures', 'Gold Futures'];
        this.targetColumn = 'WTI';
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
                    const csvText = e.target.result;
                    this.parseCSV(csvText);
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
        const requiredColumns = [...this.featureColumns, this.targetColumn];
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        if (missingColumns.length > 0) {
            throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
        }

        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => parseFloat(v.trim()));
            if (values.some(isNaN)) continue; // Skip invalid rows
            
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            parsedData.push(row);
        }

        if (parsedData.length < this.sequenceLength + 1) {
            throw new Error('Insufficient data for sequence length');
        }

        this.data = parsedData;
    }

    prepareSequences() {
        if (!this.data) {
            throw new Error('No data loaded. Call loadCSV first.');
        }

        const sequences = [];
        const targets = [];

        // Extract feature values
        const featureValues = this.featureColumns.map(col => 
            this.data.map(row => row[col])
        );

        // Extract target values
        const targetValues = this.data.map(row => row[this.targetColumn]);

        // Create sequences and targets
        for (let i = 0; i < this.data.length - this.sequenceLength; i++) {
            const sequence = [];
            for (let j = 0; j < this.sequenceLength; j++) {
                const features = this.featureColumns.map(col => 
                    this.data[i + j][col]
                );
                sequence.push(features);
            }
            sequences.push(sequence);
            targets.push(targetValues[i + this.sequenceLength]);
        }

        // Convert to tensors
        const sequencesTensor = tf.tensor3d(sequences);
        const targetsTensor = tf.tensor1d(targets);

        // Split into train/test
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        
        const X_train = sequencesTensor.slice([0, 0, 0], [splitIndex, -1, -1]);
        const X_test = sequencesTensor.slice([splitIndex, 0, 0], [-1, -1, -1]);
        const y_train = targetsTensor.slice([0], [splitIndex]);
        const y_test = targetsTensor.slice([splitIndex], [-1]);

        // Clean up intermediate tensors
        sequencesTensor.dispose();
        targetsTensor.dispose();

        return {
            X_train,
            y_train,
            X_test,
            y_test,
            trainSize: splitIndex,
            testSize: sequences.length - splitIndex
        };
    }

    dispose() {
        // Cleanup method for tensor disposal
        if (this.dataTensors) {
            Object.values(this.dataTensors).forEach(tensor => {
                if (tensor && tensor.dispose) tensor.dispose();
            });
        }
    }
}
