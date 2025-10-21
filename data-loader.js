import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';

export class DataLoader {
    constructor() {
        this.data = [];
        this.trainData = [];
        this.testData = [];
        this.sequenceLength = 30;
        this.featureColumns = ['WTI', 'GOLD', 'US DOLLAR INDEX'];
        this.targetColumn = 'WTI';
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const csvText = e.target.result;
                    this.parseCSV(csvText);
                    this.normalizeData();
                    this.createSequences();
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
        const headers = lines[0].split(';');
        
        this.data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(';');
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                let value = values[index].trim();
                
                // Handle European decimal format (comma to dot)
                if (value.includes(',')) {
                    value = value.replace(',', '.');
                }
                
                // Handle missing values
                if (value === '#N/A' || value === '' || isNaN(parseFloat(value))) {
                    value = null;
                } else {
                    value = parseFloat(value);
                }
                
                row[header.trim()] = value;
            });
            
            // Only add rows with valid data for all required columns
            const hasValidData = this.featureColumns.every(col => 
                row[col] !== null && !isNaN(row[col])
            ) && row[this.targetColumn] !== null;
            
            if (hasValidData) {
                this.data.push(row);
            }
        }
        
        console.log(`Loaded ${this.data.length} valid records`);
    }

    normalizeData() {
        if (this.data.length === 0) return;

        this.normalizationParams = {};
        
        this.featureColumns.forEach(col => {
            const values = this.data.map(row => row[col]).filter(v => v !== null);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
            
            this.normalizationParams[col] = { mean, std };
            
            // Normalize the data
            this.data.forEach(row => {
                if (row[col] !== null) {
                    row[`${col}_normalized`] = (row[col] - mean) / std;
                }
            });
        });
        
        // Normalize target separately
        const targetValues = this.data.map(row => row[this.targetColumn]);
        const targetMean = targetValues.reduce((a, b) => a + b, 0) / targetValues.length;
        const targetStd = Math.sqrt(targetValues.reduce((a, b) => a + Math.pow(b - targetMean, 2), 0) / targetValues.length);
        
        this.normalizationParams[this.targetColumn] = { mean: targetMean, std: targetStd };
        
        this.data.forEach(row => {
            row[`${this.targetColumn}_normalized`] = (row[this.targetColumn] - targetMean) / targetStd;
        });
    }

    createSequences() {
        const sequences = [];
        const targets = [];
        
        for (let i = this.sequenceLength; i < this.data.length; i++) {
            const sequence = [];
            let validSequence = true;
            
            for (let j = i - this.sequenceLength; j < i; j++) {
                const features = this.featureColumns.map(col => {
                    const value = this.data[j][`${col}_normalized`];
                    if (value === null || isNaN(value)) {
                        validSequence = false;
                    }
                    return value;
                });
                
                if (validSequence) {
                    sequence.push(features);
                }
            }
            
            if (validSequence && this.data[i][`${this.targetColumn}_normalized`] !== null) {
                sequences.push(sequence);
                targets.push([this.data[i][`${this.targetColumn}_normalized`]]);
            }
        }
        
        // Split data (80% train, 20% test)
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.trainData = {
            sequences: sequences.slice(0, splitIndex),
            targets: targets.slice(0, splitIndex)
        };
        
        this.testData = {
            sequences: sequences.slice(splitIndex),
            targets: targets.slice(splitIndex)
        };
        
        console.log(`Created ${sequences.length} sequences (${this.trainData.sequences.length} train, ${this.testData.sequences.length} test)`);
    }

    getTrainTensors() {
        if (this.trainData.sequences.length === 0) {
            throw new Error('No training data available');
        }

        const X = tf.tensor3d(this.trainData.sequences);
        const y = tf.tensor2d(this.trainData.targets);
        
        return { X, y };
    }

    getTestTensors() {
        if (this.testData.sequences.length === 0) {
            throw new Error('No test data available');
        }

        const X = tf.tensor3d(this.testData.sequences);
        const y = tf.tensor2d(this.testData.targets);
        
        return { X, y };
    }

    denormalizeTarget(normalizedValues) {
        const params = this.normalizationParams[this.targetColumn];
        if (!params) return normalizedValues;
        
        if (normalizedValues instanceof tf.Tensor) {
            return normalizedValues.mul(params.std).add(params.mean);
        }
        
        return normalizedValues.map(val => val * params.std + params.mean);
    }

    dispose() {
        // Cleanup if needed
    }
}
