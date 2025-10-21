import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

export class DataLoader {
  constructor() {
    this.data = null;
    this.trainData = null;
    this.testData = null;
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
          this.data = this.parseCSV(csvText);
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
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(';');
      if (values.length !== headers.length) continue;
      
      const row = {};
      headers.forEach((header, index) => {
        let value = values[index].trim();
        // Replace comma with dot for decimal numbers and handle #N/A
        value = value.replace(',', '.');
        if (value === '#N/A') {
          row[header] = null;
        } else {
          row[header] = parseFloat(value);
        }
      });
      
      // Only add rows with valid numbers
      if (Object.values(row).every(val => val !== null && !isNaN(val))) {
        data.push(row);
      }
    }
    
    return data;
  }

  prepareSequences() {
    if (!this.data || this.data.length === 0) {
      throw new Error('No data available. Please load CSV first.');
    }

    const sequences = [];
    const targets = [];
    
    // Create sequences and targets
    for (let i = 0; i < this.data.length - this.sequenceLength; i++) {
      const sequence = [];
      for (let j = 0; j < this.sequenceLength; j++) {
        const row = this.data[i + j];
        const features = this.featureColumns.map(col => row[col]);
        sequence.push(features);
      }
      
      const targetRow = this.data[i + this.sequenceLength];
      sequences.push(sequence);
      targets.push(targetRow[this.targetColumn]);
    }

    // Split into train/test (80/20 split)
    const splitIndex = Math.floor(sequences.length * 0.8);
    
    const X_train = sequences.slice(0, splitIndex);
    const y_train = targets.slice(0, splitIndex);
    const X_test = sequences.slice(splitIndex);
    const y_test = targets.slice(splitIndex);

    // Convert to tensors
    const X_train_tensor = tf.tensor3d(X_train);
    const y_train_tensor = tf.tensor1d(y_train);
    const X_test_tensor = tf.tensor3d(X_test);
    const y_test_tensor = tf.tensor1d(y_test);

    return {
      X_train: X_train_tensor,
      y_train: y_train_tensor,
      X_test: X_test_tensor,
      y_test: y_test_tensor,
      trainSize: X_train.length,
      testSize: X_test.length
    };
  }

  dispose() {
    if (this.trainData) {
      Object.values(this.trainData).forEach(tensor => {
        if (tensor instanceof tf.Tensor) tensor.dispose();
      });
    }
    if (this.testData) {
      Object.values(this.testData).forEach(tensor => {
        if (tensor instanceof tf.Tensor) tensor.dispose();
      });
    }
  }
}
