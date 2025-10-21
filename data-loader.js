<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Prediction with TensorFlow.js</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .control-panel { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .chart-container { position: relative; height: 400px; margin-bottom: 20px; }
        button { padding: 8px 16px; margin: 5px; cursor: pointer; }
        .progress { margin: 10px 0; }
        progress { width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Price Prediction with GRU Model</h1>
        
        <div class="control-panel">
            <h3>Data Input</h3>
            <input type="file" id="csvFile" accept=".csv">
            <button id="loadData">Load Data</button>
            
            <h3>Model Training</h3>
            <button id="trainModel">Train Model</button>
            <button id="predict">Predict</button>
            <button id="saveModel">Save Model</button>
            <button id="loadModel">Load Model</button>
            
            <div class="progress">
                <label>Training Progress:</label>
                <progress id="trainingProgress" value="0" max="100"></progress>
                <span id="progressText">0%</span>
            </div>
            
            <div id="trainingInfo"></div>
        </div>

        <div class="chart-container">
            <canvas id="lossChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="predictionChart"></canvas>
        </div>
    </div>

    <script type="module" src="data-loader.js"></script>
    <script type="module" src="gru.js"></script>
    <script type="module" src="app.js"></script>
</body>
</html>
