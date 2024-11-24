import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";

const PurchaseForecast = () => {
  const [data, setData] = useState([
    { date: "2024-11-01", quantity: 120 },
    { date: "2024-11-02", quantity: 25 },
    { date: "2024-11-03", quantity: 27 },
    { date: "2024-11-04", quantity: 330 },
    { date: "2024-11-05", quantity: 335 },
    { date: "2024-11-07", quantity: 27 },
    { date: "2024-11-08", quantity: 330 },
    { date: "2024-11-09", quantity: 352 },
  ]); // Example sales data
  const [prediction, setPrediction] = useState(null);

  // Prepare data for training
  const prepareData = () => {
    const quantities = data.map((entry) => entry.quantity);
    const xs = tf.tensor1d(quantities.slice(0, -1)); // All except last input
    const ys = tf.tensor1d(quantities.slice(1)); // All except first output for training
    return { xs, ys };
  };

const checkForInvalidData = (tensor) => {
    const data = tensor.arraySync(); // Convert tensor to a JavaScript array
    return data.some(value => !isFinite(value) || isNaN(value));
};
// Train a simple linear regression model
const trainModel = async () => {
    const { xs, ys } = prepareData();

    // Ensure xs and ys are finite and properly scaled before training
    if (checkForInvalidData(xs) || checkForInvalidData(ys)) {
        throw new Error("Invalid data: xs or ys contain non-finite or NaN values.");
    }
    
    // Define the model
    const model = tf.sequential();
    //inputShape: [1] indicates a single input tensor with one dimension
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    // Use a learning rate of 0.01 for fine-tuning the model. The mean squared error loss function is suitable for regression tasks.
    model.compile({ optimizer: tf.train.adam(0.01), loss: "meanSquaredError" });

    // Train the model
    await model.fit(xs, ys, {
        epochs: 110, // Combine the epochs from both fit calls
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: Loss = ${logs.loss}`);
                if (logs.loss > 1e10) {
                    console.log("Loss too high, stopping training.");
                    model.stopTraining = true;
                }
            },
        },
    });

    return model;
};

  // Make predictions
  const predictNextQuantity = async () => {
    const model = await trainModel();

    const lastQuantity = data[data.length - 1].quantity;
    const inputTensor = tf.tensor1d([lastQuantity]); // Use the last quantity sold as input

    const outputTensor = model.predict(inputTensor);
    const predictionValue = outputTensor.dataSync()[0];

    setPrediction(predictionValue.toFixed(2));
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>Sales Prediction</h1>
      <h2>Past Sales Data</h2>
      <table border="1" style={{ margin: "0 auto", width: "50%" }}>
        <thead>
          <tr>
            <th>Date</th>
            <th>Quantity Sold</th>
          </tr>
        </thead>
        <tbody>
          {data.map((entry, index) => (
            <tr key={index}>
              <td>{entry.date}</td>
              <td>{entry.quantity}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={predictNextQuantity} style={{ marginTop: "20px", padding: "10px 20px" }}>
        Predict Next Quantity
      </button>
      {prediction && (
        <h2 style={{ marginTop: "20px" }}>
          Predicted Quantity for Next Purchase: {prediction}
        </h2>
      )}
    </div>
  );
};

export default PurchaseForecast;
