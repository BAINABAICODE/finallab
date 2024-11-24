import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, registerables } from "chart.js";

ChartJS.register(...registerables);

const SalesForecast = () => {
  const [data, setData] = useState([]);
  const [processedData, setProcessedData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        complete: (result) => {
          // Parse and format the data
          const parsedData = result.data
            .map((entry) => ({
              sales_date: entry.created?.split(" ")[0].slice(0, 7), // Extract YYYY-MM
              product_description: entry.short_desc,
              quantity_sold: parseFloat(entry.total_sold),
            }))
            .filter(
              (entry) =>
                entry.sales_date &&
                entry.product_description &&
                !isNaN(entry.quantity_sold)
            ); // Filter out invalid rows
          setData(parsedData);
        },
        header: true,
        skipEmptyLines: true,
      });
    }
  };

  const preprocessData = () => {
    const uniqueProducts = [...new Set(data.map((d) => d.product_description))];
    const productMap = uniqueProducts.reduce(
      (map, product, index) => ({ ...map, [product]: index }),
      {}
    );

    const processed = data.map((entry) => ({
      sales_date: new Date(entry.sales_date).getMonth() + 1, // Convert month to numerical
      product_description: productMap[entry.product_description],
      quantity_sold: entry.quantity_sold,
    }));

    setProcessedData(processed);
    return { processed, productMap };
  };

  const trainAndPredict = async () => {
    if (data.length === 0) {
      alert("Please upload valid sales data!");
      return;
    }

    setLoading(true);
    const { processed, productMap } = preprocessData();

    const salesDate = processed.map((d) => d.sales_date);
    const productDesc = processed.map((d) => d.product_description);
    const quantitySold = processed.map((d) => d.quantity_sold);

    // Prepare tensors
    const salesDateTensor = tf.tensor1d(salesDate);
    const productDescTensor = tf.tensor1d(productDesc);
    const quantitySoldTensor = tf.tensor1d(quantitySold);

    const inputs = tf.stack([salesDateTensor, productDescTensor], 1);
    const outputs = quantitySoldTensor;

    // Define model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, inputShape: [2], activation: "relu" }));
    model.add(tf.layers.dense({ units: 1 }));

    // Compile and train model
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    await model.fit(inputs, outputs, { epochs: 100 });

    // Forecasting
    const lastMonth = Math.max(...salesDate);
    const nextMonths = Array.from({ length: 6 }, (_, i) => lastMonth + i + 1);
    const productIds = Object.values(productMap);

    const predictions = [];
    for (const productId of productIds) {
      const inputTensor = tf.tensor2d(nextMonths.map((month) => [month, productId]));
      const forecast = model.predict(inputTensor);
      const predictedQuantities = forecast.dataSync();
      predictions.push({ product: productId, predictedQuantities });
    }

    setPredictions(predictions);
    setLoading(false);
  };

  const chartData = {
    labels: [
      ...data.map((d) => d.sales_date),
      ...Array.from({ length: 6 }, (_, i) => `Future Month ${i + 1}`),
    ],
    datasets: predictions.flatMap(({ product, predictedQuantities }) => [
      {
        label: `Product ${product} - Actual`,
        data: data
          .filter((d) => d.product_description === product)
          .map((d) => d.quantity_sold),
        borderColor: "blue",
        backgroundColor: "rgba(0, 0, 255, 0.1)",
        borderWidth: 2,
      },
      {
        label: `Product ${product} - Predicted`,
        data: [
          ...Array(data.filter((d) => d.product_description === product).length).fill(null),
          ...predictedQuantities,
        ],
        borderColor: "red",
        backgroundColor: "rgba(255, 0, 0, 0.1)",
        borderWidth: 2,
      },
    ]),
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Sales Forecasting</h1>
      <div style={{ marginBottom: "20px" }}>
        <input type="file" accept=".csv" onChange={handleFileUpload} />
        <button onClick={trainAndPredict} style={{ marginLeft: "10px" }}>
          Train & Predict
        </button>
      </div>
      {loading && <p>Training the model... Please wait.</p>}
      {!loading && processedData.length > 0 && (
        <Line
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { position: "top" },
              title: { display: true, text: "Sales Forecasting Chart" },
            },
            scales: {
              x: { title: { display: true, text: "Months" } },
              y: { title: { display: true, text: "Quantity Sold" } },
            },
          }}
        />
      )}
    </div>
  );
};

export default SalesForecast;
