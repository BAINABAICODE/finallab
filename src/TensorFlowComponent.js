import React, { useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const TensorFlowComponent = () => {
  useEffect(() => {
    // Perform a basic TensorFlow.js operation
    const a = tf.tensor([1, 2, 3, 4]);
    const b = tf.tensor([2, 3, 4, 5]);
    const result = a.add(b);

    // Print the result
    result.print();
  }, []);

  return (
    <div>
      <h1>TensorFlow.js Example</h1>
      <p>Check the console for the result of the TensorFlow operation.</p>
    </div>
  );
};

export default TensorFlowComponent;
