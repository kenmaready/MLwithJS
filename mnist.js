import MultiLogisticRegression from "./MulitLogisticRegression.js";
import * as tf from "@tensorflow/tfjs";
import plot from "node-remote-plot";
import mnist from "mnist-data";
import _ from "lodash";

function loadData(data) {
    const features = data.images.values.map((image) => _.flatMap(image));

    const labels = data.labels.values.map((label) => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });
    return { features, labels };
}

const { features, labels } = loadData(mnist.training(0, 60000));

const model = new MultiLogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 80,
    batchSize: 500,
    scale: false,
});

model.train();
plot({ x: model.crossentropies, title: "my_plot", name: "my_plot" });

const { features: testFeatures, labels: testLabels } = loadData(
    mnist.testing(0, 10000),
);

const accuracy = model.test(testFeatures, testLabels);
console.log("accuracy:", accuracy);
