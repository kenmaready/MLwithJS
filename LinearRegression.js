import * as tf from "@tensorflow/tfjs";
import STDScaler from "./STDScaler.js";

const defaultOptions = { learningRate: 0.1, iterations: 1000 };

export default class LinearRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        this.scaler = new STDScaler();
        this.scaler.fit(features);
        this.features = this.scaler.transform(this.features);
        this.features = this._addOnes(this.features);

        this.labels = tf.tensor(labels);
        this.options = {
            ...defaultOptions,
            ...options,
        };

        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent(features, labels) {
        const currentPredictions = features.matMul(this.weights);
        const errors = currentPredictions.sub(labels);
        const mse = tf.mean(errors.pow(2)).arraySync();

        const gradients = features
            .transpose()
            .matMul(errors)
            .div(features.shape[0]);

        this.weights = this.weights.sub(
            gradients.mul(this.options.learningRate),
        );

        return { mse };
    }

    train(options) {
        options = { ...{ batchSize: this.features.shape[0] }, ...options };
        const numBatches = Math.floor(
            this.features.shape[0] / options.batchSize,
        );
        this.mses = [];
        let mse;
        for (let i = 1; i <= this.options.iterations; i++) {
            for (let j = 0; j < numBatches; j++) {
                const startRow = j * options.batchSize;
                const featureSlice = this.features.slice(
                    [startRow, 0],
                    [options.batchSize, -1],
                );
                const labelSlice = this.labels.slice(
                    [startRow, 0],
                    [options.batchSize, -1],
                );
                mse = this.gradientDescent(featureSlice, labelSlice).mse;
            }

            this.mses.push(mse);
            this._updateLearningRate();
        }

        console.log(
            "Training complete. Current weights:",
            this.weights.arraySync(),
        );
        console.log("mses:", this.mses.slice(this.mses.length - 10));
    }

    test(features, labels) {
        const predictions = this.predict(features);
        labels = tf.tensor(labels);

        const mean = tf.mean(labels);
        const sst = labels.sub(mean).pow(2).sum().arraySync();
        const ssr = labels.sub(predictions).pow(2).sum().arraySync();
        const r2 = 1 - ssr / sst;

        console.log("predictions:", predictions.arraySync().slice(0, 10));
        console.log("mean of labels:", tf.mean(labels).arraySync());
        console.log("sst:", sst);
        console.log("ssr:", ssr);
        console.log("r2:", r2);
    }

    predict(features) {
        features = tf.tensor(features);
        features = this.scaler.transform(features);
        features = this._addOnes(features);

        const prediction = features.matMul(this.weights);
        return prediction;
    }

    _addOnes(tensor) {
        return tf.ones([tensor.shape[0], 1]).concat(tensor, 1);
    }

    _recordMSE() {}

    _updateLearningRate() {
        if (this.mses.length < 2) return;

        const current = this.mses.slice(-1);
        const previous = this.mses.slice(-2);
        if (current > previous) this.options.learningRate /= 2;
        else this.options.learningRate *= 1.05;
    }
}
