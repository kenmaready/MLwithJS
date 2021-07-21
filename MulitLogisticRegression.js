import * as tf from "@tensorflow/tfjs";
import STDScaler from "./STDScaler.js";

const defaultOptions = { learningRate: 0.1, iterations: 60000 };

export default class MultiLogisticRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        if (options.scale) {
            this.scaler = new STDScaler();
            this.scaler.fit(features);
            this.features = this.scaler.transform(this.features);
        }
        this.features = this._addOnes(this.features);

        this.labels = tf.tensor(labels);
        this.options = {
            ...defaultOptions,
            ...options,
        };

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    }

    gradientDescent(features, labels) {
        const currentPredictions = features.matMul(this.weights).softmax();
        const errors = currentPredictions.sub(labels);

        // get crossentropy value:
        const sigmoidGuesses = features.matMul(this.weights).sigmoid();
        const termOne = labels
            .transpose()
            .matMul(sigmoidGuesses.add(1e-7).log());
        const termTwo = labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(sigmoidGuesses.mul(-1).add(1).add(1e-7).log());
        const crossentropy = termOne
            .add(termTwo)
            .div(features.shape[0])
            .mul(-1)
            .arraySync()[0][0];

        const gradients = features
            .transpose()
            .matMul(errors)
            .div(features.shape[0]);
        this.weights = this.weights.sub(
            gradients.mul(this.options.learningRate),
        );
        return { crossentropy };
    }

    train(options) {
        options = { ...{ batchSize: this.features.shape[0] }, ...options };
        const numBatches = Math.floor(
            this.features.shape[0] / options.batchSize,
        );
        this.crossentropies = [];
        let crossentropy;
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
                crossentropy = this.gradientDescent(
                    featureSlice,
                    labelSlice,
                ).crossentropy;
            }

            this.crossentropies.push(crossentropy);
            this._updateLearningRate();
        }

        // console.log(
        //     "Training complete. Current weights:",
        //     this.weights.arraySync(),
        // );
        // console.log(
        //     "crossentropies:",
        //     this.crossentropies.slice(this.crossentropies.length - 10),
        // );
    }

    test(features, labels) {
        const predictions = this.predict(features);
        labels = tf.tensor(labels).argMax(1);

        const incorrect = predictions.notEqual(labels).sum().arraySync();

        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    predict(features) {
        features = tf.tensor(features);
        if (this.options.scale) {
            features = this.scaler.transform(features);
        }
        features = this._addOnes(features);

        const prediction = features.matMul(this.weights).softmax().argMax(1);
        return prediction;
    }

    _addOnes(tensor) {
        return tf.ones([tensor.shape[0], 1]).concat(tensor, 1);
    }

    _recordMSE() {}

    _updateLearningRate() {
        if (this.crossentropies.length < 2) return;

        const current = this.crossentropies.slice(-1);
        const previous = this.crossentropies.slice(-2);
        if (current > previous) this.options.learningRate /= 2;
        else this.options.learningRate *= 1.05;
    }
}
