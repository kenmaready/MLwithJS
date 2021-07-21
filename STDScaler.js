import * as tf from "@tensorflow/tfjs";

export default class STDScaler {
    fit(features) {
        const { mean, variance } = tf.moments(features, 0);
        this.mean = mean;
        this.std = variance.pow(0.5);
        console.log(
            "mean:",
            this.mean.arraySync(),
            "std:",
            this.std.arraySync(),
        );
    }

    transform(features) {
        return features.sub(this.mean).div(this.std);
    }
}
