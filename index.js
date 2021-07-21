import * as tf from "@tensorflow/tfjs";
import plot from "node-remote-plot";
import _ from "lodash";
import loadCSV from "./load-csv.js";
import LinearRegression from "./LinearRegression.js";
import LogisticRegression from "./LogisticRegression.js";
import MultiLogisticRegression from "./MulitLogisticRegression.js";

// let mode = "linear";
// if (process.argv[2] === "log") mode = "log";
// if (process.argv[2] === "multilog") mode = "multilog";

let mode;
switch (process.argv[2]) {
    case "log":
        mode = "log";
        break;
    case "multilog":
        mode = "multilog";
        break;
    default:
        mode = "linear";
}

// console.log(features, labels);

if (mode === "linear") {
    console.log("Linear Regression Mode....");

    let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
        shuffle: true,
        splitTest: 50,
        dataColumns: ["horsepower", "weight", "displacement"],
        labelColumns: ["mpg"],
    });

    const lr = new LinearRegression(features, labels, {
        learningRate: 0.01,
        iterations: 20,
    });
    console.log("lr:", lr);
    lr.train({ batchSize: 10 });
    lr.test(testFeatures, testLabels);

    plot({ x: lr.mses, title: "my_plot", name: "my_plot" });

    const predictions = lr.predict(testFeatures.slice(0, 10));
    console.log("predictions:", predictions.arraySync());
}

if (mode === "log") {
    console.log("Logistic Regression Mode....");

    let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
        shuffle: true,
        splitTest: 50,
        dataColumns: ["horsepower", "weight", "displacement"],
        labelColumns: ["passedemissions"],
        converters: {
            passedemissions: (value) => {
                return value === "TRUE" ? 1 : 0;
            },
        },
    });

    const lr = new LogisticRegression(features, labels, {
        learningRate: 0.01,
        iterations: 30,
    });
    console.log("lr:", lr);
    lr.train({ batchSize: 10 });
    const accuracy = lr.test(testFeatures, testLabels);
    console.log("Testing accuracy:", accuracy);

    plot({ x: lr.crossentropies, title: "my_plot", name: "my_plot" });

    const predictions = lr.predict(testFeatures).arraySync();

    let correct = 0;
    for (let i = 0; i < testLabels.length; i++) {
        console.log(
            "Guess:",
            Math.round(predictions[i]),
            "Actual:",
            testLabels[i],
        );
        if (Math.round(predictions[i]) === testLabels[i][0]) correct++;
    }
    console.log(
        "No. correct:",
        correct,
        "Accuracy:",
        correct / predictions.length,
    );
}

if (mode === "multilog") {
    console.log("Multinomial Logistic Regression Mode....");

    let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
        shuffle: true,
        splitTest: 50,
        dataColumns: ["horsepower", "weight", "displacement"],
        labelColumns: ["mpg"],
        converters: {
            mpg: (value) => {
                if (value < 15) return [1, 0, 0];
                else if (value < 30) return [0, 1, 0];
                else return [0, 0, 1];
            },
        },
    });

    labels = _.flatMap(labels);
    testLabels = _.flatMap(testLabels);

    const mlr = new MultiLogisticRegression(features, labels, {
        learningRate: 0.01,
        iterations: 30,
        scale: true,
    });

    mlr.train();
    const accuracy = mlr.test(testFeatures, testLabels);
    console.log("accuracy:", accuracy);

    plot({ x: mlr.crossentropies, title: "my_plot", name: "my_plot" });

    // const predictions = mlr.predict(testFeatures);
    // console.log("Predictions:", predictions.arraySync());
}
