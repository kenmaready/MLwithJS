import fs from "fs";
import _ from "lodash";
import shuffleSeed from "shuffle-seed";

const defaultOptions = {
    dataColumns: [],
    labelColumns: [],
    shuffle: true,
    splitTest: false,
    converters: {},
};

function extractColumns(data, columnNames) {
    const headers = _.first(data);
    const indexes = _.map(columnNames, (column) => headers.indexOf(column));
    const extracted = _.map(data, (row) => _.pullAt(row, indexes));
    return extracted;
}

export default function loadCSV(filename, options) {
    options = { ...defaultOptions, ...options };

    const file = fs.readFileSync(filename, { encoding: "utf-8" });
    const rows = file.split("\n");
    let data = rows.map((row) => row.split(","));
    data = data.map((row) => _.dropRightWhile(row, (val) => val === ""));

    // get column names
    const headers = _.first(data);

    // parse data
    data = data.map((row, index) => {
        if (index === 0) return row;

        return row.map((element, index) => {
            if (options.converters[headers[index]]) {
                const converted = options.converters[headers[index]](element);
                return _.isNaN(converted) ? element : converted;
            }

            const result = parseFloat(element);
            return _.isNaN(result) ? element : result;
        });
    });

    let features = extractColumns(data, options.dataColumns);
    let labels = extractColumns(data, options.labelColumns);

    features.shift();
    labels.shift();

    if (options.shuffle) {
        features = shuffleSeed.shuffle(features, "phrase");
        labels = shuffleSeed.shuffle(labels, "phrase");
    }

    if (options.splitTest) {
        const trainSize = _.isNumber(options.splitTest)
            ? options.splitTest
            : Math.floor(data.length / 2);

        return {
            features: features.slice(trainSize),
            labels: labels.slice(trainSize),
            testFeatures: features.slice(0, trainSize),
            testLabels: labels.slice(0, trainSize),
        };
    } else {
        return { features, labels };
    }
}

const { features, labels, testFeatures, testLabels } = loadCSV("cars.csv", {
    converters: {
        passedemissions: (value) => value === "TRUE ",
    },
    dataColumns: ["passedemissions", "displacement", "cylinders"],
    labelColumns: ["mpg"],
    splitTest: 30,
});

console.log(testFeatures);
console.log(testLabels);
