require("@tensorflow/tfjs-node");
const { op, tensor, tensor1d } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
    const headers = _.first(data);
  
    const indexes = _.map(columnNames, column => headers.indexOf(column));
    const extracted = _.map(data, row => _.pullAt(row, indexes));
  
    return extracted;
};
  
function loadCSV(
    filename,
    {
      dataColumns = [],
      labelColumns = [],
      converters = {},
      shuffle = false,
      splitTest = false
    }
  ) {
    let data = fs.readFileSync(filename, { encoding: 'utf-8' });
    data = _.map(data.split('\n'), d => d.split(','));
    data = _.dropRightWhile(data, val => _.isEqual(val, ['']));
    const headers = _.first(data);
  
    data = _.map(data, (row, index) => {
      if (index === 0) {
        return row;
      }
      return _.map(row, (element, index) => {
        if (converters[headers[index]]) {
          const converted = converters[headers[index]](element);
          return _.isNaN(converted) ? element : converted;
        }
  
        const result = parseFloat(element.replace('"', ''));
        return _.isNaN(result) ? element : result;
      });
    });
  
    let labels = extractColumns(data, labelColumns);
    data = extractColumns(data, dataColumns);
  
    data.shift();
    labels.shift();
  
    if (shuffle) {
      data = shuffleSeed.shuffle(data, 'phrase');
      labels = shuffleSeed.shuffle(labels, 'phrase');
    }
  
    if (splitTest) {
      const trainSize = _.isNumber(splitTest)
        ? splitTest
        : Math.floor(data.length / 2);
  
      return {
        features: data.slice(trainSize),
        labels: labels.slice(trainSize),
        testFeatures: data.slice(0, trainSize),
        testLabels: labels.slice(0, trainSize)
      };
    } else {
      return { features: data, labels };
    }
};

function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
  
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
  
    return (
      features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
    );
}
  
let { features, labels, testFeatures, testLabels } = loadCSV(
    'GPA_data.csv',
    {
      shuffle: true,
      splitTest: 10,
      dataColumns: [`hsGPA`, `SATM`, `SATV`, `collegeYear`],
      labelColumns: [`collegeGPA`]
    }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);
const err = (testLabels[0][0] - result) / testLabels[0][0];
console.log('Error', err * 100);
console.log(`Prediction:`, result.toFixed(2), `Real Value:`, (testLabels[0][0]).toFixed(2));
