const { op } = require("@tensorflow/tfjs");
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

class LinearRegression {
    constructor (features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
    
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000}, options);

        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent() {
        const currentGuesses = this.features.matMul(this.weights);
        const differences = currentGuesses.sub(this.labels);

        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }
    
    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights);
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);

        const res = testLabels.sub(predictions)
            .pow(2)
            .sum()
            .arraySync();
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .arraySync();
        
        return 1 - res / tot;
    }

    processFeatures(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        }   else {
            features = this.standardise(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    standardise(features) {
        const { mean, variance } = tf.moments(features, 0);
         
        this.mean = mean;
        this.variance = variance.add(1e-7);
         
        features = features.sub(mean).div(variance.add(1e-7));
         
        return features;
    }

    recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .arraySync();
        
        this.mseHistory.unshift(mse);
    }

    updateLearningRate() {
        if (this.mseHistory.length < 2) {
            return;
        }

        if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else{
            this.options.learningRate *= 1.05;
        }
    }
};


let { features, labels, testFeatures, testLabels } = loadCSV(`./GPA_data.csv`, {
    shuffle: true,
    splitTest: 50,
    dataColumns: [`hsGPA`, `SATM`, `SATV`, `collegeYear`],
    labelColumns: [`collegeGPA`],
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100,
});

// console.log(features, labels);

regression.train();

const r2 = regression.test(testFeatures, testLabels);
console.log(`R2 is`, r2);

regression.predict([
    [4,740,790,2]
]).print();