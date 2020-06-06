import React, {useState} from 'react';
import update from 'immutability-helper';
import {Button} from '@material-ui/core';
import * as tf from '@tensorflow/tfjs';
import './TensoflowExample.css';
import TextField from "@material-ui/core/TextField";

const TensorflowExample = () => {
    //initial state of value pairs
    const [valuePairsSate, setValuePairsState] = useState([
        {x: -1, y: -3},
        {x: 0, y: -1},
        {x: 1, y: 1},
        {x: 2, y: 3},
        {x: 3, y: 5},
        {x: 4, y: 7}
    ])

    //Model definition state
    const [modelState, setModelState] = useState({
        model: null,
        trained: false,
        predictedValue: 'Click on train!',
        valueToPredict: 1
    })

    //Event handlers
    const handleValuePairChange = (e) => {
        const updatedValuePairs = update(valuePairsSate, {
            [e.target.dataset.index]: {
                [e.target.name]: {$set: parseInt(e.target.value)}
            }
        })

        setValuePairsState(
            updatedValuePairs
        )
    };

    const handleAddItem = () => {
        setValuePairsState([
            ...valuePairsSate,
            {x: 1, y: 1}
        ])
    }

    const handleModelChange = (e) => setModelState({
        ...modelState,
        [e.target.name]: [parseInt(e.target.value)]
    })

    const handleTrainModel = () => {
        let xValues = [],
            yValues = [];

        valuePairsSate.forEach((val, index) => {
            xValues.push(val.x);
            yValues.push(val.y);
        })

        // Definition of model for linear regression
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [1]}));//dense layer that takes shape of 1, and input of shape 1

        // Prepare model for training.
        //minimizing the squaredError from the prediction
        model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});//stocastic gradient descent
        //the sgd optimizer reaches the minimum error point
        const xs = tf.tensor2d(xValues, [xValues.length, 1]);
        const ys = tf.tensor2d(yValues, [yValues.length, 1]);

        // Train the model using the data
        model.fit(xs, ys, {epochs: 250}).then(() => {
            setModelState({
                ...modelState,
                model: model,
                trained: true,
                predictedValue: 'Ready for making predictions'
            })
        })
    }

    const handlePredict = () => {
        const predictedValue = modelState.model.predict(tf.tensor2d([modelState.valueToPredict], [1, 1]))//pass in value to predict in form of tensor
            .arraySync()[0][0];//to convert from tensorflow session

        setModelState({
            ...modelState,
            predictedValue: predictedValue
        })
    }

    return (
        <div className="lead">
            <h1 className="title">Tensorflow Neural Network</h1>
            <h2 className="section"> Training Data (x,y) pairs</h2>
            <div className="labels">
                <div className="field-label">X</div>
                <div className="field-label">Y</div>
            </div>
            {valuePairsSate.map((val, index) => {
                return (
                    <div key={index} className="input-group">
                        <TextField className="field"
                                   value={val.x}
                                   name="x"
                                   data-index={index}
                                   onChange={handleValuePairChange}
                                   type="number"
                                   variant="filled"
                                   pattern="[0-9]*"
                        />
                        <TextField className="field"
                                   value={val.y}
                                   name="y"
                                   data-index={index}
                                   variant="filled"
                                   onChange={handleValuePairChange}
                                   type="number"
                        />
                    </div>
                )
            })}
            <div className="btn-group">
                <Button
                    className="btn"
                    onClick={handleAddItem}
                    variant="contained"
                >
                    Add
                </Button>
                <Button
                    className="btn"
                    onClick={handleTrainModel}
                    variant="contained">
                    Train
                </Button>
            </div>
            <div className="predict-controls">
                <h2 className="section">Predict</h2>
                <TextField className="field"
                           value={modelState.predictedValue}
                           name="valueToPredict"
                           variant="filled"
                           onChange={handleModelChange}
                           placeholder="Enter an integer"
                           type="number"
                />
                <br/>
                <div className="element">
                    {modelState.predictedValue}
                </div>
                <Button
                    className="btn"
                    onClick={handlePredict}
                    disabled={!modelState.trained}
                    variant="contained">
                    Predict
                </Button>
            </div>
        </div>
    );
};

export default TensorflowExample
