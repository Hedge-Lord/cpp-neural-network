#include "neuralnetwork.h"
#include <SFML/Graphics.hpp>

double sigmoid(double num) { return 1 / (1 + exp(-num)); }

double dSigmoid(double num) {
	double a = sigmoid(num);
	return a * (1 - a);
}

double nodeCost(double output, double expected) {
	double a = output - expected;
	return a * a;
}

double dNodeCost(double val, double expected) { return 2 * (val - expected); }

Layer::Layer(int nodesIn, int nodesOut) {
	inputSize = nodesIn;
	outputSize = nodesOut;
	for (int i = 0; i < nodesOut; i++) {
		weights.push_back({});
		costWeightGradients.push_back({});
		biases.push_back(0);
		costBiasGradients.push_back(0);
		weightedInputs.push_back(0);
		for (int j = 0; j < nodesIn; j++) {
			inputs.push_back(0);
			weights[i].push_back(0);
			costWeightGradients[i].push_back(0);
		}
	}
}

vector<double> Layer::calculateOutputs(vector<double>& inputs) {
	vector<double> outputs(outputSize);
	this->inputs = inputs;

#pragma omp parallel for
	for (int i = 0; i < outputSize; i++) {
		double sum = biases[i];
#pragma omp parallel for
		for (int j = 0; j < inputSize; j++) {
			sum += weights[i][j] * inputs[j];
		}
		outputs[i] = (sigmoid(sum));
		weightedInputs[i] = sum;
	}
	return outputs;
}

vector<double> Layer::calculateOutputsNoSaving(vector<double>& inputs) {
	vector<double> outputs(outputSize);
	for (int i = 0; i < outputSize; i++) {
		double sum = biases[i];
		for (int j = 0; j < inputSize; j++) {
			sum += weights[i][j] * inputs[j];
		}
		outputs[i] = (sigmoid(sum));
	}
	return outputs;
}

void Layer::updateGradients(vector<double>& nodeValues) {
#pragma omp parallel for
	for (int i = 0; i < outputSize; i++) {
#pragma omp parallel for
		for (int j = 0; j < inputSize; j++) {
			costWeightGradients[i][j] += inputs[j] * nodeValues[i];
		}
		costBiasGradients[i] += nodeValues[i];
	}
}

void Layer::applyGradients(double learningRate) {
#pragma omp parallel for
	for (int i = 0; i < outputSize; i++) {
#pragma omp parallel for
		for (int j = 0; j < inputSize; j++) {
			weights[i][j] -= costWeightGradients[i][j] * learningRate;
		}
		biases[i] -= costBiasGradients[i] * learningRate;
	}
}

void Layer::clearGradients() {
#pragma omp parallel for
	for (int i = 0; i < outputSize; i++) {
#pragma omp parallel for
		for (int j = 0; j < inputSize; j++) {
			costWeightGradients[i][j] = 0;
		}
		costBiasGradients[i] = 0;
	}
}

vector<double> Layer::calculateHiddenLayerNodeValues(Layer& prevLayer, vector<double> prevNodeValues) {
	vector<double> nodeVals(outputSize);
#pragma omp parallel for
	for (int i = 0; i < outputSize; i++) {
		double nodeVal = 0;
#pragma omp parallel for
		for (int j = 0; j < prevNodeValues.size(); j++) {
			nodeVal += prevNodeValues[j] * prevLayer.weights[j][i];
		}
		nodeVal *= dSigmoid(weightedInputs[i]);
		nodeVals[i] = nodeVal;
	}
	return nodeVals;
}

NeuralNetwork::NeuralNetwork(vector<int> layerCounts) {
	for (int i = 0; i < layerCounts.size() - 1; i++) {
		network.push_back(Layer(layerCounts[i], layerCounts[i + 1]));
	}
}

vector<double> NeuralNetwork::forwardProp(vector<double>& inputs) {
	for (int i = 0; i < network.size(); i++) inputs = network[i].calculateOutputs(inputs);
	return inputs;
}

vector<double> NeuralNetwork::calculateOutputsNoSaving(vector<double>& inputs) {
	for (int i = 0; i < network.size(); i++) inputs = network[i].calculateOutputsNoSaving(inputs);
	return inputs;
}

void NeuralNetwork::randomizeWB() {
	for (int a = 0; a < network.size(); a++) {
		Layer& layer = network[a];
		for (int i = 0; i < layer.outputSize; i++) {
			for (int j = 0; j < layer.biases.size(); j++) layer.biases[j] = (1.0 * (rand() % 101) / 100);
			for (int j = 0; j < layer.weights.size(); j++)
				for (int k = 0; k < layer.weights[j].size(); k++) layer.weights[j][k] = (1.0 * (rand() % 101) / 100);
		}
	}
}

double NeuralNetwork::pointCost(DataPoint data) {
	vector<double> outputs = forwardProp(data.inputs);
	int size = outputs.size();
	double cost = 0;
	for (int i = 0; i < size; i++) cost += nodeCost(outputs[i], data.expectedOutputs[i]);
	return cost;
}

double NeuralNetwork::cost(vector<DataPoint> data) {
	double cost = 0;
#pragma omp parallel for
	for (int i = 0; i < data.size(); i++) {
		cost += pointCost(data[i]);
	}
	return cost / data.size();
}


vector<double> NeuralNetwork::calculateOutputNodeValues(Layer& layer, vector<double> expectedOutputs, vector<double> outputs) {
	vector<double> nodeVals(layer.outputSize);
#pragma omp parallel for
	for (int i = 0; i < layer.outputSize; i++) nodeVals[i] = (dNodeCost(outputs[i], expectedOutputs[i]) * dSigmoid(layer.weightedInputs[i]));
	return nodeVals;
}

void NeuralNetwork::updateAllGradients(DataPoint data) {
	vector<double> outputActivations = forwardProp(data.inputs);
	Layer& outputLayer = network[network.size() - 1];
	vector<double> nodeValues = calculateOutputNodeValues(outputLayer, data.expectedOutputs, outputActivations);
	outputLayer.updateGradients(nodeValues);

	for (int i = network.size() - 2; i >= 0; i--) {
		Layer& hiddenLayer = network[i];
		nodeValues = hiddenLayer.calculateHiddenLayerNodeValues(network[i + 1], nodeValues);
		hiddenLayer.updateGradients(nodeValues);
	}
}

void NeuralNetwork::applyAllGradients(double learningRate) {
#pragma omp parallel for
	for (int i = 0; i < network.size(); i++) {
		network[i].applyGradients(learningRate);
	}
}

void NeuralNetwork::clearAllGradients() {
#pragma omp parallel for
	for (int i = 0; i < network.size(); i++) {
		network[i].clearGradients();
	}
}

void NeuralNetwork::learn(vector<DataPoint> data, double learningRate) {
	for (int i = 0; i < data.size(); i++)
		updateAllGradients(data[i]);
	applyAllGradients(learningRate);
	clearAllGradients();
}

void DataList::addDataPoint(DataPoint dataPoint, sf::Vector2f pos, sf::Color color) {
	sf::CircleShape circle;
	circle.setRadius(8);
	circle.setOutlineColor(sf::Color::Black);
	circle.setOutlineThickness(-2);
	circle.setFillColor(color);
	if (color == sf::Color::Black) circle.setOutlineColor(sf::Color::White);
	circle.setPosition(pos);
	circle.setOrigin(sf::Vector2f(8, 8));
	sprites.push_back(circle);
	data.push_back(dataPoint);
}

void DataList::clearDataPoints() {
	sprites.clear();
	data.clear();
}

void DataList::display(sf::RenderWindow& window) {
	for (int i = 0; i < sprites.size(); i++) {
		window.draw(sprites[i]);
	}
}