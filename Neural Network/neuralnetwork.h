#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <omp.h>
#include <SFML/Graphics.hpp>
using namespace std;

struct DataPoint {
	DataPoint(vector<double>& input, vector<double> expected) : inputs(input), expectedOutputs(expected) {}
	vector<double> inputs;
	vector<double> expectedOutputs;
};

struct DataList {
	vector<sf::CircleShape> sprites;
	vector<DataPoint> data;

	void addDataPoint(DataPoint dataPoint, sf::Vector2f pos, sf::Color color);
	void clearDataPoints();
	void display(sf::RenderWindow& window);
};

struct Layer {
	vector<vector<double>> weights;
	vector<double> biases;
	vector<vector<double>> costWeightGradients;
	vector<double> costBiasGradients;
	vector<double> inputs;
	vector<double> weightedInputs;
	int inputSize, outputSize;

	Layer(int nodesIn, int nodesOut);
	vector<double> calculateOutputs(vector<double>& inputs); 
	vector<double> calculateOutputsNoSaving(vector<double>& inputs);
	void updateGradients(vector<double>& nodeValues);
	void applyGradients(double learningRate);
	void clearGradients();
	vector<double> calculateHiddenLayerNodeValues(Layer& prevLayer, vector<double> prevNodeValues);
};

class NeuralNetwork {
public:
	NeuralNetwork(vector<int> layerCounts);
	vector<double> forwardProp(vector<double>& inputs);
	vector<double> calculateOutputsNoSaving(vector<double>& inputs);
	void randomizeWB();
	double pointCost(DataPoint data);
	double cost(vector<DataPoint> data);
	vector<double> calculateOutputNodeValues(Layer& layer, vector<double> expectedOutputs, vector<double> outputs);
	void updateAllGradients(DataPoint data);
	void applyAllGradients(double learningRate);
	void clearAllGradients();
	void learn(vector<DataPoint> data, double learningRate);

private:
	vector<Layer> network;
};


#endif
