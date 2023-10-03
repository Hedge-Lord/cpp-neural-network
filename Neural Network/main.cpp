#include <SFML/Graphics.hpp>
#include <iostream>
#include "neuralnetwork.h"
#include <omp.h>

// SCREEN_SIZE and SCREEN_RES must be integers, doubles are used to concise the code
const double SCREEN_SIZE = 60;
const double SCREEN_RES = 720;
const double SCALE_FACTOR = SCREEN_RES / SCREEN_SIZE;

// CHANGE NUM NODES PER LAYER AND NUM LAYERS
const vector<int> layerCounts = { 2, 4, 3 };



int main()
{
    NeuralNetwork net(layerCounts);
    net.randomizeWB();

    sf::Image image;
    image.create(SCREEN_SIZE, SCREEN_SIZE);
    sf::Sprite sprite;
    sf::Texture texture;
    texture.loadFromImage(image);
    sprite.setTexture(texture);
    sprite.setScale(SCALE_FACTOR, SCALE_FACTOR);
    vector<double> outputs;
    vector<double> inputs;
    DataList data;
    bool training = false;

    sf::RenderWindow window(sf::VideoMode(SCREEN_RES, SCREEN_RES), "Neural Network");

    while (window.isOpen()) {
#pragma omp parallel num_threads(1) // process mouse and keyboard input
        {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) window.close();

                if (event.type == sf::Event::MouseButtonPressed) {

                    sf::Vector2i v = sf::Mouse::getPosition(window);
                    vector<double> input = { v.x / SCREEN_RES, v.y / SCREEN_RES };

                    if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                        vector<double> output = { 1, 1, 0 };
                        data.addDataPoint(DataPoint(input, output), sf::Vector2f(v.x, v.y), sf::Color(255, 255, 0));
                    }

                    else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
                        vector<double> output = { 0, 1, 1 };
                        data.addDataPoint(DataPoint(input, output), sf::Vector2f(v.x, v.y), sf::Color(0, 255, 255));
                    }

                    else if (sf::Mouse::isButtonPressed(sf::Mouse::Middle)) {
                        vector<double> output = { 1, 0, 1 };
                        data.addDataPoint(DataPoint(input, output), sf::Vector2f(v.x, v.y), sf::Color(255, 0, 255));
                    }

                }

                if (event.type == sf::Event::KeyPressed) {
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {
                        net.randomizeWB();
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Tab)) {
                        for (int i = 0; i < 1000; i++)
                            net.learn(data.data, 0.999);
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Return)) {
                        training = !training;
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
                        data.clearDataPoints();
                        net.randomizeWB();
                        training = !training;
                    }
                }
            }
        }

        // train
#pragma omp parallel num_threads(1) 
        {
            if (training)
                for (int i = 0; i < 100; i++)
                    net.learn(data.data, 0.999);
            double cost = net.cost(data.data);
            cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ntotal loss: " << cost;
            cout << "\n--------\nLCLICK -> yellow point\nRCLICK -> cyan point\nMCLICK -> magenta point\n--------\nENTER  -> start/stop training\nTAB    -> run 1000 training iterations\nLCTRL  -> randomize weights and biases\nLSHIFT -> clear points";
        }

        // update screen pixels with neural network predictions
#pragma omp parallel num_threads(1) 
        {
#pragma omp parallel for
            for (int i = 0; i < SCREEN_SIZE; i++) {
#pragma omp parallel for
                for (int j = 0; j < SCREEN_SIZE; j++) {
                    inputs = { i / SCREEN_SIZE, j / SCREEN_SIZE };
                    outputs = net.calculateOutputsNoSaving(inputs);
                    image.setPixel(i, j, sf::Color(outputs[0] * 255, outputs[1] * 255, outputs[2] * 255));
                    texture.update(image);
                }
            }
        }

        // update screen
            window.clear();
            window.draw(sprite);
            data.display(window);
            window.display();
    }

}

