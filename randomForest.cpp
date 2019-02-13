#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // TODO(you)
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
    vector<vector<int>> ret;
    int line;
    vector<int> lines;

    std::random_device rd;
    mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, samples.size() - 1);

    for (int i = 0; i < num_to_return; i++) {
        do {
            line = dist(mt) % samples.size();
        } while (std::find(lines.begin(), lines.end(), line) != lines.end());

        // Se retine fiecare index al liniei pentru a nu se repeta.
        lines.push_back(line);
        ret.push_back(samples[line]);
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar

    vector<int> digits(10, 0);
    vector<int> predictions(num_trees);
    int pmax = 0;

    // Se interogheaza fiecare decision tree si se retine rezultatul prezis,
    // determinandu-se cel mai mare numar de aparitii al unei clase.
    for (int i = 0; i < num_trees; i++) {
        int digit = trees[i].predict(image);
        predictions[i] = digit;
        digits[digit]++;

        if (digits[pmax] < digits[digit]) {
            pmax = digit;
        }
    }

    // Dintre clasele cu cel mai mare numar de aparitii se alege prima.
    for (int i = 0; i < num_trees; i++) {
        if (digits[pmax] == digits[predictions[i]]) {
            pmax = predictions[i];
            break;
        }
    }

    return pmax;
}
