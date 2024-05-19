#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

using namespace std;

struct TreeNode {
    bool isLeaf;
    int classification; // ?????子???的分?
    int attribute; // ????非?子???的分割?性
    std::map<int, TreeNode*> children; // 每?分支的子??
};

// ?算多??
int pluralityValue(const std::vector<std::vector<double>>& examples) {
    std::map<int, int> frequency;
    for (const auto& example : examples) {
        frequency[example.back()]++;
    }
    return std::max_element(frequency.begin(), frequency.end(), 
                        [](const std::pair<const int, int>& a, const std::pair<const int, int>& b) { 
                            return a.second < b.second; 
                        })->first;
}

// ?算信息增益（基于熵）
double entropy(const std::vector<std::vector<double>>& examples) {
    std::map<int, int> frequency;
    for (const auto& example : examples) {
        frequency[example.back()]++;
    }
    double ent = 0.0;
    for (const std::pair<const int, int>& pair : frequency) {
    	const int& key = pair.first;
    	const int& value = pair.second;
        double p = static_cast<double>(value) / examples.size();
        ent -= p * std::log2(p);
    }
    return ent;
}

double informationGain(const std::vector<std::vector<double>>& examples, int attribute) {
    double originalEntropy = entropy(examples);
    std::map<int, std::vector<std::vector<double>>> subsets;
    for (const std::vector<double> example : examples) {
        subsets[example[attribute]].push_back(example);
    }
    double weightedEntropy = 0.0;
    for (const std::pair<const double, std::vector<std::vector<double>>>& pair : subsets) {
    	const double& value = pair.first;
    	const std::vector<std::vector<double>>& subset = pair.second;
        weightedEntropy += static_cast<double>(subset.size()) / examples.size() * entropy(subset);
    }
    return originalEntropy - weightedEntropy;
}

// ??最优?性
int chooseBestAttribute(const std::vector<std::vector<double>>& examples, const std::vector<int>& attributes) {
    int bestAttribute = -1;
    double bestGain = -1.0;
    for (int attribute : attributes) {
        double gain = informationGain(examples, attribute);
        if (gain > bestGain) {
            bestGain = gain;
            bestAttribute = attribute;
        }
    }
    return bestAttribute;
}

// 核心??函?
TreeNode* learnDecisionTree(const std::vector<std::vector<double>>& examples, 
                            std::vector<int> attributes, 
                            const std::vector<std::vector<double>>& parent_examples) {
    if (examples.empty()) {
        // 若?前?本集?空，返回父??中?本?量最多的分?
        TreeNode* leaf = new TreeNode{true, pluralityValue(parent_examples), -1, {}};
        return leaf;
    }

    // ?查是否所有?本?于同一??
    bool allSame = std::all_of(examples.begin(), examples.end(), [&](const std::vector<double>& e) { 
        return e.back() == examples.front().back(); 
    });
    if (allSame) {
        TreeNode* leaf = new TreeNode{true, examples.front().back(), -1, {}};
        return leaf;
    }

    if (attributes.empty()) {
        // 若?有更多?性可供分割，返回?前?本集中的多??
        TreeNode* leaf = new TreeNode{true, pluralityValue(examples), -1, {}};
        return leaf;
    }

    // ??最优?性
    int bestAttribute = chooseBestAttribute(examples, attributes);
    TreeNode* tree = new TreeNode{false, -1, bestAttribute, {}};

    // 去掉已??的?性
    auto it = std::remove(attributes.begin(), attributes.end(), bestAttribute);
    attributes.erase(it, attributes.end());

    // 构建子?
    std::map<double, std::vector<std::vector<double>>> subsets;
    for (const auto& example : examples) {
        subsets[example[bestAttribute]].push_back(example);
    }

    for (const auto& subset_pair : subsets) {
        const double& value = subset_pair.first;
        const std::vector<std::vector<double>>& subset = subset_pair.second;
        TreeNode* subtree = learnDecisionTree(subset, attributes, examples);
        tree->children[value] = subtree;
    }

    return tree;
}
// 打印?策?（?助函?）
void printTree(TreeNode* node, const std::string& prefix = "") {
    if (node->isLeaf) {
        std::cout << prefix << "Leaf: " << node->classification << "\n";
    } else {
        std::cout << prefix << "Attribute: " << node->attribute << "\n";
        for (const std::pair<const double, TreeNode*>& pair : node->children) {
        	const double& value = pair.first;
    	TreeNode* child = pair.second;
            printTree(child, prefix + "--" + std::to_string(value) + "--> ");
        }
    }
}

vector<vector<double>> examples;
vector<vector<double>> parent_examples;
vector<int> attributes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};


int main() {
	//ifstream file("https://raw.githubusercontent.com/chenboyu1/linkgenerate/main/train.csv");
	FILE* fin = fopen("train.csv", "rb");
	FILE* output = fopen("ouput.txt", "w");
	char s[10000];
	//string s;
	int idx = 0;
	for(int i = 0; i < 15; i++){
	 	fgets(s, 10000, fin);
        stringstream ss(s);
        string token;
        int att = 0;
        vector<double> line;
        while (getline(ss, token, ',')) {
            line.push_back((double)stof(token));
            //printf("%f ", examples[idx][att]);
            att++;
        }
        examples.push_back(line);
        ++idx;
        //printf("\n");
    }
    TreeNode* tree = learnDecisionTree(examples, attributes, {});
	
	printTree(tree);
	return 0;
}
