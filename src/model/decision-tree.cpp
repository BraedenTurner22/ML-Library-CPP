#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <memory>

// ─────────────────────────────────────────────
//  TreeNode
// ─────────────────────────────────────────────

struct TreeNode {
    int    feature_index = -1;          // -1 means this is a leaf
    std::string label;                  // set for leaf nodes
    std::map<std::string, std::shared_ptr<TreeNode>> children;

    bool is_leaf() const { return feature_index == -1; }
};

class DecisionTree {

}