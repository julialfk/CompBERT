module Compare

import IO;
import lang::json::IO;
import lang::java::m3::AST;
import util::Math;
import Node;
import List;
import Set;
import Map;


// Calculate the similarity of two subtrees for Type III clones
// Based on the paper from Baxter et al.

// Similarity = 2 x S / (2 x S + L + R)
// where:
//     S = number of shared nodes
//     L = number of different nodes in sub-tree 1
//     R = number of different nodes in sub-tree 2
real calcSimilarity(node n1, node n2) {
    list[node] tree1 = [];
    list[node] tree2 = [];

    visit (n1) { case node n: tree1 += n; }
    visit (n2) { case node n: tree2 += n; }

    int sharedNodes = size(tree1 & tree2);
    int diffN1 = size(tree1) - sharedNodes;
    int diffN2 = size(tree2) - sharedNodes;

    real similarity = (2.0 * sharedNodes / (2.0 * sharedNodes + diffN1 + diffN2)) * 100.0;

    return precision(similarity, 5);
}


public bool areSimilar(node n1, node n2) {
    n1 = unsetRec(n1, "src");
    n2 = unsetRec(n2, "src");
    return calcSimilarity(n1, n2) >= 99.9;
}