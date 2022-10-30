#ifndef ALPHA_TREE_H
#define ALPHA_TREE_H

#include <opencv2/core.hpp>
#include "EdgeQueue.h" 
#include <set>
#include <vector>
#include <math.h>
#include <iostream>

#define AT_BOTTOM -1

namespace at
{    
    // Nodes in the Alpha-tree
    class AlphaTreeNode
    {
        int    d_parent;      // Parent idx of the current node (array tree representation)
        double d_alpha;       // Alpha of flat zone
        cv::Vec3b d_outVal;     // The output color value of the node

        public:
            AlphaTreeNode();
            AlphaTreeNode(double alpha);  

            double getAlpha();
            void setAlpha(double alpha);

            int getParent();
            void setParent(int parent);

            cv::Vec3b getOutval();
            void setOutval(cv::Vec3b bgrVector);
            void setOutval(uchar b, uchar g, uchar r);
            void setRandOutval();
    };

    // Alpha-tree class
    class AlphaTree
    {
        // Data related to the tree
        std::vector<at::AlphaTreeNode> d_node;  // Array of nodes in the tree
        int d_curSize;
        int d_maxSize;
        // Output image size
        int d_height;
        int d_width;
        
        int d_connectivity;

        public:
            // Constructors
            AlphaTree();
            AlphaTree(cv::Mat alpha_h, cv::Mat alpha_v, double lambdamin=0);  // Create an alpha tree based on 2 precomputed alpha matrices

            // Support functions
            void printNodes();

            // Filters
            cv::Mat SalienceFilter(double lambda);
            
            // getter functions
            int size();
            int imgSize();
            double maxAlpha();
            int getConnectivity();
            cv::Mat getHistogram(int bins, bool normalised);
            std::vector<double> getAlphas();

        private:
            int NewNode(int* root, double alpha);
            int FindRoot(int* root, int p);
            int FindRoot1(int* root, int p);
            int LevelRoot(int p);
            bool IsLevelRoot(int i);
            void MakeSet(int* root, int p);
            void GetAncestors(int* root, int& p, int& q);
            void Union(int* root, int p, int q);
            void Union2(int* root, int p, int q);
            void Phase1(at::EdgeQueue* queue, int* root, cv::Mat alpha_h, cv::Mat alpha_v, double lambdamin);
            void Phase2(at::EdgeQueue* queue, int* root);
    };

    void printHistogram(cv::Mat histogram);
}


#endif // ALPHA_TREE_H