#ifndef AT_FITNESS_H
#define AT_FITNESS_H

#include <opencv2/imgcodecs.hpp> //debug
#include <opencv2/core.hpp>
#include "alpha-tree.h"
#include "support.h"
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <stack>
#include <utility> // For pairs

#define SIM_STEPS 20

namespace at
{
    // The fitness function
    double fitness_function(at::AlphaTree& tree, cv::Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI);
    double fitness_function(at::AlphaTree& tree, cv::Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI, double& a_h);

    double fitness_function_quick(at::AlphaTree& tree, cv::Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI);
    double fitness_function_quick(at::AlphaTree& tree, cv::Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI, double& a_h);

    // The score components of the fitness function
    double node_index(at::AlphaTree& tree);
    double depth_index(at::AlphaTree& tree, double a_h);
    double plateau_length_index(at::AlphaTree& tree, std::vector<double>& alphas, std::vector<double>& vec_AS, double AS_h, int idx_h);
    double area_score(cv::Mat& img_f, std::map<int,int>& cc_img_f, cv::Mat& img_gt, std::map<int,int>& cc_img_gt, int connectivity=4);

    // Extra functions to compute the area score
    double under_merging_error(cv::Mat& img_f, cv::Mat& img_gt, std::map<int,int>& cc_img_gt, int connectivity=4);
    double over_merging_error( cv::Mat& img_f, cv::Mat& img_gt, std::map<int,int>& cc_img_f,  int connectivity=4);
    double merging_error(      cv::Mat& img1,  cv::Mat& img2,   std::map<int,int>& cc_img2,   int connectivity=4);
    std::map<int, int> cc_bfs_4(cv::Mat& img1, cv::Mat& img2, cv::Mat& mask, int x, int y);
    std::map<int, int> cc_bfs_8(cv::Mat& img1, cv::Mat& img2, cv::Mat& mask, int x, int y);
}   


#endif // AT_FITNESS_H
