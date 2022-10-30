#ifndef _SIMULATION_H_
#define _SIMULATION_H_
// Test include for cmake
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "alphaGen.h"
#include "alpha-tree.h"
#include "at-fitness.h"

enum combMeth
{  
    ArithmaticMean,
    GeometricMean
};

at::AlphaTree generate_gabor_alphatree(cv::Mat& img, double l_p, 
                                       double lambda_o, double sigma_o, double gamma_o,
                                       int p_edge, double slope_edge, double center_edge, double weight_edge,
                                       double lambda_e, double sigma_e, double gamma_e,
                                       int p_ridge, double slope_ridge, double center_ridge, double weight_ridge);

at::AlphaTree generate_cd_1dl_alphatree(cv::Mat& img, double l_p,
                                        double p_edge, double slope_edge, double center_edge, double weight_edge,
                                        double p_ridge, double slope_ridge, double center_ridge, double weight_ridge);

at::AlphaTree generate_wilkinson_alphatree(cv::Mat& img, double p_f, double w_f, double p_b, double w_b, double p_c, double w_c);

std::vector<double> compute_scores(at::AlphaTree img_tree, std::vector<cv::Mat> ground_truths, enum combMeth method);

#endif // _SIMULATION_H_