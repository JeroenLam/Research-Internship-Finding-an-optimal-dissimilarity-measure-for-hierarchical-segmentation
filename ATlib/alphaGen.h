#ifndef _ALPHAGEN_H_
#define _ALPHAGEN_H_
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "alpha-tree.h"


// L_p norm over channels
cv::Mat Lp_over_channels(cv::Mat& img, double p);
std::vector<cv::Mat> Lp_over_channels(std::vector<cv::Mat>& imgages, double p);

// Gabor filter
std::vector<cv::Mat> compute_gabor(cv::Mat& img, int size, std::vector<double>& thetas, double lambda, double sigma, double gamma, double psi, int ddepth=-1, int borderType=cv::BORDER_DEFAULT);
std::vector<cv::Mat> even_gabor(cv::Mat& img, int size, std::vector<double>& thetas, double lambda, double sigma, double gamma, int ddepth=-1, int borderType=cv::BORDER_DEFAULT);
std::vector<cv::Mat> odd_gabor(cv::Mat& img, int size, std::vector<double>& thetas, double lambda, double sigma, double gamma, int ddepth=-1, int borderType=cv::BORDER_DEFAULT);

// Dissimilarities
std::vector<cv::Mat> dissimilarity_l_p_4(cv::Mat& img, double p);

// Sigmoidal activation
cv::Mat sigmoidal_activation(cv::Mat &images, double slope, double center, double upb=1, double lwb=0, bool inplace=false);
std::vector<cv::Mat> sigmoidal_activation(std::vector<cv::Mat> images, double slope, double center, double upb, double lwb, bool inplace=false);

// CD & 1dL
cv::Mat backward_difference(cv::Mat& img, int direction);
cv::Mat forward_difference(cv::Mat& img, int direction);
cv::Mat central_difference(cv::Mat& img, int direction);
cv::Mat Laplacian_1d(cv::Mat& img, int direction);

std::vector<cv::Mat> backward_difference_4(cv::Mat& img);
std::vector<cv::Mat> forward_difference_4(cv::Mat& img);
std::vector<cv::Mat> central_difference_4(cv::Mat& img);
std::vector<cv::Mat> Laplacian_1d_4(cv::Mat& img);


// Wilkinson detector
std::vector<cv::Mat> compute_Wilkinson(std::vector<cv::Mat> for_dif, std::vector<cv::Mat> back_dif, std::vector<cv::Mat> cent_dif, std::vector<double> weigths);


// Example code
std::vector<cv::Mat> alpha_example(cv::Mat& img);
double simpleSalience(cv::Vec3b p, cv::Vec3b q);
double WeightedSalience(cv::Vec3b p, cv::Vec3b q);
double EdgeStrengthX(cv::Mat img, int x, int y);
double EdgeStrengthY(cv::Mat img, int x, int y);

#endif // _ALPHAGEN_H_