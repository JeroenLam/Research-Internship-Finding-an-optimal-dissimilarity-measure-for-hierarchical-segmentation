#include "simulation.h"

using namespace std;
using namespace cv;

// ============================================================
//                           Simulation
// ============================================================
/**
 * @brief Generate an alpha tree using the odd and even Gabor filters to compute the dissimilarity used in the tree construction.
 * 
 * @param img The image of which to compute the alpha tree.
 * @param l_p The norm to use for the base dissimilarity computation.
 * @param size_o Diameter of the odd Gabor filter kernel
 * @param lambda_o Wavelength of the odd Gabor filter
 * @param sigma_o Standard deviation of the Gaussian factor of the odd Gabor filter
 * @param gamma_o Spatial aspect ratio of the odd Gabor filter
 * @param slope_o Slope of the logistic function for the edge signals
 * @param center_o Horizontal offset of the logistic function for the edge signals
 * @param weight_o Weight with which to combine the edge signal with the dissimilarity
 * @param size_e Diameter of the even Gabor filter kernel
 * @param lambda_e Wavelength of the even Gabor filter
 * @param sigma_e Standard deviation of the Gaussian factor of the even Gabor filter
 * @param gamma_e Spatial aspect ratio of the even Gabor filter
 * @param slope_e Slope of the logistic function for the ridge signals
 * @param center_e Horizontal offset of the logistic function for the ridge signals
 * @param weight_e Weight with which to combine the ridge signal with the dissimilarity
 * @return at::AlphaTree 
 */
at::AlphaTree generate_gabor_alphatree(Mat& img, double l_p, 
                                       double lambda_o, double sigma_o, double gamma_o,
                                       int p_edge, double slope_edge, double center_edge, double weight_edge,
                                       double lambda_e, double sigma_e, double gamma_e,
                                       int p_ridge, double slope_ridge, double center_ridge, double weight_ridge)
{
    int size = 9;
    // TODO: Not hardcode 4 connectivity
    vector<double> thetas;
    thetas.push_back(0);
    thetas.push_back(0.5*CV_PI);

    // Compute the dissimilarities
    vector<Mat> dissimilarity = dissimilarity_l_p_4(img, l_p);

    // Compute the odd gabor filters
    vector<Mat> edge_sig = odd_gabor(img, size, thetas, lambda_o, sigma_o, gamma_o, -1, cv::BORDER_DEFAULT);
    edge_sig             = Lp_over_channels(edge_sig, p_edge);
    edge_sig             = sigmoidal_activation(edge_sig, slope_edge, center_edge, weight_edge, 0);

    // Compute the even gabor filters    
    vector<Mat> ridge_sig = even_gabor(img, size, thetas, lambda_e, sigma_e, gamma_e, -1, cv::BORDER_DEFAULT);
    ridge_sig             = Lp_over_channels(ridge_sig, p_ridge);
    ridge_sig             = sigmoidal_activation(ridge_sig, slope_ridge, center_ridge, weight_ridge, 0);

    // Define loop constants
    int vec_size = dissimilarity.size();
    int rows = img.rows - 1;
    int cols = img.cols - 1;
    // Combine to compute alpha values
    for (int v_idx = 0; v_idx < vec_size; ++v_idx)
        for (int y_idx = 0; y_idx < rows; ++y_idx)
            for (int x_idx = 0; x_idx < cols; ++x_idx)
                dissimilarity[v_idx].at<double>(y_idx, x_idx) *= edge_sig[v_idx].at<double>(y_idx, x_idx) * ridge_sig[v_idx].at<double>(y_idx, x_idx);

    // Build and return the alpha tree
    at::AlphaTree retTree(dissimilarity[0], dissimilarity[1]);

    return retTree;
}


at::AlphaTree generate_cd_1dl_alphatree(Mat& img, double l_p,
                                        double p_edge, double slope_edge, double center_edge, double weight_edge,
                                        double p_ridge, double slope_ridge, double center_ridge, double weight_ridge)
{
    // Compute the dissimilarities
    vector<Mat> dissimilarity = dissimilarity_l_p_4(img, l_p);

    // Compute the odd gabor filters
    vector<Mat> edge_sig = central_difference_4(img);
    edge_sig             = Lp_over_channels(edge_sig, p_edge);
    edge_sig             = sigmoidal_activation(edge_sig, slope_edge, center_edge, weight_edge, 0);

    // Compute the even gabor filters    
    vector<Mat> ridge_sig = Laplacian_1d_4(img);
    ridge_sig             = Lp_over_channels(ridge_sig, p_ridge);
    ridge_sig             = sigmoidal_activation(ridge_sig, slope_ridge, center_ridge, weight_ridge, 0);

    // Define loop constants
    int vec_size = dissimilarity.size();
    int rows = img.rows - 1;
    int cols = img.cols - 1;
    // Combine to compute alpha values
    for (int v_idx = 0; v_idx < vec_size; ++v_idx)
        for (int y_idx = 0; y_idx < rows; ++y_idx)
            for (int x_idx = 0; x_idx < cols; ++x_idx)
                dissimilarity[v_idx].at<double>(y_idx, x_idx) *= edge_sig[v_idx].at<double>(y_idx, x_idx) * ridge_sig[v_idx].at<double>(y_idx, x_idx);

    // Build and return the alpha tree
    at::AlphaTree retTree(dissimilarity[0], dissimilarity[1]);

    return retTree;
}

at::AlphaTree generate_wilkinson_alphatree(Mat& img, double p_f, double w_f, double p_b, double w_b, double p_c, double w_c)
{
    // Compute the differences
    vector<Mat> for_dif  = forward_difference_4(img);
    for_dif              = Lp_over_channels(for_dif, p_f);
    vector<Mat> back_dif = backward_difference_4(img);
    back_dif             = Lp_over_channels(back_dif, p_b);
    vector<Mat> cent_dif = central_difference_4(img);
    cent_dif             = Lp_over_channels(cent_dif, p_c);

    vector<double> weights{w_f, w_b, w_c};
    // Compute the alpha values
    vector<Mat> alphas = compute_Wilkinson(for_dif, back_dif, cent_dif, weights);

    // Build and return the alpha tree
    at::AlphaTree retTree(alphas[0], alphas[1]);

    return retTree;
}


/**
 * @brief Compute the scores of an alpha-tree and a vector of ground truth images. Combine the computed scores as defined by the method.
 * 
 * @param img_tree The alpha-tree of which to compute the score.
 * @param ground_truths A vector of ground truth images used for score computation.
 * @param method An enum indicating how we want to compute the final score based on the individual scores of the ground truths.
 * @return The final score
 */
vector<double> compute_scores(at::AlphaTree img_tree, vector<Mat> ground_truths, enum combMeth method)
{
    // Compute the fitness score for each ground truth
    vector<double> scores;
    for (int idx = 0; idx < ground_truths.size(); ++idx)
    {
        double a_h;
        double score = at::fitness_function_quick(img_tree, ground_truths[idx], 0.7, 0.1, 0.1, 0.1, a_h);
        scores.push_back(score);
        scores.push_back(a_h);
        cerr << "score: " << score << '\n';
        cerr << "Optimal alpha: " << a_h << '\n';
    }

    // Combine the scores based on the cosen method
    double score;
    switch (method)
    {
        case ArithmaticMean:
            score = 0.0;
            for (int idx = 0; idx < scores.size(); idx += 2)
                score += scores[idx];
            score /= scores.size();
            break;
        case GeometricMean:
            score = 1.0;
            for (int idx = 0; idx < scores.size(); idx += 2)
                score *= scores[idx];
            score = pow(score, 1.0/(scores.size()/2));
            break;
    }

    cerr << "final score: " << score << '\n';

    // Add the final score to the first position of the vector
    scores.insert(scores.begin(), score);

    // Return the final score
    return scores;
}