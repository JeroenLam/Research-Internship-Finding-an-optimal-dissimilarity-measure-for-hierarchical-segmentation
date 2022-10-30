#include <opencv2/imgcodecs.hpp>    // Reading images from disk
#include <iostream>
#include "ATlib/simulation.h"

using namespace std;

int main(int argc, char** argv)
{
    // ==============================================================================
    //                        User input parser and path control
    // ==============================================================================
    // If no parameters are provided
    int method = 0; // Gabor, CDL, Wilkinson
    int mode = 0;   // Compute score, Evaluate alpha
    if (argc < 1)
    {
        cout << "Run '-h' for information on how to run the program\n";
        return 0;
    }
    else
    {
        string arg1(argv[1]);
        if (arg1.compare("-h") == 0)
        {
            if (argc == 2)
            {
                cout << "Run '-h <Gabor, CDL, Wilkinson>' for the parameter information for each mode\n";
                return 0;
            }
            string arg2(argv[2]);
            if (arg2.compare("Gabor") == 0 || arg2.compare("gabor") == 0)
            {
                cout << "Run the following to compute the scores based on the provided ground truths and filter parameters:\n";
                cout << "  <program name> Gabor <img path> <#gt> (<gt1 path> ... <gt(n) path>) <l_p dissimilarity> <lambda odd> <sigma odd> <gamma odd> <l_p odd> <slope odd> <center odd> <weight odd> <lambda even> <sigma even> <gamma even> <l_p even> <slope even> <center even> <weight even>\n'";
                cout << "Run the following to evaluta the filter at multiple alpha levels and store the resulting images to disk:\n";
                cout << "  <program name> Gabor <img path> -<#alphas> (<alpha(1)> ... <alpha(n)>) <l_p dissimilarity> <lambda odd> <sigma odd> <gamma odd> <l_p odd> <slope odd> <center odd> <weight odd> <lambda even> <sigma even> <gamma even> <l_p even> <slope even> <center even> <weight even>\n";
                return 0;
            }
            else if (arg2.compare("CDL") == 0 || arg2.compare("Cdl") == 0 || arg2.compare("cdl") == 0)
            {
                cout << "Run the following to compute the scores based on the provided ground truths and filter parameters:\n";
                cout << "  <program name> CDL <img path> <#gt> (<gt1 path> ... <gt(n) path>) <l_p dissimilarity> <l_p edge> <slope edge> <center edge> <weight edge> <l_p ridge> <slope ridge> <center ridge> <weight ridge>\n";
                cout << "Run the following to evaluta the filter at multiple alpha levels and store the resulting images to disk:\n";
                cout << "  <program name> CDL <img path> -<#alphas> (<alpha(1)> ... <alpha(n)>) <l_p dissimilarity> <l_p edge> <slope edge> <center edge> <weight edge> <l_p ridge> <slope ridge> <center ridge> <weight ridge>\n";
                return 0;
            }
            else if (arg2.compare("Wilkinson") == 0 || arg2.compare("wilkinson") == 0)
            {
                cout << "Run the following to compute the scores based on the provided ground truths and filter parameters:\n";
                cout << "  <program name> Wilkinson <img path> <#gt> (<gt1 path> ... <gt(n) path>) <l_p over FD> <weight of FD> <l_p over BD> <weight of BD> <l_p over CD> <weight of CD>\n'";
                cout << "Run the following to evaluta the filter at multiple alpha levels and store the resulting images to disk:\n";
                cout << "  <program name> Wilkinson <img path> -<#alphas> (<alpha(1)> ... <alpha(n)>) <l_p over FD> <weight of FD> <l_p over BD> <weight of BD> <l_p over CD> <weight of CD>\n'";
                return 0;
            }
        }
        else if (arg1.compare("Gabor") == 0 || arg1.compare("gabor") == 0)
        {
            method = 1;
        }
        else if (arg1.compare("CDL") == 0 || arg1.compare("Cdl") == 0 || arg1.compare("cdl") == 0)
        {
            method = 2;
        }
        else if (arg1.compare("Wilkinson") == 0 || arg1.compare("wilkinson") == 0)
        {
            method = 3;
        }
        else
        {
            cout << "Run '-h <Gabor, CDL, Wilkinson>' for the parameter information for each method\n";
            return 0;
        }
    }

    // ==============================================================================
    //                        Loading images and preprocessing
    // ==============================================================================
    if (argc < 5)
    {
        cout << "Please provide the correct parameters, see '-h' for more info. (parsing input images)\n";
        return 0;
    }
    // Parse the image path
    string img_path(argv[2]);
    // Parse the number of ground truths / alpha values
    int num_gt = atoi(argv[3]);
    // Check if we are scoring or evaluating the filter
    if (num_gt < 0)
    {
        mode = 1;
        num_gt *= -1;
    }
    // Define an offset to be used for further aprameter parsing
    int par_off = num_gt + 4;

    // Load image
    cv::Mat img = cv::imread(img_path);
    img.convertTo(img, CV_64FC3);
    // If scoring, parse ground truths, otherwise parse alpha values
    vector<cv::Mat> groundTruths;
    vector<double> alphas;
    if (mode == 0)
        for (int idx = 0; idx < num_gt; ++idx)
        {
            cv::Mat img_gt = cv::imread(argv[4 + idx]);
            groundTruths.push_back(img_gt);
        }
    else if (mode == 1)
        for (int idx = 0; idx < num_gt; ++idx)
            alphas.push_back(atof(argv[4 + idx]));

    // ==============================================================================
    //                            Compute the alpha values
    // ==============================================================================
    at::AlphaTree tree;
    if (method == 1)      // Gabor
    {
        double p        = atof(argv[par_off]);
        double lambda_o = atof(argv[par_off+1]);
        double sigma_o  = atof(argv[par_off+2]);
        double gamma_o  = atof(argv[par_off+3]);
        double p_o      = atof(argv[par_off+4]);
        double slope_o  = atof(argv[par_off+5]);
        double center_o = atof(argv[par_off+6]);
        double weight_o = atof(argv[par_off+7]);
        double lambda_e = atof(argv[par_off+8]);
        double sigma_e  = atof(argv[par_off+9]);
        double gamma_e  = atof(argv[par_off+10]);
        double p_e      = atof(argv[par_off+11]);
        double slope_e  = atof(argv[par_off+12]);
        double center_e = atof(argv[par_off+13]);
        double weight_e = atof(argv[par_off+14]);
        tree = generate_gabor_alphatree(img, p,
                                        lambda_o, sigma_o, gamma_o,
                                        p_o, slope_o, center_o, weight_o,
                                        lambda_e, sigma_e, gamma_e,
                                        p_e, slope_e, center_e, weight_e);
    }
    else if (method == 2)     // CDL
    {
        double p            = atof(argv[par_off]);
        double p_edge       = atof(argv[par_off+1]);
        double slope_edge   = atof(argv[par_off+2]);
        double center_edge  = atof(argv[par_off+3]);
        double weight_edge  = atof(argv[par_off+4]);
        double p_ridge      = atof(argv[par_off+5]);
        double slope_ridge  = atof(argv[par_off+6]);
        double center_ridge = atof(argv[par_off+7]);
        double weight_ridge = atof(argv[par_off+8]);
        tree = generate_cd_1dl_alphatree(img, p,
                                         p_edge, slope_edge, center_edge, weight_edge,
                                         p_ridge, slope_ridge, center_ridge, weight_ridge);
    }
    else if (method == 3)     // Wilkinson
    {
        double p_f = atof(argv[par_off]);
        double w_f = atof(argv[par_off+1]);
        double p_b = atof(argv[par_off+2]);
        double w_b = atof(argv[par_off+3]);
        double p_c = atof(argv[par_off+4]);
        double w_c = atof(argv[par_off+5]);

        tree = generate_wilkinson_alphatree(img, p_f, w_f, p_b, w_b, p_c, w_c);
    }

    // ==============================================================================
    //                               Compute the score
    // ==============================================================================
    if (mode == 0)
    {
        // Compute the scores for each ground truth and combine using a geometric mean
        vector<double> scores = compute_scores(tree, groundTruths, GeometricMean);
        // Output the result to cout to be parsed by the python script
        for (int idx = 0; idx < scores.size(); ++idx)
        {
            cout << scores[idx];
            if (idx == scores.size()-1)
                cout << '\n';
            else
                cout << ',';
        }
    }   
    // ==============================================================================
    //                                Filter the tree
    // ==============================================================================
    else if (mode == 1)
    {
        // Remove extension from path
        int start = img_path.rfind('/');
        start = start == string::npos ? 0 : start + 1;
        int end = img_path.rfind('.');
        end = end == string::npos ? img_path.size() : end;
        string img_name(img_path, start, end - start);
        
        string method_name = string(argv[1]);

        cerr << tree.maxAlpha() << '\n';

        // For each alpha value
        for (double alpha : alphas)
        {        
            // Filter the tree
            cv::Mat filtered_img = tree.SalienceFilter(alpha);

            // Define the file name
            string out_path = "../img/output/";
            string out_name = img_name + '_' + method_name + '_' + to_string(alpha) + ".png";

            // Store the image to disk
            cv::imwrite(out_path + out_name, filtered_img);
        }
    }
}