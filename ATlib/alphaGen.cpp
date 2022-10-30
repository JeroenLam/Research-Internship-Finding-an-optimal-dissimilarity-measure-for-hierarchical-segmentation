#include "alphaGen.h"

using namespace std;
using namespace cv;

// ============================================================
//                    L_p norm over channels
// ============================================================
/**
 * @brief Compute the L_p norm of the vector define by the colour channels
 * 
 * @param img Input image (double with arbitrary channels)
 * @param p Norm to consider
 * @return A single channel matrix containing the normed values
 */
Mat Lp_over_channels(Mat& img, double p)
{
    // Allocate output matrix
    Mat outImg = Mat::zeros(img.rows, img.cols, CV_64F);

    // Define the number of channels
    int channels = img.channels();

    // Loop over all pixels
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
        {
            // Add all the vector components
            for (int c = 0; c < channels; ++c)
                outImg.at<double>(y, x) += pow(abs(img.at<double>(y, (x*channels) + c)), p);
            outImg.at<double>(y, x) = pow(outImg.at<double>(y, x), 1.0/p);
        }
    
    return outImg;
}

/**
 * @brief Compute the L_p norm of the vector define by the colour channels for each image in the vector
 * 
 * @param images Vector containing images (double with arbitrary channels)
 * @param p Norm to consider
 * @return A vector of single channel matrices containing the normed values 
 */
vector<Mat> Lp_over_channels(vector<Mat>& images, double p)
{
    vector<Mat> results;
    for (Mat img : images)
    {
        results.push_back(Lp_over_channels(img, p));
    }
    return results;
}


// ============================================================
//                         Gabor filters
// ============================================================
vector<Mat> compute_gabor(Mat& img, int size, vector<double>& thetas, double lambda, double sigma, double gamma, double psi, int ddepth, int borderType)
{
    // Allocate the return vector
    vector<Mat> retVec;

    // For each of the desired theta values
    for (double theta : thetas)
    {
        // Compute the filter
        Mat filterGabor = getGaborKernel(Size(size, size), sigma, theta, lambda, gamma, psi);

        // Allocate destination matrix
        Mat dst;

        // Apply the convolution
        filter2D(img, dst, ddepth, filterGabor, Point(-1,-1), 0, borderType);

        // Add filtered image to the return vector
        retVec.push_back(dst);
    }

    return retVec;
}

vector<Mat> even_gabor(Mat& img, int size, vector<double>& thetas, double lambda, double sigma, double gamma, int ddepth, int borderType)
{
    return compute_gabor(img, size, thetas, lambda, sigma, gamma, 0, ddepth, borderType);
}

vector<Mat> odd_gabor(Mat& img, int size, vector<double>& thetas, double lambda, double sigma, double gamma, int ddepth, int borderType)
{
    return compute_gabor(img, size, thetas, lambda, sigma, gamma, -0.5*CV_PI, ddepth, borderType);
}

// ============================================================
//                       Dissimilarities
// ============================================================
vector<Mat> dissimilarity_l_p_4(Mat& img, double p)
{
    // Allocate alpha arrays
    Mat alpha_h(img.rows - 1, img.cols - 1, CV_64F, 0.0);
    Mat alpha_v(img.rows - 1, img.cols - 1, CV_64F, 0.0);
    
    int channels = img.channels();

    // Compute the first row
    for (int x = 1; x < img.cols; ++x)
    {
        for (int c = 0; c < channels; ++c)
            alpha_h.at<double>(0, x-1) = pow(img.at<double>(0, (x*channels)+c) - img.at<double>(0, ((x-1)*channels)+c), p);
        alpha_h.at<double>(0, x-1) = pow(alpha_h.at<double>(0, x-1), 1.0/p);
    }

    // for all other rows
    for (int y = 1; y < img.rows; ++y)
    {
        // Compute the first collumn
        for (int c = 0; c < img.channels(); ++c)
            alpha_v.at<double>(y-1, 0) = pow(img.at<double>(y, c) - img.at<double>(y-1, c), p);
        alpha_v.at<double>(y-1, 0) = pow(alpha_v.at<double>(y-1, 0), 1.0/p);

        // for each column in the current row
        for (int x = 1; x < img.cols; ++x)
        {
            // Compute the vertical dissimilarity
            for (int c = 0; c < img.channels(); ++c)
                alpha_v.at<double>(y-1, x-1) = pow(img.at<double>(y, (x*channels)+c) - img.at<double>(y-1, (x*channels)+c), p);
            alpha_v.at<double>(y-1, x-1) = pow(alpha_v.at<double>(y-1, x-1), 1.0/p);
            
            // Compute the horizontal dissimilarity
            for (int c = 0; c < img.channels(); ++c)
                alpha_h.at<double>(y-1, x-1) = pow(img.at<double>(y, (x*channels)+c) - img.at<double>(y, ((x-1)*channels)+c), p);
            alpha_h.at<double>(y-1, x-1) = pow(alpha_h.at<double>(y-1, x-1), 1.0/p);
        }
    }

    // Combine and return alpha values
    vector<Mat> alphas;
    alphas.push_back(alpha_h);
    alphas.push_back(alpha_v);
    return alphas;
}


// ============================================================
//                      Sigmoidal activation
// ============================================================
/**
 * @brief Computes the element wise sigmoid activation using the fuction 
 *    y = @(x) ((1 / (1 + exp( -slope * ((x-center)/2) ))) * (upb - lwb)) + lwb
 * 
 * @param image The image of which we want to compute the activation
 * @param slope Slope of the sigmoid
 * @param center Curvature tipping point (i.e. middle of the sigmoid)
 * @param upb Upper bound of the response (default 1)
 * @param lwb Lower bound of the response (default 0)
 * @param inplace Defailt:false Set if you want the data in the input image to be the output image (default false)
 * @return Mat A matrix containing the resulting values of the sig
 */
Mat sigmoidal_activation(Mat &image, double slope, double center, double upb, double lwb, bool inplace)
{
    // Set output matrix
    Mat out = image;

    // If not inplace allocate output matrix
    if (!inplace)
    {
        Mat newOut(image.rows, image.cols, image.type());
        out = newOut;
    }

    int channels = image.channels();

    for (int row = 0; row < image.rows; ++row)
        for (int col = 0; col < image.cols; ++col)
            for (int c = 0; c < image.channels(); ++c)
            {
                double element = image.at<double>(row, (col*channels) + c);
                out.at<double>(row, (col*channels) + c) = ((1 / (1 + exp( -slope * ((element-center)/2) ))) * (upb - lwb)) + lwb;
            }
    return out;
}

/**
 * @brief Computes the element wise sigmoid activation using the fuction 
 *    y = @(x) ((1 / (1 + exp( -slope * ((x-center)/2) ))) * (upb - lwb)) + lwb
 * 
 * @param images The images of which we want to compute the activation
 * @param slope Slope of the sigmoid
 * @param center Curvature tipping point (i.e. middle of the sigmoid)
 * @param upb Upper bound of the response (default 1)
 * @param lwb Lower bound of the response (default 0)
 * @param inplace Defailt:false Set if you want the data in the input image to be the output image (default false)
 * @return Mat A matrix containing the resulting values of the sig
 */
vector<Mat> sigmoidal_activation(vector<Mat> images, double slope, double center, double upb, double lwb, bool inplace)
{
    // Allocate an return matrix
    vector<Mat> retVec;

    // Fill the return matrix
    for (Mat image : images)
        retVec.push_back(sigmoidal_activation(image, slope, center, upb, lwb, inplace));

    // Return the result
    return retVec;
}


// ============================================================
//                           CD & 1dL
// ============================================================
/**
 * @brief Compute the backwards difference of an image
 * 
 * @param img 
 * @param direction 
 * @return Mat 
 */
Mat backward_difference(Mat& img, int direction)
{
    Mat kernel;
    switch (direction)
    {
    case 0:
        kernel = (Mat_<double>(1,3) <<  -1, 1, 0);
        break;
    case 1:
        kernel = (Mat_<double>(3,3) <<  -1, 0, 0, 0, 1, 0, 0, 0, 0);
        break;
    case 2:
        kernel = (Mat_<double>(3,1) <<  -1, 1, 0);
        break;
    case 3:
        kernel = (Mat_<double>(3,3) <<  0, 0, -1, 0, 1, 0, 0, 0, 0);
        break;
    default:
        cerr << "abs_backward_difference only supports directions {1,2,3,4}\n";
        assert(false);
        break;
    }

    // Allocate destination matrix
    Mat dst;

    // Apply the convolution
    filter2D(img, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    // Return the result
    return dst;
}

/**
 * @brief 
 * 
 * @param img 
 * @param direction 
 * @return Mat 
 */
Mat forward_difference(Mat& img, int direction)
{
    Mat kernel;
    switch (direction)
    {
    case 0:
        kernel = (Mat_<double>(1,3) <<  0, -1, 1);
        break;
    case 1:
        kernel = (Mat_<double>(3,3) <<  0, 0, 0, 0, -1, 0, 0, 0, 1);
        break;
    case 2:
        kernel = (Mat_<double>(3,1) <<  0, -1, 1);
        break;
    case 3:
        kernel = (Mat_<double>(3,3) <<  0, 0, 0, 0, -1, 0, 1, 0, 0);
        break;
    default:
        cerr << "abs_forward_difference only supports directions {1,2,3,4}\n";
        assert(false);
        break;
    }

    // Allocate destination matrix
    Mat dst;

    // Apply the convolution
    filter2D(img, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    // Return the result
    return dst;
}

/**
 * @brief 
 * 
 * @param img 
 * @param direction 
 * @return Mat 
 */
Mat central_difference(Mat& img, int direction)
{
    Mat kernel;
    switch (direction)
    {
    case 0:
        kernel = (Mat_<double>(1,3) <<  -1, 0, 1);
        break;
    case 1:
        kernel = (Mat_<double>(3,3) <<  -1, 0, 0, 0, 0, 0, 0, 0, 1);
        break;
    case 2:
        kernel = (Mat_<double>(3,1) <<  -1, 0, 1);
        break;
    case 3:
        kernel = (Mat_<double>(3,3) <<  0, 0, -1, 0, 0, 0, 1, 0, 0);
        break;
    default:
        cerr << "abs_central_difference only supports directions {1,2,3,4}\n";
        assert(false);
        break;
    }

    // Allocate destination matrix
    Mat dst;

    // Apply the convolution
    filter2D(img, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    // Return the result
    return dst;
}

/**
 * @brief 
 * 
 * @param img 
 * @param direction 
 * @return Mat 
 */
Mat Laplacian_1d(Mat& img, int direction)
{
    Mat kernel;
    switch (direction)
    {
    case 0:
        kernel = (Mat_<double>(1,3) <<  -1, 2, -1);
        break;
    case 1:
        kernel = (Mat_<double>(3,3) <<  -1, 0, 0, 0, 2, 0, 0, 0, -1);
        break;
    case 2:
        kernel = (Mat_<double>(3,1) <<  -1, 2, -1);
        break;
    case 3:
        kernel = (Mat_<double>(3,3) <<  0, 0, -1, 0, 2, 0, -1, 0, 0);
        break;
    default:
        cerr << "abs_central_difference only supports directions {1,2,3,4}\n";
        assert(false);
        break;
    }

    // Allocate destination matrix
    Mat dst;

    // Apply the convolution
    filter2D(img, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    // Return the result
    return dst;
}

/**
 * @brief Compute a bank of backward difference signals for 4 connectivity
 * 
 * @param img The image of which to compute the signals
 * @return vector<Mat> 
 */
vector<Mat> backward_difference_4(Mat& img)
{
    vector<Mat> retVec;
    retVec.push_back(backward_difference(img, 0));
    retVec.push_back(backward_difference(img, 2));
    return retVec;
}

/**
 * @brief Compute a bank of forward difference signals for 4 connectivity
 * 
 * @param img The image of which to compute the signals
 * @return vector<Mat> 
 */
vector<Mat> forward_difference_4(Mat& img)
{
    vector<Mat> retVec;
    retVec.push_back(forward_difference(img, 0));
    retVec.push_back(forward_difference(img, 2));
    return retVec;
}

/**
 * @brief Compute a bank of central difference signals for 4 connectivity
 * 
 * @param img The image of which to compute the signals
 * @return vector<Mat> 
 */
vector<Mat> central_difference_4(Mat& img)
{
    vector<Mat> retVec;
    retVec.push_back(central_difference(img, 0));
    retVec.push_back(central_difference(img, 2));
    return retVec;
}

/**
 * @brief Compute a bank of 1d Laplacian signals for 4 connectivity
 * 
 * @param img The image of which to compute the signals
 * @return vector<Mat> 
 */
vector<Mat> Laplacian_1d_4(Mat& img)
{
    vector<Mat> retVec;
    retVec.push_back(Laplacian_1d(img, 0));
    retVec.push_back(Laplacian_1d(img, 2));
    return retVec;
}


// ============================================================
//                      Wilkinson Detector
// ============================================================
vector<Mat> compute_Wilkinson(vector<Mat> for_dif, vector<Mat> back_dif, vector<Mat> cent_dif, vector<double> weigths)
{
    // Allocate alpha arrays
    vector<Mat> alphas;

    // For each direction
    for (int idx = 0; idx < 2; ++idx)
    {
        // Create an empty alpha image
        alphas.emplace_back(for_dif[0].rows - 1, for_dif[0].cols - 1, CV_64F, 0.0);

        // For each pixel in the alpha matrix
        for (int y = 0; y < alphas[idx].rows; ++y)
            for (int x = 0; x < alphas[idx].cols; ++x)
            {
                alphas[idx].at<double>(y,x) = abs(weigths[0] * for_dif[idx].at<double>(y,x) + 
                                                  weigths[1] * back_dif[idx].at<double>(y,x) + 
                                                  weigths[2] * cent_dif[idx].at<double>(y,x));
            }
    }
    
    return alphas;
}


// ============================================================
//                       Example code
// ============================================================
double OrthogonalEdgeWeight = 1.0;
double MainEdgeWeight = 1.0;
double RGBweight[3] = {0.5, 0.5, 0.5};

vector<Mat> alpha_example(Mat& img)
{
    // Allocate alpha arrays
    Mat alpha_h(img.rows - 1, img.cols - 1, CV_64F, 0.0);
    Mat alpha_v(img.rows - 1, img.cols - 1, CV_64F, 0.0);
    
    // Compute the first row
    for (int x = 1; x < img.cols; ++x)
        alpha_h.at<double>(0, x-1) = EdgeStrengthX(img, x, 0);

    // for all other rows
    for (int y = 1; y < img.rows; ++y)
    {
        alpha_v.at<double>(y-1, 0) = EdgeStrengthY(img, 0, y);

        // for each column in the current row
        for (int x = 1; x < img.cols; ++x)
        {
            alpha_v.at<double>(y-1, x-1) = EdgeStrengthY(img, x, y);
            // repeat process in x-direction
            alpha_h.at<double>(y-1, x-1) = EdgeStrengthX(img, x, y);
        }
    }

    // Combine and return alpha values
    vector<Mat> alphas;
    alphas.push_back(alpha_h);
    alphas.push_back(alpha_v);
    return alphas;
}

/**
 * @brief Computes the salience between two pixels using an unweighted average.
 * computes sqrt(sum((P1 - P2)^2))
 * 
 * @param p First Pixel
 * @param q Second Pixel
 * @return double result of the computation
 */
double simpleSalience(Vec3b p, Vec3b q)
{
    double result = 0;
    for (int i = 0; i < 3; ++i)
        result += ((double)p[i] - (double)q[i]) * ((double)p[i] - (double)q[i]);
    return sqrt(result);
}

/**
 * @brief Computes the salience between two pixels using a weighted average.
 * computes sqrt(sum(W*(P1 - P2)^2))
 * 
 * @param p First Pixel
 * @param q Second Pixel
 * @return double result of the computation
 */
double WeightedSalience(Vec3b p, Vec3b q)
{
    double result = 0;
    for (int i = 0; i < 3; ++i)
        result += RGBweight[i] * ((double)p[i] - (double)q[i]) * ((double)p[i] - (double)q[i]);
    return sqrt(result);
}

/**
 * @brief Computes the edge strength in the x direction at a given position (x,y)
 * 
 * @param img Image to compute in
 * @param x x-coordinate of the position
 * @param y y-coordinate of the position
 * @return double The edge strength
 */
double EdgeStrengthX(Mat img, int x, int y)
{
    int yminus1 = y - (y > 0);
    int yplus1 = y + (y < img.rows - 1);

    // We use the minimum salience between the sourrounding rows at (x-1) and x
    double temp1 = WeightedSalience(img.at<Vec3d>(yminus1, x-1), img.at<Vec3d>(yplus1, x-1));
    double temp2 = WeightedSalience(img.at<Vec3d>(yminus1, x  ), img.at<Vec3d>(yplus1, x  ));
    double ygrad = (temp1 > temp2 ? temp2 : temp1);

    return (
        OrthogonalEdgeWeight * 
        ygrad + 
        MainEdgeWeight *
        WeightedSalience(img.at<Vec3d>(y, x-1), img.at<Vec3d>(y, x))
    );
}

/**
 * @brief Computes the edge strength in the y direction at a given position (x,y)
 * 
 * @param img Image to compute in
 * @param x x-coordinate of the position
 * @param y y-coordinate of the position
 * @return double The edge strength
 */
double EdgeStrengthY(Mat img, int x, int y)
{
    int xminus1 = x - (x > 0);
    int xplus1 = x + (x < img.cols - 1);

    // We use the minimum salience between the sourrounding columns at (y-1) and y
    double temp1 = WeightedSalience(img.at<Vec3d>(y  , xplus1), img.at<Vec3d>(y  , xminus1));
    double temp2 = WeightedSalience(img.at<Vec3d>(y-1, xplus1), img.at<Vec3d>(y-1, xminus1));

    double xgrad = (temp1 > temp2 ? temp2 : temp1);

    return (
        OrthogonalEdgeWeight * 
        xgrad + 
        MainEdgeWeight *
        WeightedSalience(img.at<Vec3d>(y-1, x), img.at<Vec3d>(y, x))
    );
}
