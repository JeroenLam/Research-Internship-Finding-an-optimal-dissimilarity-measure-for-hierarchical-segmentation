#include "at-fitness.h"

using namespace std;
using namespace cv;

/**
 * @brief A function that can be used to evaluate how good a ground truth image can be segmented from the provided alpha tree.
 * 
 * @param tree The alpha tree to evaluate
 * @param img_gt The ground truth to evaluate
 * @param w_AS The weight for the Area Score
 * @param w_NI The weight for the Node Index
 * @param w_DI The weight for the Depth Index
 * @param w_PLI The weight for the Plateau Length Index
 * @param connectivity The connectivity of 
 * @return double 
 */
double at::fitness_function(at::AlphaTree& tree, Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI)
{
    double temp;
    return fitness_function(tree, img_gt, w_AS, w_NI, w_DI, w_PLI, temp);
}

/**
 * @brief A function that can be used to evaluate how good a ground truth image can be segmented from the provided alpha tree.
 * 
 * @param tree The alpha tree to evaluate
 * @param img_gt The ground truth to evaluate
 * @param w_AS The weight for the Area Score
 * @param w_NI The weight for the Node Index
 * @param w_DI The weight for the Depth Index
 * @param w_PLI The weight for the Plateau Length Index
 * @param connectivity The connectivity of 
 * @param a_h Return parameter, will be set to the optimal alpha value
 * @return double 
 */
double at::fitness_function(at::AlphaTree& tree, Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI, double& a_h)
{
    // Get all alpha values of the tree
    vector<double> alphas = tree.getAlphas();

    // Allocate vector to store the area scores
    vector<double> vec_AS;

    // Precompute lookup tables with the sizes of the connencted components for the ground truth
    map<int, int> cc_img_gt;
    for (int y = 0; y < img_gt.rows; ++y)
        for (int x = 0; x < img_gt.cols; ++x)
        {
            int col_img = 0;
            // Compute colour value
            for (int c = 0; c < img_gt.channels(); ++c)
                col_img += pow(256, c) * img_gt.at<uchar>(y, (x*img_gt.channels())+c);

            // Add to lookup table (map)
            if (cc_img_gt.count(col_img) > 0)
                cc_img_gt[col_img] += 1;
            else
                cc_img_gt[col_img] = 1;
        }

    // For each alpha, compute the area score
    for (int idx = 0; idx < alphas.size(); ++idx)
    {
        cout << idx << "/" << alphas.size() << '\r';
        // Filter the image
        Mat img = tree.SalienceFilter(alphas[idx]);
        
        // Compute the lookup map for the segmented image
            map<int, int> cc_img_f;
            for (int y = 0; y < img.rows; ++y)
                for (int x = 0; x < img.cols; ++x)
                {
                    int col_img = 0;
                    // Compute colour value
                    for (int c = 0; c < img.channels(); ++c)
                        col_img += pow(256, c) * img.at<uchar>(y, (x*img.channels())+c);

                    // Add to lookup table (map)
                    if (cc_img_f.count(col_img) > 0)
                        cc_img_f[col_img] += 1;
                    else
                        cc_img_f[col_img] = 1;
                }

        // Compute the area score w.r.t. the ground truth and add to vector
        double temp_as = area_score(img, cc_img_f, img_gt, cc_img_gt, tree.getConnectivity());
        vec_AS.push_back(temp_as);
    }

    // Find the best (highest) area score index
    auto it     = max_element(vec_AS.begin(), vec_AS.end());
    int idx_h   = it - vec_AS.begin();
    a_h  = alphas[idx_h];
    double AS_h = vec_AS[idx_h];

    // Compute the NodeIndex and the DepthIndex
    double NI_h = node_index(tree);
    double DI_h = depth_index(tree, a_h);

    // Compute the plateau length index
    double PLI_h = plateau_length_index(tree, alphas, vec_AS, AS_h, idx_h);

    cout << "Alpha, Area Score\n";
    for (int idx = 0; idx < alphas.size(); ++idx)
        cout << alphas[idx] << ',' << vec_AS[idx] << '\n';

    cout << "AS,NI,DI,PLI\n";
    cout << AS_h << ',' << NI_h << ',' << DI_h << ',' << PLI_h << '\n';

    // Combine the scores and return
    return (AS_h * w_AS) + (NI_h * w_NI) + (DI_h * w_DI) + (PLI_h * w_PLI);
}

/**
 * @brief A function that can be used to evaluate how good a ground truth image can be segmented from the provided alpha tree. It checks a smaller set of the alphas by stepping from zero with a step of a tenth of the stddev.
 * 
 * @param tree The alpha tree to evaluate
 * @param img_gt The ground truth to evaluate
 * @param w_AS The weight for the Area Score
 * @param w_NI The weight for the Node Index
 * @param w_DI The weight for the Depth Index
 * @param w_PLI The weight for the Plateau Length Index
 * @param connectivity The connectivity of 
 * @return double 
 */
double at::fitness_function_quick(at::AlphaTree& tree, Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI)
{
    double temp;
    return fitness_function_quick(tree, img_gt, w_AS, w_NI, w_DI, w_PLI, temp);
}

/**
 * @brief A function that can be used to evaluate how good a ground truth image can be segmented from the provided alpha tree. It checks a smaller set of the alphas by stepping from zero with a step of a tenth of the stddev.
 * 
 * @param tree The alpha tree to evaluate
 * @param img_gt The ground truth to evaluate
 * @param w_AS The weight for the Area Score
 * @param w_NI The weight for the Node Index
 * @param w_DI The weight for the Depth Index
 * @param w_PLI The weight for the Plateau Length Index
 * @param connectivity The connectivity of 
 * @param a_h Return parameter, will be set to the optimal alpha value
 * @return double 
 */
double at::fitness_function_quick(at::AlphaTree& tree, Mat& img_gt, double w_AS, double w_NI, double w_DI, double w_PLI, double& a_h)
{
    // Get all alpha values of the tree
    vector<double> alphas = tree.getAlphas();

    // Allocate vector to store the area scores
    vector<double> vec_AS;

    // Defining the number of steps to consider
    int steps = SIM_STEPS;

    // Precompute lookup tables with the sizes of the connencted components for the ground truth
    map<int, int> cc_img_gt;
    for (int y = 0; y < img_gt.rows; ++y)
        for (int x = 0; x < img_gt.cols; ++x)
        {
            int col_img = 0;
            // Compute colour value
            for (int c = 0; c < img_gt.channels(); ++c)
                col_img += pow(256, c) * img_gt.at<uchar>(y, (x*img_gt.channels())+c);

            // Add to lookup table (map)
            if (cc_img_gt.count(col_img) > 0)
                cc_img_gt[col_img] += 1;
            else
                cc_img_gt[col_img] = 1;
        }


    if (steps > alphas.size())
    {
        cerr << "      Using all alphas (" << steps <<" > " << alphas.size() << ")\n";
        // For each alpha, compute the area score
        for (int idx = 0; idx < alphas.size(); ++idx)
        {
            if (idx % 20 == 0)
                cerr << "Iteration " << idx << '/' << alphas.size() << '\n';

            // Filter the image
            Mat img = tree.SalienceFilter(alphas[idx]);
            
            // Compute the lookup map for the segmented image
            map<int, int> cc_img_f;
            for (int y = 0; y < img.rows; ++y)
                for (int x = 0; x < img.cols; ++x)
                {
                    int col_img = 0;
                    // Compute colour value
                    for (int c = 0; c < img.channels(); ++c)
                        col_img += pow(256, c) * img.at<uchar>(y, (x*img.channels())+c);

                    // Add to lookup table (map)
                    if (cc_img_f.count(col_img) > 0)
                        cc_img_f[col_img] += 1;
                    else
                        cc_img_f[col_img] = 1;
                }

            // Compute the area score w.r.t. the ground truth and add to vector
            vec_AS.push_back(area_score(img, cc_img_f, img_gt, cc_img_gt, tree.getConnectivity()));
        }
    }
    else
    {
        cerr << "      Using a subset of alphas (" << steps <<" < " << alphas.size() << ")\n";
        // Compute a subset of alphas based on the stddev
        // Allocate new alpha array
        vector<double> newAlphas;


        for (int iter = 0; iter < steps; ++iter)
        {
            if (iter % 20 == 0)
                cerr << "Iteration " << iter << '/' << steps-1 << '\n';

            // Compute the relavant index
            int idx = (alphas.size() - 1) * iter / (steps - 1);

            // Filter the image
            Mat img = tree.SalienceFilter(alphas[idx]);
            
            // Compute the lookup map for the segmented image
            map<int, int> cc_img_f;
            for (int y = 0; y < img.rows; ++y)
                for (int x = 0; x < img.cols; ++x)
                {
                    int col_img = 0;
                    // Compute colour value
                    for (int c = 0; c < img.channels(); ++c)
                        col_img += pow(256, c) * img.at<uchar>(y, (x*img.channels())+c);

                    // Add to lookup table (map)
                    if (cc_img_f.count(col_img) > 0)
                        cc_img_f[col_img] += 1;
                    else
                        cc_img_f[col_img] = 1;
                }

            // Compute the area score w.r.t. the ground truth and add to vector
            vec_AS.push_back(area_score(img, cc_img_f, img_gt, cc_img_gt, tree.getConnectivity()));
            newAlphas.push_back(alphas[idx]);
        }
        alphas = newAlphas;    
    }

    // Find the best (highest) area score index
    auto it     = max_element(vec_AS.begin(), vec_AS.end());
    int idx_h   = it - vec_AS.begin();
    a_h         = alphas[idx_h];
    double AS_h = vec_AS[idx_h];

    // Compute the NodeIndex and the DepthIndex
    double NI_h = node_index(tree);
    double DI_h = depth_index(tree, a_h);

    // Compute the plateau length index
    double PLI_h = plateau_length_index(tree, alphas, vec_AS, AS_h, idx_h);

    // Combine the scores and return
    cerr << "AS_h: " << AS_h << " - NI_h: " << NI_h << " - DI_h: " << DI_h << " - PLI_h: " << PLI_h << '\n'; 
    return (AS_h * w_AS) + (NI_h * w_NI) + (DI_h * w_DI) + (PLI_h * w_PLI);
}

/**
 * @brief Compute the node index of a alpha tree
 * 
 * @param tree The tree of which to compute the node index
 * @param connectivity The connectivity of the alpha tree
 * @return The node index
 */
double at::node_index(at::AlphaTree& tree)
{
    int connectivity = tree.getConnectivity();
    if (connectivity == 4) 
        return 1 - ((double)tree.size() / (double)(2 * tree.imgSize()));
    else
    {
        cerr << "node_index(): connectivity 8 not implemented yet!\n";
        assert(false);
    }
}

/**
 * @brief Compute the depth index of an alpha tree given an optimal alpha value
 * 
 * @param tree The tree of which to compute the depth index
 * @param a_h The optimal alpha value
 * @return The depth index
 */
double at::depth_index(at::AlphaTree& tree, double a_h)
{
    if (tree.maxAlpha() == 0)
        return 1;
    return 1 - (a_h / tree.maxAlpha());
}

/**
 * @brief Compute the plateau length of an alpha tree
 * 
 * @param tree The tree of which to compute the PLI
 * @param alphas A vector of alpha values
 * @param vec_AS A vector of area scores corresponding to the alpha values
 * @param AS_h The highest area score
 * @param idx_h The index of the highest area score and corresponding alpha
 * @return the plateau length index
 */
double at::plateau_length_index(at::AlphaTree& tree, vector<double>& alphas, vector<double>& vec_AS, double AS_h, int idx_h)
{
    // Find the AccuracyPlateau region PLI
    // Find the upper bound
    int up_idx = idx_h;
    for (; up_idx < vec_AS.size(); ++up_idx)
        if (vec_AS[up_idx] < 0.9 * AS_h)
        {
            --up_idx;
            break;
        }
    if (up_idx >= vec_AS.size())
        up_idx = vec_AS.size() - 1;

    // Find the lower bound
    int lw_idx = idx_h;
    for (; lw_idx >= 0; --lw_idx)
        if (vec_AS[lw_idx] < 0.9 * AS_h)
        {
            ++lw_idx;
            break;
        }
    if (lw_idx < 0)
        lw_idx = 0;

   
    double alpha_range = abs(alphas[up_idx] - alphas[lw_idx]);

    // Compute the plateau length index
    if (alphas.back() == 0)
        return 1;

    return alpha_range / alphas.back();
}


/**
 * @brief Compute the are score of an segmentation and its ground truth
 * Note that it is assumed that the images are of type CV_UC3 and that each component has a regionally unique colour
 * 
 * @param img_f The segmentation
 * @param img_gt The ground truth
 * @return doubel 
 */
double at::area_score(Mat& img_f, map<int,int>& cc_img_f, Mat& img_gt, map<int,int>& cc_img_gt, int connectivity)
{
    // Compute the under and over merging score
    double UM = under_merging_error(img_f, img_gt, cc_img_gt, connectivity); 
    double OM = over_merging_error(img_f, img_gt, cc_img_f, connectivity);

    // Return the area score
    return 1 - sqrt((UM*UM) + (OM*OM));
    
}

/**
 * @brief Computes the under merging score. This function is used in the area_score() function
 * 
 * @param img_f The segmentation
 * @param img_gt The ground truth
 * @return double 
 */
double at::under_merging_error(Mat& img_f, Mat& img_gt, map<int,int>& cc_img_gt, int connectivity)
{
    return merging_error(img_f, img_gt, cc_img_gt, connectivity);
}

/**
 * @brief Computes the over merging score. This function is used in the area_score() function
 * 
 * @param img_f The segmentation
 * @param img_gt The ground truth
 * @return double 
 */
double at::over_merging_error(Mat& img_f, Mat& img_gt, map<int,int>& cc_img_f, int connectivity)
{
    return merging_error(img_gt, img_f, cc_img_f, connectivity);
}

/**
 * @brief Computes the Over-merging error or the undermerging error
 * @param img1 The segmentation or the ground truth (uchar images with arbitrary number of channels)
 * @param img2 The ground truth or the segmentation (uchar images with arbitrary number of channels)
 * @return double 
 */
double at::merging_error(Mat& img1, Mat& img2, map<int,int>& cc_img2, int connectivity)
{
    // For every component in img1, find the component with the larges overlap in img2
    Mat mask(img1.rows, img1.cols, CV_8U);
    mask = 0;

    double score = 0.0;

    for (int y_idx = 0; y_idx < img1.rows; ++y_idx)
        for (int x_idx = 0; x_idx < img1.cols; ++x_idx)
        {
            // If we have not visited this pixel
            if (mask.at<uchar>(y_idx, x_idx) == 0)
            {
                // Find component size and the size of the larges intersection
                // Do a BFS to find the component size and count the colours in img2
                map<int, int> intersectionSize;
                if (connectivity == 4)
                    intersectionSize = at::cc_bfs_4(img1, img2, mask, x_idx, y_idx);
                else if (connectivity == 8)
                    intersectionSize = at::cc_bfs_8(img1, img2, mask, x_idx, y_idx);
                else
                {
                    cerr << "Please provide a connectivity of 4 or 8\n";
                    assert(false);
                }

                // Loop over the intersectionSize map and find the larges component
                int size_inter = 0;
                int col_largest = 0;
                for (map<int, int>::iterator it = intersectionSize.begin(); it != intersectionSize.end(); ++it)
                    if (it->second > size_inter)
                    {
                        col_largest = it->first;
                        size_inter = it->second;
                    }

                // Get the size of the largest intersecting segment in img2
                int size_cc = cc_img2[col_largest];

                // Compute the contribution to the final score
                score += (double)((size_cc - size_inter) * size_inter) / (double)size_cc;
            }
        }

    // Normalise using the size of the image (since the image and ground truth are the same size)
    return score / img2.total();
}

/**
 * @brief Does a breath first search in order to find the connected component. It also keeps track of the encountered colours in img2 in order to find the larges intersecting component.
 * 
 * @param img1 The segmentation or the ground truth
 * @param img2 The ground truth or the segmentation
 * @param mask Matrix denoting the visited pixels
 * @param x x coordinate of the starting position
 * @param y y coordinate of the starting position
 * @param col Colour of the current connected component
 * @param colmap Return parameter, map containing the encountered colours, i.e. CC in img2
 * @return int Size of the connected component
 */
map<int, int> at::cc_bfs_4(Mat& img1, Mat& img2, Mat& mask, int x, int y)
{
    // Initialise the colour map
    map<int, int> colmap;

    // Precompute the colour value of the current component
    int col_cc = 0;
    int channels = img1.channels();
    for (int c = 0; c < channels; ++c)
        col_cc += pow(256, c) * img1.at<uchar>(y, (x*channels)+c);

    // Define a stack to keep track of new nodes
    stack<pair<int, int>> coordStack;

    // Add the initial pixel on the stack
    coordStack.push(make_pair(x, y));

    // Do a search while the stack is not empty
    while (!coordStack.empty())
    {
        // Pop a new set of coordinates
        pair<int,int> coord = coordStack.top();
        coordStack.pop();
        x = coord.first;
        y = coord.second;

        // Check if we are in bounds and fave not visited the pixel
        if (x >= 0 && x < img1.cols && y >= 0 && y < img1.rows && mask.at<uchar>(y,x) == 0)
        {
            // Compute the colour in this pixel
            int col_pix1 = 0;
            for (int c = 0; c < channels; ++c)
                col_pix1 += pow(256, c) * img1.at<uchar>(y, (x*channels)+c);

            // Check if it is the same colour
            if (col_pix1 == col_cc)
            {
                // Mark the mask to denote we have visited the pixel
                mask.at<uchar>(y,x) = 1;

                // Compute the colour of the pixel in image 2
                int col_pix2 = 0;
                for (int c = 0; c < channels; ++c)
                    col_pix2 += pow(256, c) * img2.at<uchar>(y, (x*channels)+c);
                
                // Keep track of the intersections in img2
                if (colmap.count(col_pix2) > 0)
                    colmap[col_pix2] += 1;
                else
                    colmap[col_pix2] = 1;

                // Add neighbours to the stack
                coordStack.push(make_pair(x+1, y  ));
                coordStack.push(make_pair(x-1, y  ));
                coordStack.push(make_pair(x  , y+1));
                coordStack.push(make_pair(x  , y-1));
            }
        }
    }
    return colmap;
}

/**
 * @brief Does a breath first search in order to find the connected component. It also keeps track of the encountered colours in img2 in order to find the larges intersecting component.
 * 
 * @param img1 The segmentation or the ground truth
 * @param img2 The ground truth or the segmentation
 * @param mask Matrix denoting the visited pixels
 * @param x x coordinate of the starting position
 * @param y y coordinate of the starting position
 * @param col Colour of the current connected component
 * @param colmap Return parameter, map containing the encountered colours, i.e. CC in img2
 * @return int Size of the connected component
 */
map<int, int> at::cc_bfs_8(Mat& img1, Mat& img2, Mat& mask, int x, int y)
{
    // Initialise the colour map
    map<int, int> colmap;

    // Precompute the colour value of the current component
    int col_cc = 0;
    int channels = img1.channels();
    for (int c = 0; c < channels; ++c)
        col_cc += pow(256, c) * img1.at<uchar>(y, (x*channels)+c);

    // Define a stack to keep track of new nodes
    stack<pair<int, int>> coordStack;

    // Add the initial pixel on the stack
    coordStack.push(make_pair(x, y));

    // Do a search while the stack is not empty
    while (!coordStack.empty())
    {
        // Pop a new set of coordinates
        pair<int,int> coord = coordStack.top();
        coordStack.pop();
        x = coord.first;
        y = coord.second;

        // Check if we are in bounds and fave not visited the pixel
        if (x >= 0 && x < img1.cols && y >= 0 && y < img1.rows && mask.at<uchar>(y,x) == 0)
        {
            // Compute the colour in this pixel
            int col_pix1 = 0;
            for (int c = 0; c < channels; ++c)
                col_pix1 += pow(256, c) * img1.at<uchar>(y, (x*channels)+c);

            // Check if it is the same colour
            if (col_pix1 == col_cc)
            {
                // Mark the mask to denote we have visited the pixel
                mask.at<uchar>(y,x) = 1;

                // Compute the colour of the pixel in image 2
                int col_pix2 = 0;
                for (int c = 0; c < channels; ++c)
                    col_pix2 += pow(256, c) * img2.at<uchar>(y, (x*channels)+c);
                
                // Keep track of the intersections in img2
                if (colmap.count(col_pix2) > 0)
                    colmap[col_pix2] += 1;
                else
                    colmap[col_pix2] = 1;

                // Add neighbours to the stack
                coordStack.push(make_pair(x+1, y  ));
                coordStack.push(make_pair(x-1, y  ));
                coordStack.push(make_pair(x  , y+1));
                coordStack.push(make_pair(x  , y-1));
                coordStack.push(make_pair(x+1, y+1));
                coordStack.push(make_pair(x+1, y-1));
                coordStack.push(make_pair(x-1, y+1));
                coordStack.push(make_pair(x-1, y-1));
            }
        }
    }
    return colmap;
}
