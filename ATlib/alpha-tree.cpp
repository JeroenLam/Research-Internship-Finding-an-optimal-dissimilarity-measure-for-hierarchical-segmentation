#include "alpha-tree.h"

using namespace std;
using namespace cv;

// #define DEBUG

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//                         Alpha-Tree nodes
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/**
 * @brief Create a Alpha-Tree-node object with alpha = aprent = 0
 * 
 */
at::AlphaTreeNode::AlphaTreeNode()
{
    d_alpha = 0.0;
    d_parent = 0;
}

/**
 * @brief Create a Alpha-Tree-node object
 * 
 * @param alpha Alpha value
 */
at::AlphaTreeNode::AlphaTreeNode(double alpha)
{
    d_alpha = alpha;
    d_parent = 0;
}

/**
 * @brief Returns the alpha value of the node
 * 
 * @return The alpha value
 */
double at::AlphaTreeNode::getAlpha()
{
    return d_alpha;
}

/**
 * @brief Sets the alpha value for the node
 * 
 * @param alpha The alpha value
 */
void at::AlphaTreeNode::setAlpha(double alpha)
{
    d_alpha = alpha;
}

/**
 * @brief Returns the parent value of the node
 * 
 * @return The parent value
 */
int at::AlphaTreeNode::getParent()
{
    return d_parent;
}

/**
 * @brief Sets the parent value for the node
 * 
 * @param alpha The parent value
 */
void at::AlphaTreeNode::setParent(int parent)
{
    d_parent = parent;
}

/**
 * @brief Returns the output pixel colour value of the node
 * 
 * @return The alpha value
 */
Vec3b at::AlphaTreeNode::getOutval()
{
    return d_outVal;
}

/**
 * @brief Sets the output pixel colour value for the node
 * 
 * @param bgrVector an OpenCV vector containing BGR values
 */
void at::AlphaTreeNode::setOutval(Vec3b bgrVector)
{
    d_outVal = bgrVector;
}

/**
 * @brief Sets the output pixel colour value for the node
 * 
 * @param b The blue component
 * @param g The green component
 * @param r The red component
 */
void at::AlphaTreeNode::setOutval(uchar b, uchar g, uchar r)
{
    d_outVal[0] = b;
    d_outVal[1] = g;
    d_outVal[2] = r;
}

/**
 * @brief Sets the output pixel colour value for the node to a random 8-bit BGR value
 */
void at::AlphaTreeNode::setRandOutval()
{
    randu(d_outVal, Scalar(0,0,0), Scalar(255,255,255));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//                           Alpha-Tree
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// ========================================
//              Constructors
// ========================================

/**
 * @brief Used to create an empty object. Will crash if called with member functions
 */
at::AlphaTree::AlphaTree()
{}

/**
 * @brief Create a Alpha-Tree object from the alpha values for 4 connectivity
 * 
 * @param alpha_h alpha values in the horizontal direction (type: CV_64FC1)
 * @param alpha_v alpha values in the verticel direction   (type: CV_64FC1)
 * @param lambdamin alpha signal threshold (default 0)
 */
at::AlphaTree::AlphaTree(Mat alpha_h, Mat alpha_v, double lambdamin)
{
    DEBUG_PRINT(("AlphaTree: 2 alpha arrays Constructor\n"));

    // Set the connectivity
    d_connectivity = 4;

    // Check that the provided alpha matrices are the same size and set size parameters
    assert(alpha_h.cols == alpha_v.cols);
    d_width = alpha_h.cols + 1;
    assert(alpha_h.rows == alpha_v.rows);
    d_height = alpha_h.rows + 1;
    int imgsize = d_width * d_height;
    EdgeQueue* queue = at::EdgeQueueCreate((d_connectivity / 2) * imgsize);

    // Initialise the roots
    int* root = new int[imgsize * 2];

    // Initialise the nodes
    d_curSize = imgsize;
    d_maxSize = 2 * d_curSize;
    d_node.assign(d_maxSize, 0.0);

    // Phase 1 combines nodes that are not seen as edges and fills the edge queue with found edges
    this->Phase1(queue, root, alpha_h, alpha_v, lambdamin);

    // Phase 2 runs over all edges, creates SalienceNodes and 
    this->Phase2(queue, root);

    at::EdgeQueueDelete(queue);
    delete[] root;
}

// ========================================
//            Public functions
// ========================================

/**
 * @brief Prints the nodes in the alpha tree with the corresponding parent and alpha value
 */
void at::AlphaTree::printNodes()
{
    DEBUG_PRINT(("AlphaTree:printNodes:\n"));
    cout << "Printing Alpha-tree nodes: " << d_curSize << '/' << d_node.size()  << "\n";
    for (int idx = 0; idx < d_curSize; ++idx)
            cout << "  [" << idx << "]parent,alpha : " << d_node[idx].getParent() << ", " << d_node[idx].getAlpha() << '\n';
}

/**
 * @brief Computes the horizontal cut of the alpha tree based on the provided lambda treshold
 * 
 * @param lambda The threshold to use when cutting the alpha-tree
 * @return A filtered version of the input image based on the alpha threshold chosen 
 */
Mat at::AlphaTree::SalienceFilter(double lambda)
{
    DEBUG_PRINT(("AlphaTree:SalienceFilter:\n"));
    // Compute the image size
    int imgsize = d_width * d_height;
    
    // Check if lambda is larger then the root
    if (lambda <= d_node[d_curSize-1].getAlpha())
    {
        // Set the output value of last nodes (Random colour)
        d_node[d_curSize-1].setRandOutval();

        // Set colours of the other nodes
        for (int idx = d_curSize - 2; idx >= 0; --idx)
        {
            // check if we are dealing with the level root and if it has the right salience
            if (this->IsLevelRoot(idx) && (d_node[idx].getAlpha() >= lambda))
            {
                // set the color of the level root (Random colour)
                d_node[idx].setRandOutval();
            }
            else
            {
                // use parents color
                d_node[idx].setOutval(d_node[d_node[idx].getParent()].getOutval());
            }
        }
    }
    else 
    {
        // Set all values to zero (i.e. black)
        for_each(d_node.begin(), d_node.end(),
            [](at::AlphaTreeNode node)
            {
                node.setOutval(0, 0, 0);
            });
    }
    
    // Allocate and set colors of the out image
    Mat outImg(d_height, d_width, CV_8UC3);
    for (int idx = 0; idx < imgsize; ++idx)
    {
        int x = idx % d_width;
        int y = idx / d_width;

        outImg.at<Vec3b>(y, x) = d_node[idx].getOutval();
    }

    return outImg;
}

/**
 * @brief Return the number of nodes present in the tree
 * 
 * @return Number of nodes in the tree
 */
int at::AlphaTree::size()
{
    return d_curSize;
}

/**
 * @brief Return the size of the original image
 * 
 * @return The size of the original image
 */
int at::AlphaTree::imgSize()
{
    return d_width * d_height;
}

/**
 * @brief Get the alpha of the root node
 * 
 * @return The alpha value
 */
double at::AlphaTree::maxAlpha()
{
    return d_node[d_curSize - 1].getAlpha();
}

/**
 * @brief Get the connectivity of the alpha tree
 * 
 * @return The connectivity
 */
int at::AlphaTree::getConnectivity()
{
    return d_connectivity;
}

/**
 * @brief Computes and returns a matrix containing a histogram of the alpha values of the tree nodes. mat(x,y) such that the size of x is equal to the desired number of bins. y=0 are the amounts and y=1 the upper limit of the bins.
 * 
 * @param bins The number of bins you want to have in the final histogram
 * @param normalised Denotes if you want the final histogram to be normalised
 * @return A matrix containing the histogram values and the bin limits
 */
Mat at::AlphaTree::getHistogram(int bins, bool normalised)
{
    // Create a OpenCV matrix to store the histogram
    Mat hist(bins, 2, CV_64F, 0.0);

    // Find the highest alpha value (root)
    double maxAlpha = d_node[d_curSize-1].getAlpha();
    double binStep = maxAlpha / (bins - 1);

    // Compute the boundaries of the bins
    for (int idx = 0; idx < bins; ++idx)
        hist.at<double>(idx, 1) = idx * binStep;

    // Fill the histogram
    for (int idx = 0; idx < d_node.size(); ++idx)
    {
        // Get the alpha of the node
        double alpha = d_node[idx].getAlpha();

        // Compute the corresponding bin
        if (alpha == 0.0)
            ++hist.at<double>(0,0);
        else
            ++hist.at<double>(ceil(alpha / binStep), 0);
    }

    // Normalise the histogram if desired
    if (normalised)
        for (int idx = 0; idx < bins; ++idx)
            hist.at<double>(idx, 1) /= d_node.size();

    // Return the histogram
    return hist;
}

/**
 * @brief Returns a vector containing all unique alpha values in the tree
 * 
 * @return vector<double> 
 */
vector<double> at::AlphaTree::getAlphas()
{
    // Create a set
    set<double> alphas;

    // Loop over the alpha tree and add the values to the set
    for (int idx = 0; idx < d_curSize; ++idx)
        alphas.insert(d_node[idx].getAlpha());

    vector<double> retVec(alphas.begin(), alphas.end());

    // Return the set
    return retVec;
}


// ========================================
//             Private Functions
// ========================================
/**
 * @brief Create a new node in the Alpha-Tree
 * 
 * @param root Array containing the root positions durring construction
 * @param alpha Alpha value to add to the tree
 */
int at::AlphaTree::NewNode(int* root, double alpha)
{
    DEBUG_PRINT(("AlphaTree:NewNode:\n"));
    // Set the parameters of the last node
    d_node[d_curSize].setAlpha(alpha);
    d_node[d_curSize].setParent(AT_BOTTOM);
    root[d_curSize] = AT_BOTTOM;

    int retIdx = d_curSize++;

    // Return the index of the new element
    return retIdx;
}

/**
 * @brief Find the root of a given node p.
 * 
 * @param root The root array used in the alpha tree construction
 * @param p Node of which we want to find the root
 * @return The index of the root
 */
int at::AlphaTree::FindRoot(int* root, int p)
{
    DEBUG_PRINT(("AlphaTree:FindRoot\n"));
    int r = p;

    while (root[r] != AT_BOTTOM)
        r = root[r];
    int i = p;
    while (i != r)
    {
        int j = root[i];
        root[i] = r;
        i = j;
    }
    return r;
}

/**
 * @brief Find the root of a given node p and add it to the tree
 * 
 * @param root The root array used in the alpha tree construction
 * @param p The node of which we wat to find the root
 * @return The index of the root
 */
int at::AlphaTree::FindRoot1(int* root, int p)
{
    DEBUG_PRINT(("AlphaTree:FindRoot1\n"));
    int r = p;

    // find the root of the tree and set to r
    while (root[r] != AT_BOTTOM)
        r = root[r];
    int i = p;
    /*
    * r = ROOT
    * i = current Pixel
    * invariant: current Pixel != ROOT
    */
    while (i != r)
    {
        int j = root[i];
        // i's root becomes the total root
        root[i] = r;
        // also change the parent in the tree to the total root
        d_node[i].setParent(r);
        // i becomes its own root
        i = j;
    }
    return r;
}

/**
 * @brief Finds the root node of the alpha level of a given node.
 * 
 * @param p Node to find the level root of
 * @return int Index of the level root
 */
int at::AlphaTree::LevelRoot(int p)
{
    DEBUG_PRINT(("LevelRoot:\n"));
    int r = p;

    // Find the root of the node
    while (!(this->IsLevelRoot(r)))
        r = d_node[r].getParent();

    int i = p;

    // Add the node to the branch
    while (i != r)
    {
        int j = d_node[i].getParent();
        d_node[i].setParent(r);
        i = j;
    }
    return r;
}

/**
 * @brief Determine if a node at a given index is the root of its level of the tree.
 * A node is considered the level root if it has the same alpha level as its parent
 * or does not have a parent.
 * 
 * @param i Index of the node
 * @return true if the node is at root level
 * @return false if the node is not at root level
 */
bool at::AlphaTree::IsLevelRoot(int i)
{
    DEBUG_PRINT(("AlphaTree:IsLevelRoot\n"));
    int parent = d_node[i].getParent();

    if (parent == AT_BOTTOM)
        return true;
    return (d_node[i].getAlpha() != d_node[parent].getAlpha());
}

/**
 * @brief Initializes a given node in the AlphaTree so that it can be used
 * in the algorithm.
 * 
 * @param root Root array used in construction of the alpha-tree
 * @param gval Array of pixels in the original image
 * @param p Index of the node in the tree
 */
void at::AlphaTree::MakeSet(int* root, int p)
{   
    DEBUG_PRINT(("MakeSet:\n"));
    d_node[p].setParent(AT_BOTTOM);
    root[p] = AT_BOTTOM;
    d_node[p].setAlpha(0.0);
}

/**
 * @brief 
 * 
 * @param root The root array used in the alpha tree construction
 * @param p 
 * @param q 
 */
void at::AlphaTree::GetAncestors(int* root, int& p, int& q)
{
    DEBUG_PRINT(("AlphaTree:GetAncestors\n"));
    // get root of each pixel and ensure correct order
    p = this->LevelRoot(p);
    q = this->LevelRoot(q);
    if (p < q)
    {
        int temp = p;
        p = q;
        q = temp;
    }

    // while both nodes are not the same and are not the root of the tree
    while ((p != q) && (root[p] != AT_BOTTOM) && (root[q] != AT_BOTTOM))
    {
        q = root[q];
        if (p < q)
        {
            int temp = p;
            p = q;
            q = temp;
        }
    }

    // if either node is the tree root find the root of the other
    if (root[p] == AT_BOTTOM)
        q = FindRoot(root, q);
    else if (root[q] == AT_BOTTOM)
        p = FindRoot(root, p);
}

/**
 * @brief Combines the regions of two pixels.
 * 
 * @param root The root array used in the alpha tree construction
 * @param p First Pixel
 * @param q Second Pixel
 */
void at::AlphaTree::Union(int* root, int p, int q)
{ /* p is always current pixel */
    DEBUG_PRINT(("AlphaTree:Union\n"));

    q = this->FindRoot1(root, q);

    // if q's parent is not p
    if (q != p)
    {
        // set p to be q's parent
        d_node[q].setParent(p);
        root[q] = p;
    }
}


/**
 * @brief Combines the regions of two pixels.
 * 
 * @param root The root array used in the alpha tree construction
 * @param p First Pixel
 * @param q Second Pixel
 */
void at::AlphaTree::Union2(int* root, int p, int q)
{
    DEBUG_PRINT(("AlphaTree:Union2\n"));
    int i;
    d_node[q].setParent(p);
    root[q] = p;
}

/**
 * @brief Create an alpha tree based on the provided alpha values.
 * These alpha values are either combined in the salience tree or
 * they are stored as edges in the edge queue.
 * 
 * @param queue Edge queue to push to
 * @param root Array used in alpha-tree construction
 * @param alpha_h Horizontal alpha values
 * @param alpha_v Vertical alpha values
 * @param lambdamin threshold to determine if we have encountered an edge
 */
void at::AlphaTree::Phase1(at::EdgeQueue* queue, int* root, Mat alpha_h, Mat alpha_v, double lambdamin)
{
    DEBUG_PRINT(("AlphaTree:Phase1: OpenCV (4CC-alpha values\n"));

    // root is a separate case
    this->MakeSet(root, 0);

    // for the first row in the image
    for (int x = 1; x < d_width; ++x)
    {
        // ready current node and find edge strength of the current position
        this->MakeSet(root, x);
        double edgeSalience = alpha_h.at<double>(0, x-1);
        if (edgeSalience < lambdamin)
            // if we evaluate as no edge then we combine the current and last pixel
            this->Union(root, x, x - 1);
        else
            // otherwise we store the found edge
            at::EdgeQueuePush(queue, x, x - 1, edgeSalience);
    }

    // for all other rows
    for (int y = 1; y < d_height; ++y)
    {
        // p is the first pixel in the row
        int p = y * d_width;
        // ready current node and find edge strength of the current position
        this->MakeSet(root, p);
        double edgeSalience = alpha_v.at<double>(y-1, 0);

        if (edgeSalience < lambdamin)
            // if we evaluate as no edge then we combine the current and last pixel
            this->Union(root, p, p - d_width);
        else
            // otherwise we store the found edge
            at::EdgeQueuePush(queue, p, p - d_width, edgeSalience);
        ++p;

        // for each column in the current row
        for (int x = 1; x < d_width; ++x, ++p)
        {
            // reapeat process in y-direction
            this->MakeSet(root, p);
            edgeSalience = alpha_v.at<double>(y-1, x-1);
            if (edgeSalience < lambdamin)
                this->Union(root, p, p - d_width);
            else
                at::EdgeQueuePush(queue, p, p - d_width, edgeSalience);

            // repeat process in x-direction
            edgeSalience = alpha_h.at<double>(y-1, x-1);
            if (edgeSalience < lambdamin)
                this->Union(root, p, p - 1);
            else
                at::EdgeQueuePush(queue, p, p - 1, edgeSalience);
        }
    }
}

/**
 * @brief Create additional nodes in the tree based on the collected edges in phase1 and construct the final tree.
 * 
 * @param queue The queue containing the edges that need to be processed
 * @param root The root array used in the alpha tree construction
 * @return 
 */
void at::AlphaTree::Phase2(at::EdgeQueue* queue, int* root)
{
    DEBUG_PRINT(("AlphaTree:Phase2:\n"));
    double oldalpha = 0;
    while (!IsEmpty(queue))
    {
        // deque the current edge and temporarily store its values
        at::Edge* currentEdge = at::EdgeQueueFront(queue);
        int v1 = currentEdge->p;
        int v2 = currentEdge->q;
        double alpha12 = currentEdge->alpha;
        this->GetAncestors(root, v1, v2);

        at::EdgeQueuePop(queue);
        if (v1 != v2)
        {
            if (v1 < v2)
            {
                int temp = v1;
                v1 = v2;
                v2 = temp;
            }
            if (d_node[v1].getAlpha() < alpha12)
            {
                // if the higher node has a lower alpha level than the edge
                // we combine the two nodes in a new salience node
                int r = this->NewNode(root, alpha12);
                this->Union2(root, r, v1);
                this->Union2(root, r, v2);
            }
            else
            {
                // otherwise we add the lower node to the higher node
                this->Union2(root, v1, v2);
            }
        }
        // store last edge alpha
        oldalpha = alpha12;
    }
}


// ============================================================
//                      Printing histogram
// ============================================================
/**
 * @brief Prints the hirtogram obtained from the AlphaTree.getHistogram() function
 * 
 * @param histogram The matrix containing the histogram data
 */
void at::printHistogram(Mat histogram)
{
    fprintf(stderr, "bin range | amount\n");
    for (int idx = 0; idx < histogram.rows; ++idx)
    {
        double amount = histogram.at<double>(idx,0);
        double bin = histogram.at<double>(idx,1);
        fprintf(stderr, " %8.3f | %d\n", bin, (int)amount);
    }   
}