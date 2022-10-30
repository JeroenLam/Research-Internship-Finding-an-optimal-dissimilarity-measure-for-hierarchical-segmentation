#include "support.h"

using namespace std;

// ============================================================
//                      Support functions
// ============================================================
/**
 * @brief Compute the standard deviation of a vector of double
 * 
 * @param data Vector of doubles of which to compute the standard deviation
 * @return The standard deviation
 */
double stdDev(vector<double> data)
{
    // Compute the mean
    double mean = 0.0;
    for (double num : data)
        mean += num;
    mean /= data.size();

    // Compute the variance
    double variance = 0.0;
    for (double num : data)
        variance += pow(num - mean, 2);
    variance /= data.size();

    return sqrt(variance);
}