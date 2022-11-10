import subprocess
import json
import requests

# Create an empty Baysian optimisation dict
def generateBODictGabor():
    data = {
        "domain_info": {
            "dim": 12,
            "domain_bounds": [
                # { "max" :  2, "min" : 2 },      # p
                { "max" :  5, "min" : 0.5 },    # lambda_o  (Based on odd gabor paper)
                { "max" :  2, "min" : 0.5 },    # sigma_o   
                { "max" :  2, "min" : 0 },      # gamma_o
                # { "max" :  2, "min" : 2 },      # p_o
                { "max" :  5, "min" : 0.1 },    # slope_o
                { "max" : 40, "min" : 0 },      # center_o
                { "max" : 10, "min" : 1 },      # weight_o
                { "max" :  5, "min" : 0.5 },    # lambda_e
                { "max" :  2, "min" : 0.5 },    # sigma_e
                { "max" :  2, "min" : 0 },      # gamma_e
                # { "max" :  2, "min" : 2 },      # p_e
                { "max" :  5, "min" : 0.1 },    # slope_e
                { "max" : 40, "min" : 0 },      # center_e
                { "max" :  1, "min" : 0 },      # weight_e
            ]
        },
        "gp_historical_info": {
            "points_sampled": []
        },
        "num_to_sample": 1
    }
    return data

# Create an empty Baysian optimisation dict
def generateBODictCDL():
    data = {
        "domain_info": {
            "dim": 6,
            "domain_bounds": [
                # { "max" :  2, "min" : 2 },      # p
                # { "max" :  2, "min" : 2 },      # p_edge
                { "max" :  5, "min" : 0.1 },    # slope_edge
                { "max" : 40, "min" : 0 },      # center_edge
                { "max" : 10, "min" : 1 },      # weight_edge
                # { "max" :  2, "min" : 2 },      # p_ridge
                { "max" :  5, "min" : 0.1 },    # slope_ridge
                { "max" : 40, "min" : 0 },      # center_ridge
                { "max" :  1, "min" : 0 },      # weight_ridge
            ]
        },
        "gp_historical_info": {
            "points_sampled": []
        },
        "num_to_sample": 1
    }
    return data

# Create an empty Baysian optimisation dict
def generateBODictDifference():
    data = {
        "domain_info": {
            "dim": 3,
            "domain_bounds": [
                # { "max" :  2, "min" : 2 },      # p_f
                { "max" :  1, "min" : -1 },     # w_f
                # { "max" :  2, "min" : 2 },      # p_b
                { "max" :  1, "min" : -1 },     # w_b
                # { "max" :  2, "min" : 2 },      # p_c
                { "max" :  1, "min" : -1 },     # w_c
            ]
        },
        "gp_historical_info": {
            "points_sampled": []
        },
        "num_to_sample": 1
    }
    return data

# Add new score to the sapled points in the dict
def addSample(dict, par, score):
    dict["gp_historical_info"]["points_sampled"].append({"value_var": 0.01, "value": score, "point": par})
    return dict

# Write the dict to disk (for persistance)
def writeDict(path, dict):
    with open(path, 'w') as convert_file:
        convert_file.write(json.dumps(dict))


# Read the dict from disk (for persistance)
def readDict(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

# Get the next set of parameters to evaluate
def getNewPoints(api_url, dict):
    resp = requests.post(api_url, json.dumps(dict))
    data_resp = resp.json()
    return data_resp["points_to_sample"][0]


