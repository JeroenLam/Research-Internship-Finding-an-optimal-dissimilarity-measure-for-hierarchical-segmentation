from support import *
from os.path import exists
import os
import time
from dotenv import load_dotenv

load_dotenv()

# .env defined scores
api_url = os.getenv('API_URL')
ex_path = os.getenv('APP_PATH')
threads = int(os.getenv('THREADS'))
img_num = int(os.getenv('JL_IMG_NUM'))
sleep_time = int(os.getenv('SLEEP_TIME'))
dict_path_MOE = os.getenv('MOE_GAB') + '_' + str(img_num) + ".json"
dict_path_raw = os.getenv('RAW_GAB') + '_' + str(img_num) + ".json"

# Define constants for each run
mode     = "Gabor"
img_path = "../img/Campus/zernike180701p" + str(img_num) + ".png"
num_gt   = '3'

def gt_path(gt_num, img_num):
    return "../img/Campus/GT" + str(gt_num) + "p" + str(img_num) + ".png"

args_img = [ex_path, mode, img_path, num_gt, gt_path(1,img_num), gt_path(2,img_num), gt_path(3,img_num)]

# ===================================================
#                   Initialise data
# ===================================================

# Check if the MOE dict exists, if so, load it, otherwise create an new empty dict
if exists(dict_path_MOE):
    data_moe = readDict(dict_path_MOE)
else:
    data_moe = generateBODictGabor()

# Check if the raw data dict exists, if so, load it, otherwise create an new empty dict
if exists(dict_path_raw):
    data_raw = readDict(dict_path_raw)
else:
    data_raw = {"data": []}

# Define a process array to keep track of the running processes with their parameters
processes = []

while (True):
    # Create new processes if neccesary
    if len(processes) < threads:
        while len(processes) < threads:
            print(" ===== Gabor: Created new process =====")
            # Get new parameters
            newPar = getNewPoints(api_url, data_moe)
            # Insert the p values
            moePar = newPar.copy()
            newPar.insert(0,'2.0')
            newPar.insert(4,'2.0')
            newPar.insert(11,'2.0')
            # Spawn new subprocess
            process = subprocess.Popen(args_img + newPar, stdout = subprocess.PIPE)
            # Append to the processes array
            processes.append([process, newPar, moePar])

    # Sleep for the desired time
    time.sleep(sleep_time)

    # For each process
    for idx in range(len(processes)-1, -1, -1):
        # Check if processes are done
        if (processes[idx][0].poll != None):
            print(" ===== Gabor: Process finished =====")
            # If so, process the scores
            scores = str(processes[idx][0].communicate())[3:-10].split(",")
            final_score = float(scores[0])
            parameters_raw = processes[idx][1]
            parameters_moe = processes[idx][2]

            # Add to the dicts as a new sampled point
            data_moe = addSample(data_moe, parameters_moe, final_score)
            data_raw["data"].append(parameters_raw + scores)

            # Store both dicts on disk
            writeDict(dict_path_MOE, data_moe)
            writeDict(dict_path_raw, data_raw)

            # Remove the process from the process array
            processes.pop(idx)
