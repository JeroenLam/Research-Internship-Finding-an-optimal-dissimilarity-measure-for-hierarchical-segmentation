# Starting all the scripts
export JL_IMG_NUM=$1

screen -dmS jlGabor$1 bash -c "python3 run_gabor.py; exec bash"
screen -dmS jlCDL$1 bash -c "python3 run_CDL.py; exec bash"
screen -dmS jlWilkinson$1 bash -c "python3 run_wilkinson.py; exec bash"

# screen -ls                : list of existing screens
# screen -r <name>          : reconnect to screen to see output
# screen -S <name> -X quit  : Kill specific screen
# pkill screen              : remove all screens