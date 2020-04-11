import subprocess
import os

path = 'data/05/'
for fname in os.listdir(path):
    if(fname.endswith('.MOV') or fname.endswith('.mov')):
        subprocess.check_call(['ffmpeg', '-i', path+fname, '-vcodec', 'copy', '-acodec', 'copy', path+fname.replace('MOV', 'mp4').replace('mov', 'mp4')])