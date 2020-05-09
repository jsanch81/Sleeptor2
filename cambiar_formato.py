import subprocess
import os

folds = [x for x in os.listdir('data/') if x.startswith('Fold')]
for fold in folds:
    people = [x for x in os.listdir('data/'+fold) if x.isnumeric()]
    for person in people:
        path = 'data/' + fold + '/' + person + '/'
        for fname in os.listdir(path):
            if((fname.endswith('.MOV') or fname.endswith('.mov')) and not os.path.isfile(path + fname[:fname.index('.')] + '.mp4')):
                subprocess.check_call(['ffmpeg', '-i', path+fname, '-vcodec', 'copy', '-acodec', 'copy', path+fname.replace('MOV', 'mp4').replace('mov', 'mp4')])