import os

folds = [x for x in os.listdir('data/') if x.startswith('Fold') and not '.zip' in x]
for fold in folds:
    people = [x for x in os.listdir('data/'+fold) if x.isnumeric()]
    for person in people:
        name_10 = 'data/'+fold+'/'+person+'/10.mp4'
        name_1 = 'data/'+fold+'/'+person+'/1.mp4'
        if(os.path.isfile(name_10) and not os.path.isfile(name_1)):
            os.rename(name_10, name_1)
