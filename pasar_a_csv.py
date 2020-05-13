import numpy as np
import pandas as pd

size = 50
step = 1
n_samples = 3000

# pasar data
data = np.load('data/data_' + str(size) + '_' + str(step) + '_' + str(n_samples) + '.npy', allow_pickle=True)
df = pd.DataFrame(data)#, columns = ['ear', 'mar', 'cir', 'mouth_eye', 'target'])
df.to_csv('data/data_' + str(size) + '_' + str(step) + '_' + str(n_samples) + '.csv', index=False)

# pasar medias
medias = np.load('data/medias_' + str(size) + '_' + str(step) + '_' + str(n_samples) + '.npy', allow_pickle=True)
df = pd.DataFrame(medias)#, columns = ['ear', 'mar', 'cir', 'mouth_eye', 'target'])
df.to_csv('data/medias_' + str(size) + '_' + str(step) + '_' + str(n_samples) + '.csv', index=False)