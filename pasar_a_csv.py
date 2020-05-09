import numpy as np
import pandas as pd

# pasar data
data = np.load('data/data.npy', allow_pickle=True)
df = pd.DataFrame(data)#, columns = ['ear', 'mar', 'cir', 'mouth_eye', 'target'])
df.to_csv('data/data.csv', index=False)

# pasar medias
medias = np.load('data/medias.npy', allow_pickle=True)
df = pd.DataFrame(medias)#, columns = ['ear', 'mar', 'cir', 'mouth_eye', 'target'])
df.to_csv('data/medias.csv', index=False)