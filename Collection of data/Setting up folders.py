import numpy as np
import os

data_path = os.path.join('.dataCollection')
actions = np.array(['cSitting', 'cStanding'])
data_number = 30
data_length = 10

for action in actions:
    for sequence in range(data_number):
        try:
            os.makedirs(os.path.join(data_path, action, str(sequence)))
        except:
            pass
