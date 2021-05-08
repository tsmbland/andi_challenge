import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import *

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=10000, tasks=[1, 2, 3], dimensions=[1, 2], save_dataset=True)
