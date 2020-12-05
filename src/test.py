import numpy as np
import torch
from src.main import LogisticRegression

# new = LogisticRegression()

#
the_model = torch.load("BGD_model.pt")
print(the_model.linearTransform.weight[0])
print(type(the_model))



# new.load_state_dict()


# testx = np.load('testx', allow_pickle=True)
# testy = np.load('testy', allow_pickle=True)
# for i in range(47236):
#     print(testx[0, i])
# print(testx.shape)



