#encoding:utf-8
#@Time : 2017/6/5 9:07
#@Author : JackNiu

import numpy as np
import pandas as pd

aa=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(type(aa))
bb=pd.DataFrame(aa)
print(bb)