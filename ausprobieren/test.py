import pandas as pd
import numpy as np
#from model2 import *
def ausgeben (ausgabe = 'default'):
    print(ausgabe)

if __name__ == "__main__":
    # model_meta_data = []
    # layer_type = []
    
    # cnn = densenet121()

    # for (k, v) in cnn.state_dict().items():
    #     model_meta_data.append(v.shape)
    #     layer_type.append(k)

    # #print(cnn)
    # print(model_meta_data)
    # print(layer_type)
#    d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
#    df = pd.DataFrame(data=d)
#    print(df.iloc[1][1:])
#    help = [df.iloc[1][x] for x in range(1,3)]
#    print(help)
#    print(df.shape[0])
    # a = np.array([[0, 1, 2],
    #           [0, 2, 4],
    #           [0, 3, 6]])
    # print(a)
    # print(a[2][1])
    # b = np.where(a[2]> 2)[0]
    # print(b)
    d = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,Female,68,Frontal,AP,1.0,,,,,,,,,0.0,,,,1.0"
    print(d[26:38])
    e = [d[26:38]]
    print(e)
    e. append("Test")
    print(e)