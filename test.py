from model import *

if __name__ == "__main__":
    model_meta_data = []
    layer_type = []
    
    cnn = ModerateCNN()

    for (k, v) in cnn.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    #print(cnn)
    print(model_meta_data)
    print(layer_type)