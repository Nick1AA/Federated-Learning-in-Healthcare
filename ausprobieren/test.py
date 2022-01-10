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
   for i in range(0, 6):
        if i % 2 == 0:
           text = 'etwas'
        ausgeben(text)