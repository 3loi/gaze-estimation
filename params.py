
class data:
    IMG_SHAPE = (112,224,3)
    OUT_SHAPE_1 = (3,7)
    OUT_Layers = 6
    max_h = 0.7
    min_h = -0.5
    max_v = 0.2
    min_v = -0.1

class training:
    network_name = 'mobilenet'
    epoch_end = 10
    batch_size = 16

class model:
    hp_en = False
    eye_en = True