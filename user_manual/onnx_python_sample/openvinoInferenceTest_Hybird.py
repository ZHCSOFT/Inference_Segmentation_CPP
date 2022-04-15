from openvino.inference_engine import IECore
from glob import glob
import numpy as np
import os
import cv2
import time


def getPredictedResult(exec_net, inputImage):
    result = exec_net.infer(inputs={input_blob: inputImage})
    result = np.asarray(result[output_blob]).reshape([n, 2, h, w])
    result = np.squeeze(result)
    _, result = cv2.threshold(result[1], 1, 255, cv2.THRESH_BINARY)
    result = result.astype('uint8')
    return result


if __name__ == '__main__':
    ie = IECore()
    model = '../unet/script_model/DeeplabV3Plus_resnest.onnx'
    test_path = '../unet/HC_database/test_set'
    save_path = './openvino_result'
    net = ie.read_network(model=model)
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    net.batch_size = 1

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    exec_net = ie.load_network(network=net, device_name='CPU')

    n, c, h, w = [1, 3, 513, 513]
    print(n, c, h, w)

    test_files = glob(test_path+'/*')
    test_files = [x for x in test_files if 'Annotation' not in x]

    start = time.time()

    for test_file in test_files:
        image = cv2.imread(test_file).astype('float32')
        cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)

        
        result = getPredictedResult(exec_net, image)
        cv2.imwrite(save_path+'/'+os.path.basename(test_file), result)

    print('infer finished, avg time is', (time.time() - start) / len(test_files))
