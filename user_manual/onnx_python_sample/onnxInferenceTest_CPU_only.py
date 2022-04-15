from glob import glob
import onnxruntime as rt
import numpy as np
import time
import cv2
import os

def getPredictedResult(sess, inputImage):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: inputImage})
    result = np.asarray(result).reshape([n, 2, h, w])
    result = np.squeeze(result)
    _, result = cv2.threshold(result[1], 1, 255, cv2.THRESH_BINARY)
    result = result.astype('uint8')
    return result

if __name__ == '__main__':
    model = '../unet/script_model/DeeplabV3Plus_resnest.onnx'
    test_path = '../unet/HC_database/test_set'
    save_path = './onnx_CPU_result'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sess = rt.InferenceSession(model)

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

        
        result = getPredictedResult(sess, image)
        cv2.imwrite(save_path+'/'+os.path.basename(test_file), result)
    print('infer finished, avg time is', (time.time() - start) / len(test_files))
