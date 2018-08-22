'''
Use the VERY SAME methods (as far as possible) to test on actual known heatmaps
Compute by hand what the answer *should* be
'''
import math
import numpy as np
import json
import cv2
import copy

class AccuracyTester():

    def __init__(self, joint_accur=None, joints=None, output=None, nStack=None, gtMaps=None, batchSize=None):
        self.joint_accur = joint_accur
        self.joints = joints
        self.output = output
        self.nStack = nStack
        self.gtMaps = gtMaps
        self.batchSize = batchSize

    def _accuracy_computation(self):
        """ Computes accuracy tensor
        """
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(
                self._accur(self.output[:, self.nStack - 1, :, :, i], self.gtMaps[:, self.nStack - 1, :, :, i],
                            self.batchSize))
        return self.joint_accur

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = float(0) ###  err = tf.to_float(0)
        for i in range(num_image):
            err = err + self._compute_err(pred[i], gtMap[i]) ###  err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return float(1) - (err / num_image) ###  tf.subtract(tf.to_float(1), err / num_image)


    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)

        return (math.sqrt((float(u_x - v_x)**2) + (float(u_y - v_y)**2))) / float(64)  # changed to image size
    ###  tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
    ###                     tf.to_float(56))  # changed to image size


    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of max position
        """
        resh = tensor.reshape([-1]) ###  resh = tf.reshape(tensor, [-1])
        argmax = np.argmax(resh, 0)  # (Changed function from arg_max) ### argmax = tf.argmax(resh, 0)
        return argmax % tensor.shape[0], argmax // tensor.shape[0]
    ###  argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0]


if __name__ == '__main__':
    target = []
    with open('gtMaps99.txt', 'r') as f:
        target = json.load(f)
    target = np.array(target)
    #print("Target:")
    #print(target)
    output = []
    with open('output99.txt', 'r') as f:
        output = json.load(f)
    output = np.array(output)
    #print("Output:")
    #print(output)
    nStack = 4
    batchSize = 4
    digits = [0, 1, 2, 3]
    tester = AccuracyTester(joints=digits,output=output,gtMaps=target,batchSize=batchSize,nStack=nStack)
    #print(output.shape)
    #print(target.shape)
    # for each image (there should be 4), compute the actual error
    for image in range(batchSize):
        print("\nImage: ", str(image))
        for digit in range(len(digits)):
            print("Digit: ", str(digit))
            max_target = np.amax(target[image, nStack - 1, :, :, digit])
            argmax_target = np.argmax(target[image, nStack - 1, :, :, digit])
            print("amax target:", str(max_target), "argmax target:", str(argmax_target),
                  "x,y:", str(argmax_target % 64), str(argmax_target // 64))
            max_output = np.amax(output[image, nStack - 1, :, :, digit])
            argmax_output = np.argmax(output[image, nStack - 1, :, :, digit])
            print("amax output:", str(max_output), "argmax output:", str(argmax_output),
                  "x,y:", str(argmax_output % 64), str(argmax_output // 64))
            temp = copy.deepcopy(target[image, nStack - 1, :, :, digit] * (255 / max_target))
            cv2.imwrite('testing/e99b_target'+ str(image) + '_' + str(digit) + '.jpg', temp)
            temp = copy.deepcopy(output[image, nStack - 1, :, :, digit] * (255 / max_output))
            cv2.imwrite('testing/e99b_output'+ str(image) + '_' + str(digit) + '.jpg', temp)
            error = tester._compute_err(target[image, nStack - 1, :, :, digit], output[image, nStack - 1, :, :, digit])
            print("Error:", str(error))

    accuracy = tester._accuracy_computation()
    print("\nOverall Accuracy:")
    print(accuracy)