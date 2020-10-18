import optparse

import numpy as np

import cv2
import os
import glob
from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR
import sys
from FunctionHelper import *
from pathlib import Path

logger = logging.getLogger(__name__)


# ======================================= #
# TODO: Convolve Gaussian Blur
# ======================================= #
def blur_gaussian(image, sigma):
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blur


# ======================================= #
# TODO: Downsampling without Smoothing
# ======================================= #
def downsample(img, scale):
    small_img = cv2.resize(img,  # original image
                           (0, 0),
                           fx=scale,
                           fy=scale,  # set fx and fy, not the final size
                           interpolation=cv2.INTER_NEAREST)
    return small_img


# ======================================= #
# TODO: Generate Initial Image
# ======================================= #
def generateFirstImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff,
                        sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur


# ======================================= #
# TODO: Generate Sigma
# ======================================= #
def generateGaussianSigma(BaseSigma, num_scale):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    k = 2 ** (1. / (num_scale - 3))
    gaussian_kernels = zeros(
        num_scale)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = BaseSigma

    for image_index in range(1, num_scale):
        sigma_previous = (k ** (image_index - 1)) * BaseSigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


# ======================================= #
# TODO: Make Pyramid Image
# ======================================= #
def make_pyramid(image,
                 image_name,
                 sigma,
                 num_octaves,
                 num_scale):
    '''Read Image using OpenCV'''
    gaussian_image = []

    for a in range(0, num_octaves):
        '''Adjust new sigma according to Scale Level'''
        gaussian_images_in_octave = []

        print("\nSigma octave {}".format(a + 1))

        for b in range(0, num_scale):
            print("Sigma scale {}:{}".format((b + 1), sigma[b]))
            '''Step 1 - Apply Gaussian filter to the image'''
            image = blur_gaussian(image, sigma[b])
            gaussian_images_in_octave.append(image)

            '''Prepare the folder to save the image'''
            folder_image = os.path.basename(image_name)
            if not os.path.exists('result/1_image_pyramid/{}/octave{}'.format(folder_image, a + 1)):
                os.makedirs('result/1_image_pyramid/{}/octave{}'.format(folder_image, a + 1))

            cv2.imwrite("result/1_image_pyramid/{}/octave{}/scaling{}.jpg".format(folder_image, a + 1, b + 1),
                        image)
        gaussian_image.append(gaussian_images_in_octave)
        downsampling_scale = 1 / 2

        image = downsample(gaussian_images_in_octave[0], downsampling_scale)

    print("The pyramid image already generated in 'result/1_image_pyramid/{}/".format(folder_image, b + 1))
    return array(gaussian_image, dtype=object)


# ======================================= #
# TODO: Make Different of Gaussian (DOG)
# ======================================= #
def make_DOG(image_name, gaussian_image, numoctaves=8):
    gaussian_dog = []
    for num_octaves in range(0, numoctaves):
        list_of_image = sorted(
            glob.glob('result/1_image_pyramid/' + image_name + '/octave' + str(num_octaves + 1) + '/*.jpg'))
        print("Generating DOG for {} and with Octave {}".format(image_name, num_octaves + 1))

        gaussian_dog_octaves = []
        '''Do the difference of Gaussian'''
        for a in range(len(list_of_image) - 1):
            im1 = gaussian_image[num_octaves][a]
            im1_ = cv2.imread(list_of_image[a], cv2.IMREAD_GRAYSCALE)
            im2 = gaussian_image[num_octaves][a + 1]
            im2_ = cv2.imread(list_of_image[a + 1], cv2.IMREAD_GRAYSCALE)

            DoGim = subtract(im1, im2)
            DoGim_ = im2_ - im1_

            '''Prepare the folder to save the image'''
            if not os.path.exists('result/2_image_DOG/{}/octave{}'.format(image_name, num_octaves + 1)):
                os.makedirs('result/2_image_DOG/{}/octave{}'.format(image_name, num_octaves + 1))

            '''Write the DOG Result to the folder'''
            cv2.imwrite("result/2_image_DOG/{}/octave{}/DOG{}-{}.jpg".format(image_name, num_octaves + 1, a + 1, a + 2),
                        DoGim_)

            gaussian_dog_octaves.append(DoGim)

            print("DOG for {}, Octave {}, DOGScalling {}-{} Successfully created".format(image_name, num_octaves + 1,
                                                                                         a + 1, a + 2))

        print("See the result in result/2_image_DOG/{}/octave{}/".format(image_name, num_octaves + 1))
        gaussian_dog.append(gaussian_dog_octaves)

    return array(gaussian_dog, dtype=object)


###############################
# Scale-space extrema related
###############################

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width,
                          contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    logger.debug('Finding scale-space extrema...')
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
                zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i - 1:i + 2, j - 1:j + 2], second_image[i - 1:i + 2, j - 1:j + 2],
                                         third_image[i - 1:i + 2, j - 1:j + 2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index,
                                                                              num_intervals, dog_images_in_octave,
                                                                              sigma, contrast_threshold,
                                                                              image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index,
                                                                                           gaussian_images[
                                                                                               octave_index][
                                                                                               localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


# =========== #
#     SIFT    #
# =========== #

def SIFT(image,
         sigma=1.6,
         num_octaves=4,
         num_scale=5,
         assumed_blur=0.5,
         image_border_width=5):
    # ======================================= #
    # TODO: implement your SIFT function here
    # ======================================= #

    image_name = os.path.basename(image)
    image = cv2.imread(image, 0)
    image = image.astype('float32')

    firstImage = generateFirstImage(image, sigma, assumed_blur)
    gaussianKernel = generateGaussianSigma(sigma, num_scale)

    gaussian_images = make_pyramid(firstImage, image_name, gaussianKernel, num_octaves, num_scale)

    dog_images = make_DOG(image_name, gaussian_images, num_octaves)

    num_intervals = num_scale - 3
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)

    return keypoints, descriptors


def draw_keypoint(img, kp):
    img_with_kp = cv2.drawKeypoints(cv2.imread(img), kp, np.array([]), (255, 0, 0))
    image_name = os.path.basename(img)

    '''Prepare the folder to save the image'''
    if not os.path.exists('result/3_image_keypoint/{}'.format(image_name)):
        os.makedirs('result/3_image_keypoint/{}'.format(image_name))

    # Save the image with keypoints and show them on the report
    cv2.imwrite("result/3_image_keypoint/{}/image-with_kp.jpg".format(image_name), img_with_kp)

    print("Keypoint for {}, Successfully created".format(image_name))


# =========== #
#   MATCHING  #
# =========== #
def Matching(img1, img2, kp1, kp2, des1, des2):
    MIN_MATCH_COUNT = 3
    image_set_name = os.path.dirname(img1)
    image_set_name = os.path.basename(image_set_name)

    img1 = cv2.imread(img1, 0)
    img2 = cv2.imread(img2, 0)
    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        '''Prepare the folder to save the image'''
        if not os.path.exists('result/4_image_matching/{}'.format(image_set_name)):
            os.makedirs('result/4_image_matching/{}'.format(image_set_name))

        # Save the image with keypoints and show them on the report
        cv2.imwrite("result/4_image_matching/{}/image-matching.jpg".format(image_set_name), newimg)
        print("Matching for {}, Successfully created".format(image_set_name))

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def Matching_2(img2, img1, kp2, kp1, des2, des1):
    MIN_MATCH_COUNT = 3
    image_set_name = os.path.dirname(img1)
    image_set_name = os.path.basename(image_set_name)

    img1 = cv2.imread(img1, 0)
    img2 = cv2.imread(img2, 0)
    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        '''Prepare the folder to save the image'''
        if not os.path.exists('result/4_image_matching/{}'.format(image_set_name)):
            os.makedirs('result/4_image_matching/{}'.format(image_set_name))

        # Save the image with keypoints and show them on the report
        cv2.imwrite("result/4_image_matching/{}/image-matching.jpg".format(image_set_name), newimg)
        print("Matching for {}, Successfully created".format(image_set_name))

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def main(argv=None):
    if argv is None:
        argv = sys.argv

    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputSet",
                          help="please choose either bamboo_fox, mountain, tree, own_image", default=None)
        parser.add_option("-o", "--octave", action="store", dest="octave_number", help="input octave", default=4)
        parser.add_option("-n", "--num_scale", action="store", dest="scale_num", help="input scale_num", default=5)

        (options, args) = parser.parse_args(argv)

        # validate options
        if options.inputSet is None:
            raise Exception(
                "Must specify input of set image name using -i or --input option. For example --input bamboo_fox")

        print("Hello, this is the SHIFT Program for Applied Computer Vision Course")
        print("Ivan Surya H - 0860812")
        print("Default Octave = 4, default scale=5")

        list_of_image = glob.glob("img/{}/*.jpg".format(options.inputSet))
        if not list_of_image:
            print(
                "No that kind of set_image inside /img folder, please choose either bamboo_fox, mountain, tree, own_image")

        for a in range(len(list_of_image)):
            list_of_image[a] = str(Path(list_of_image[a]))

        img1 = list_of_image[0]
        img2 = list_of_image[1]

        # Compute keypoints and descriptors by SIFT
        kp1, des1 = SIFT(img1, num_octaves=int(options.octave_number), num_scale=int(options.scale_num))
        kp2, des2 = SIFT(img2, num_octaves=int(options.octave_number), num_scale=int(options.scale_num))

        # Show your keypoints on the image
        draw_keypoint(img1, kp1)
        draw_keypoint(img2, kp2)

        '''In case of Value Error, try to swap between destination and source image'''
        try:
            Matching(img1, img2, kp1, kp2, des1, des2)
        except (ValueError, cv2.error):
            Matching_2(img1, img2, kp1, kp2, des1, des2)

        print('All the SHIFT Process Already Completed!')

    except Exception as info:
        raise


if __name__ == '__main__':
    main()
