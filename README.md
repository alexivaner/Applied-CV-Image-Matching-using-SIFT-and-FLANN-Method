# Image Matching using SIFT
 Image Matching using SIFT and FLANN Method
How to run this program:

From command prompt:
You need to specify the input using set_name of inside /img folder,
then it will automatically save all the pyramid image, dog images, keypoint, and image matching inside /result:

If you want to use your own set of image, put image inside a set of folder, for example for matching 2 images:
Put that two images in /img/set_name.

You could run the program by following example:

    python 0860812.py --input bamboo_fox
    python 0860812.py --input mountain
    python 0860812.py --input tree
    python 0860812.py --input own_image

If you want to specify the octave or scaling you can do that by:
python 0860812.py --input bamboo_fox --octave 4 --num_scale 5
The default value of octave will be 4 and num_scale will be 5

The result will be shown in the /result folder separated as:
1) Image Pyramid 
2) Image DoG (Different of Gaussian)
3) Image KeyPoints
4) Image Matching Result

## Reference
SIFT is originally created by David G. Lowe <br>
  [Distinctive Image Feature from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)<br>


This code is combined and modified from this source:<br>
 [rmIslam PythonSIFT](https://github.com/rmislam/PythonSIFT)<br>
 Thank you for your clear code about how to find extrema keypoint.

Good Paper about step by step implementing SIFT:<br>
  [Anatomy of the SIFT Method](https://www.ipol.im/pub/art/2014/82/article.pdf)<br>
 
## Result and Explanation
[embed]https://github.com/alexivaner/Image-Matching-using-SIFT-and-FLANN-Method/blob/main/Report_Ivan%20Surya%20H_0860812.pdf[/embed]


