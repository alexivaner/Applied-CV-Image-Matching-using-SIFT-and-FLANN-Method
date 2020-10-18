How to run this program:

From command prompt:
You need to specify the input using set_name of inside /img folder,
then it will automatically save all the pyramid image, dog images, keypoint, and image matching inside /result:

    For example:

    python 0860812.py --input bamboo_fox
    python 0860812.py --input mountain
    python 0860812.py --input tree
    python 0860812.py --input own_image


    If you want to specify the octave or scaling you can do that by:
    python 0860812.py --input bamboo_fox --octave 4 --num_scale 5

    The default value of octave will be 4 and num_scale will be 5

The result will be shown in the /result folder like this kind of folder trees:
├───1_image_pyramid
│   ├───1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───0.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───m1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───m0.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───head.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───rotate_L.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───own_image_1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   └───own_image_2.jpg
│       ├───octave1
│       ├───octave2
│       ├───octave3
│       └───octave4
├───3_image_keypoint
│   ├───m1.jpg
│   ├───m0.jpg
│   ├───head.jpg
│   ├───rotate_L.jpg
│   ├───1.jpg
│   ├───0.jpg
│   ├───own_image_1.jpg
│   └───own_image_2.jpg
├───2_image_DOG
│   ├───1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───0.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───m1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───m0.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───head.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───rotate_L.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   ├───own_image_1.jpg
│   │   ├───octave1
│   │   ├───octave2
│   │   ├───octave3
│   │   └───octave4
│   └───own_image_2.jpg
│       ├───octave1
│       ├───octave2
│       ├───octave3
│       └───octave4
└───4_image_matching
    ├───mountain
    ├───bamboo_fox
    ├───tree
    └───own_image
