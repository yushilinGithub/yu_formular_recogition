import os
import random as rnd

from PIL import Image, ImageFilter, ImageStat
import numpy as np
from pandas import array
from engine import distorsion_generator
import cv2
def getbackgroundImage(foreground, backgroundList):

    backgroundFile = rnd.choice(backgroundList)
    background = Image.open(backgroundFile)
    if foreground.mode != background.mode:
        foreground = foreground.convert(background.mode)
    fwidth,fheight = foreground.size
    bwidth,bheight = background.size
    if bwidth>fwidth and bheight>fheight:
        top = rnd.randint(0,bheight-fheight)
        left = rnd.randint(0,bwidth-fwidth)
        background = background.crop((left, top, left+fwidth, top+fheight))
    else:
        background = background.resize(foreground.size)
    return background

class FakeformularDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls, foreImage, backgroundList,
        skewing_angle,
        random_skew, blur,random_blur, distorsion_type,
        distorsion_orientation, margins,
        image_mode="RGB", 
    ):
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################S
        # get foreground image #
        ##########################
        image = cv2.imread(foreImage)
        mask = 255-image
        mask = np.where(mask>175,255,0)
        mask = Image.fromarray(mask.astype(np.uint8))
        image = Image.fromarray(image)


        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )
        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )


        #############################
        # Generate background image #
        #############################

        background_img =getbackgroundImage(
                distorted_img, backgroundList
            )


        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = distorted_img.size

        background_img.paste(distorted_img, (margin_left, margin_top), distorted_mask.convert("L"))
        #background_img.paste(image, (margin_left, margin_top), mask.convert("L"))
        
        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        final_image = background_img.filter(gaussian_filter)


        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################
        
        final_image = final_image.convert(image_mode)
   

        return final_image
    