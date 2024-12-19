# CS73 Assignment@
## Jemely Robles 

#
#DISCLAIMERS:
#I  was unable to use ginput for color correction as I was getting a matplotlib backend error. I instead chose the coordinates to base my patches on within the code itself. 

#I am using 4 of the late days for this assignment

#HOW TO USE CODE:
# the code is already set up to run the door_stack images. If running my personal images, the exposure started at 1/4000 for al three sets of images, so the part of code in the hdr.py readImagesandExposures func that says 1/2048 would need to be changed to 1/4000. Also, my images are all titled ex(n), so the file path variable in the same function as before, needs to be changed from 'exposure' to 'ex'. For white balancing my images, hdr.py has a white_balance func, it is currently set up for the given images, but code is commented out at the top that would be used for my images. To use, simply uncomment the code between the bar of pound signs, and comment out the rest. In the demo.py file, the directory is set up for the given images, and to change you simply change the folder name. when it comes to exposure numbers, the blue_room images have 16, the my_room images have 14, and the my_room2 images have 13. otherwise, you run the code by simply running demo.py
