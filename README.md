# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.
 # Program : 
 ```
 Name : Sanjay v
 Reg No : 212223230188
```
 ```
import cv2
import numpy as np
import matplotlib.pyplot as plt


faceImage = cv2.imread('ws.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")


faceImage.shape

glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")


glassPNG = cv2.resize(glassPNG,(500,400))
print("image Dimension ={}".format(glassPNG.shape))


glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]


plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');


faceWithGlassesNaive = faceImage.copy()

faceWithGlassesNaive[230:630,250:750]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])


glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

glassMask = np.uint8(glassMask/255)

faceWithGlassesArithmetic = faceImage.copy()

eyeROI= faceWithGlassesArithmetic[230:630,250:750]

maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

maskedGlass = cv2.multiply(glassBGR,glassMask)

eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")


faceWithGlassesArithmetic[230:630,250:750]=eyeRoiFinal

plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");

```

# Output : 

<img width="433" height="450" alt="image" src="https://github.com/user-attachments/assets/4cef0f62-3902-44ab-8eda-08eb2accbd67" />

<img width="487" height="533" alt="image" src="https://github.com/user-attachments/assets/33adcfef-7980-4857-8be9-e25458706834" />

<img width="1345" height="543" alt="image" src="https://github.com/user-attachments/assets/adad4c51-486c-491e-afa5-d05c0028d432" />

<img width="517" height="536" alt="image" src="https://github.com/user-attachments/assets/1e34012b-2859-4b72-b1be-1cfceae05ddf" />

<img width="1760" height="537" alt="image" src="https://github.com/user-attachments/assets/8c5c8ca3-3ac2-4fbe-b32d-fb40508a3980" />

<img width="830" height="398" alt="image" src="https://github.com/user-attachments/assets/b38bb01b-8619-4bc5-b91a-26158deb7851" />




Feel free to fork, contribute, or customize this project for your creative needs!
