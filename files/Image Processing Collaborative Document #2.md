![](https://i.imgur.com/iywjz8s.png)


# Image Processing Collaborative Document #2

11-03-2024 to 14-03-2024 Image Processing.

Welcome to The Workshop Collaborative Document #2 12-03-2024.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [tinyurl.com/2024-03-dc-image-processing-2](<https://codimd.carpentries.org/s/VPP2s13Cc#>)

Collaborative Document day 1: [link](<https://codimd.carpentries.org/s/tiVLgPXHk#>)

Collaborative Document day 2: [tinyurl.com/2024-03-dc-image-processing-2](<[url](https://codimd.carpentries.org/s/VPP2s13Cc#)>)

Collaborative Document day 3: [link](<https://codimd.carpentries.org/s/QV6dtCxX1#>)

Collaborative Document day 4: [link](<https://codimd.carpentries.org/s/QzhGE3HTu>)

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, raise your virtual hand in Zoom. A helper will come to assist you as soon as possible, maybe in a private chat, but you can also ask for a breakout room.

## 🖥 Workshop websites

[where you bought tickets](<https://esciencecenter-digital-skills.github.io/2024-03-11-dc-image-processing/>)


[data carpentry website](<https://datacarpentry.org/image-processing/>)

🛠 Setup

[link to setup ](<https://github.com/esciencecenter-digital-skills/image-processing/blob/main/setup.md>)

Download files: Data files can be obtained through the repository
[data](<https://github.com/esciencecenter-digital-skills/image-processing/>)

## 👩‍🏫👩‍💻🎓 Instructors

Giulia Crocioni, Jaro Camphuijsen, Candace Makeda Moore

## 🧑‍🙋 Helpers

Thijs Vroegh 

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ Organization
 
| Name               | Organisation                |
| ------------------ | --------------------------- |
|Stripped Data     | Stripped Data       |
|  |

## 💻❄️ :snowflake: 	⛄ :snowman:🔬❄️ :snowflake: ⛄ :snowman::blue_heart:🔧 Ice Breaker |



## 🗓️ Agenda
| Times Day One | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro     |
|          9:20 | Image Basics  |
|          10:10| Working with Skimage  | 
|         10:30| Coffee            |
|    10:40 | Drawing     |              
|         11:40 | Bitwise operations       |
|    12:30 |  Start histograms and Wrap-up    |
|         12: 40| Urgent feedback collection and updates |
|     12:50     | Optional extra tutoring    |
|         13:00 | END                                             |



| Times Day Two | Topic                            |
| -------------:|:-------------------------------- |
|          9:00 | Welcome back + Questions |
|          9:10 | Drawing            |
|         10:00 |   Bitwise ops |
|         10:50 |   finishing Histograms  |
|         11:20 |  Summary lecture day 1+          |
|         11:35 |  coffee break         |
|         12:00 | Blurring |
|         12:45 |  Wrap up with feedback collection  |
|          12:55     | Urgent issues                    |
|        13:00       |    END day 2                |




| Times Day Three | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro    |
|          9:25 | Thresholding Images I          |
| 10:30 | Coffee break                          |
|          10:45 | Thresholding Images II continued       |
| 11:15 |  Connected components analysis |
|         13:00 | END                                             |

|  Times Day Four | Topic                        |
| -------:|:----------------------------- |
|  9:00 | Welcome back |
| 9:30 | Optional Challenge              |
| 10:30 | Coffee break                          |
| 10:45 | Optional Challenge continued                      |
| 11:30 |  Bonus lecture: transformations, affine |
| 12:30 | Final feedback collection and notices           |
| 13:00 | End course                       |
 

## 🎓🏢 Evaluation logistics
* At the end of the day you should write evaluations into the colaborative document.


## 🏢 Location logistics
* This version is online! We are using Zoom. If you have trouble with Zoom please contact us at training@esciencecenter.nl

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .


## 🎓🔧Evaluations

 
-Evaluator name (optional)| Evaluation
 -
 - Your alias | Your comments
 -
 -
 -
 -
 -
 -
 -
 -
 -
 

## 🔧 Exercises

### Drawing and bitwise operations 

#### Exercise #1:
Experiment!

```python
# create the black canvas
canvas = np.zeros(shape=(600, 800, 3), dtype="uint8")
```
Use the canvas, and add some art, modern art probably :-) using lines, polygons, circles, rectangles, whatever you can find in the `ski.draw` modulew. 

**Your answers:** 
![](https://codimd.carpentries.org/uploads/upload_33d089b42bfe790bc71ca9adfc87480e.png)


**Recommended answers:**
```python
# draw a blue circle with centre (200, 300) in (ry, cx) coordinates, and radius 100
rr, cc = ski.draw.disk(center=(200, 300), radius=100, shape=canvas.shape[0:2])
canvas[rr, cc] = (0, 0, 255)

# draw a green line from (400, 200) to (500, 700) in (ry, cx) coordinates
rr, cc = ski.draw.line(r0=400, c0=200, r1=500, c1=700)
canvas[rr, cc] = (0, 255, 0)

# and so forth and then display
```
#### Exercise #2 
load an image of the wellplate, as below
```python
# Load the image
wellplate = iio.imread(uri="data/wellplate-01.jpg")
wellplate = np.array(wellplate)

# Display the image
fig, ax = plt.subplots()
plt.imshow(wellplate)
```
Now you can use data/centers.txt in the data folder to understand where the centers of the wells are. Each well has a radius of about 16 pixels. Mask to show only the wells. 

**Your answers(if you want to show):**

```f = np.loadtxt("../../data/centers.txt")
mask = np.ones(shape = wellplate.shape[0:2], dtype = 'bool')
for ii in np.arange(f.shape[0]):
    rr, cc = ski.draw.disk(center=(f[ii,1],f[ii,0]) , radius=16, shape=wellplate.shape[0:2])
    mask[rr, cc] = False
wellplate[mask]=0
fig, ax = plt.subplots()
plt.imshow(wellplate)
```


**An example answer:**
```python
# read in original image
wellplate = iio.imread(uri="data/wellplate-01.jpg")
wellplate = np.array(wellplate)
# create the mask image
mask = np.ones(shape=wellplate.shape[0:2], dtype="bool")
# open and iterate through the centers file
with open("data/centers.txt", "r") as center_file:
    for line in center_file:
        # ... getting the coordinates of each well...
        coordinates = line.split()
        cx = int(coordinates[0])
        ry = int(coordinates[1])
        # ... and drawing a circle on the mask
        rr, cc = ski.draw.disk(center=(ry, cx), radius=16, shape=wellplate.shape[0:2])
        mask[rr, cc] = False
# apply the mask
wellplate[mask] = 0
# display the result
fig, ax = plt.subplots()
plt.imshow(wellplate)
    
```

#### Exercise 2 (bonus add-on)

If you spent some time looking at the contents of the data/centers.txt file from the previous challenge, you may have noticed that the centres of each well in the image are very regular. Assuming that the images are scanned in such a way that the wells are always in the same place, and that the image is perfectly oriented (i.e., it does not slant one way or another), we could produce our well plate mask without having to read in the coordinates of the centres of each well. Assume that the centre of the upper left well in the image is at location cx = 91 and ry = 108, and that there are 70 pixels between each centre in the cx dimension and 72 pixels between each centre in the ry dimension. Each well still has a radius of 16 pixels. Write a Python program that produces the same output image as in the previous challenge, but without having to read in the centers.txt file. Hint: use nested for loops.


**Example awnser:**

```python
# read in original image
wellplate = iio.imread(uri="data/wellplate-01.jpg")
wellplate = np.array(wellplate)

# create the mask image
mask = np.ones(shape=wellplate.shape[0:2], dtype="bool")

# upper left well coordinates
cx0 = 91
ry0 = 108

# spaces between wells
deltaCX = 70
deltaRY = 72

cx = cx0
ry = ry0

# iterate each row and column
for row in range(12):
    # reset cx to leftmost well in the row
    cx = cx0
    for col in range(8):

        # ... and drawing a circle on the mask
        rr, cc = ski.draw.disk(center=(ry, cx), radius=16, shape=wellplate.shape[0:2])
        mask[rr, cc] = False
        cx += deltaCX
    # after one complete row, move to next row
    ry += deltaRY

# apply the mask
wellplate[mask] = 0

# display the result
fig, ax = plt.subplots()
plt.imshow(wellplate)
```

### Histograms, part 2 
#### Exercise 2: Colour histogram with a mask (25 min)
We can also apply a mask to the images we apply the colour histogram process to, in the same way we did for grayscale histograms. Consider this image of a well plate, where various chemical sensors have been applied to water and various concentrations of hydrochloric acid and sodium hydroxide:

```python=
# read the image
wellplate = iio.imread(uri="data/wellplate-02.tif")

# display the image
fig, ax = plt.subplots()
plt.imshow(wellplate)
```
![](https://datacarpentry.org/image-processing/fig/wellplate-02.jpg)


Suppose we are interested in the colour histogram of one of the sensors in the well plate image, specifically, the seventh well from the left in the topmost row, which shows Erythrosin B reacting with water.

Hover over the image with your mouse to find the centre of that well and the radius (in pixels) of the well. Then create a circular mask to select only the desired well. Then, use that mask to apply the colour histogram operation to that well.

Your masked image should look like this:
![](https://datacarpentry.org/image-processing/fig/wellplate-02-masked.jpg)

![](https://datacarpentry.org/image-processing/fig/wellplate-02-histogram.png)

**Example answer:**
```python
# create a circular mask to select the 7th well in row 1
mask = np.zeros(shape=wellplate.shape[0:2], dtype="bool")
circle = ski.draw.disk(center=(240,1053), radius=49, shape=wellplate.shape[0:2])
mask[circle] = 1

# mask
masked_img = np.array(wellplate)
masked_img[~mask] = 0

# display
fig, ax = plt.subplots()
plt.imshow(masked_img)

# start dealing with colors
colors = ("red", "green", "blue")

# create the histogram plot, with three lines, 
plt.figure()
plt.xlim([0,256])

for (channel_id, color) in enumerate(colors):
    histogram, bin_edges = np.histogram(
        wellplate[:,:,channel_id][mask], bins=256, range=(0,256)
    )
    plt.plot(histogram, color=color)
    
plt.xlabel("color value")
plt.ylabel("pixel count")

```

### Blurring 

#### Exercise 1: experimenting with sigma values (10 min)

The size and shape of the kernel used to blur an image can have a significant effect on the result of the blurring and any downstream analysis carried out on the blurred image. Try running the code above with a range of smaller and larger sigma values. Generally speaking, what effect does the sigma value have on the blurred image?

**Answers***
*Stripped Data

**Possible answer:**
Generally speaking, the larger the sigma value, the more blurry the result. A larger sigma will tend to get rid of more noise in the image, which will help for other operations we will cover soon, such as thresholding. However, a larger sigma also tends to eliminate some of the detail from the image. So, we must strike a balance with the sigma value used for blur filters.


## 🧠 Collaborative Notes
![](https://)


Lecture 1:  
Reminder on code of conduct. Reminder of schedule. 
### Drawing
Opening conda in command line.

Name the notebooks (so later you don't have a lot of untitled). Selecting markdown from dropdown, then title the notebook.

We need to import libraries to do drawing + bitwise operations: 

```python
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
%matplotlib widget
```

We often only want a specific part of the image. We want to crop. We can do this using slicing. Another option is to create another special image of same size as the original. We call such an image a mask. It will reveal some parts, and others it will 'black out.' To prepare such a mask we need to draw a shape, such as a circle or rectangle. 

```python
# Load and display the maize image
maize_seedlings = iio.imread(uri="data/maize-seedlings.tif")
fig, ax = plt.subplots()
plt.imshow(maize_seedlings)
```
Suppose we want to only select the area with the roots. A boolean can be true or false. We want to create an array with the same size of the image, then assign pixels true or false. Depeneding on whether the pixel is true or false it will display or not. We thus create a mask. 
```python
# create the basic mask
mask = np.ones(shape=maize_seedlings.shape[0:2], dtype="bool")
```
We need to specify the data type as bool otherwise we would get ones. Now we want to draw a box or rectangle, on the mask image. In skimage.draw the start coordinate is upper left, end coorrdinate is lower right.

```python
rr, cc = ski.draw.rectangle(start=(357, 44), end=(740, 720))
mask[rr, cc] = False
# display mask image
fig, ax = plt.subplots()
plt.imshow(mask, cmap="gray")
```
So now we have an image of the same size as our original but we only select a certain part of it. 

We want to modify an image. Let's do it on our image.
```python
# load the original image
maize_seedlings = iio.imread(uri="data/maize-seedlings.tif")
# create the basic mask
mask = np.ones(shape=maize_seedlings.shape[0:2], dtype="bool")
# draw a filled rectangle on the mask image
rr, cc = ski.draw.rectangle(start=(357, 44), end=(740, 720))
mask[rr, cc] = False
# apply the mask (use numpy indexing)
maize_seedlings[mask] = 0
# plot it out
fig, ax = plt.subplots()
plt.imshow(maize_seedlings)
```

Lecture 2:

### Histograms part 2
Went over yesterday's content in the prefilled notebook.
Covered until creating a multicolor histogram.

Now we want to use the image wellplate:

```python=
# read the image
wellplate = iio.imread(uri="data/wellplate-02.tif")

# display the image
fig, ax = plt.subplots()
plt.imshow(wellplate)
```

See Histograms exercise 2 (above) for more explanation.

### Summary lecture day 1+
In the image-processing repository:
`episodes/extra_materials/image_processing_summary_day1_v1.pptx`


Lecture 3: 
### Blurring

For images, see the slides in your cloned repository: `episodes/06_blurring/image_processing_blurring.pptx`

We will cover the topic of filtering images, because blurring is one of the many ways to filter an image.

Filters take information from an image, for example identification of edges. You can detect edges by looking for rapid changes when going from one pixel to the pixel next to it. 

Edge detection uses a so called high-pass filter, it attenuates low frequency compenents in the image, keeping smaller details.

Low pass filter does the opposite, it attenuates high frequency components and keeps larger features. An example is blurring where you smoothen out pixels, using its environment

A filter itself is a matrix, just like the image. In general the filter matrix is smaller than the image it works on. Applying the filter on the image is done using convolution. We call the matrix that is used to filter a "kernel", to convolve the kernel, you let it slide along the signal (in our case, an image) and you multiply the kern values with the signal values and add them up to put the result of that multiplication in one single data point. In that way we get one value per signal data point, in our case that's one value per pixel in the image. This value is the added up multiplication of the kernel values times the image pixel values that the kernel overlaps with. 

When using a kernel larger than one pixel, you will lose a band of pixels around the image and your resulting image will be smaller than the original. To prevent this from happening, you can using padding, which adds a band of zeros (or any other single value) around the original image. If you take this padding to be of the size half of the kernel size, the resulting filtered image will then be the same size. 

We will be using the Gaussian blur filter, which uses a kernel with values that together form a Gaussian curve. 

A collection of Gaussian functions in 1D:
![](https://datacarpentry.org/image-processing/fig/Normal_Distribution_PDF.svg)

In 2D the Gaussian function looks like this:
![](https://datacarpentry.org/image-processing/fig/Gaussian_2D.png)

A Gaussian curve is determined by two parameters, the center point, mean (μ), and the spread or variance (σ²).

An example of a Gaussian kernel matrix:

![](https://datacarpentry.org/image-processing/fig/gaussian-kernel.png)

An example of Gaussian blur in action:

![](https://datacarpentry.org/image-processing/fig/blur-demo.gif)

Now let's do it ourselves in our Jupyter lab environment.

```python
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import skimage as ski

%matplotlib widget
```

```python
image = iio.imread(uri = "data/gaussian-original.png")

#display the image
fig, ax = plt.subplots()
plt.imshow(image)
```

```python
sigma = 3.0

# apply Gaussian blur, creating a new image
blurred = ski.filters.gaussian(image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
```
We define two sigma's because you can have different ones for x and y directions. We use the same for both. We cut off the gaussian curve at a specific value, in theory the Gaussian curve continues infinitely. We specify the colour channel (it is the third dimension of our numpy array) to be the last dimension

```python
# display the image
fig, ax = plt.subplots()
plt.imshow(blurred)
```

We will now inspect, what happens to the histogram when we apply blurring. We will use a petridish image.

```python
# read colonies color image and convert to grayscale
image = iio.imread('data/colonies-01.tif')
image_gray = ski.color.rgb2gray(image)

# define the pixels for which we want to view the intensity profile
xmin, xmax = (0, image_gray.shape[1])
Y = ymin = ymax = 150

# view the image indicating the profile pixels position
fig, ax = plt.subplots()
ax.imshow(image_gray, cmap='gray')
ax.plot([xmin, xmax], [ymin, ymax], color='red')
```

Now we will take a look at the intensity of the pixels:
```python
# select the vector of pixels along "Y"
image_gray_pixels_slice = image_gray[Y, :]

# guarantee that the intensity values are in the [0:255] range (unsigned integers)
image_gray_pixels_slice = ski.img_as_ubyte(image_gray_pixels_slice)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(image_gray_pixels_slice, color='red')
ax.set_ylim(255,0)
ax.set_ylabel('L')
ax.set_xlabel('X')

```
Now we will blur the image and analyse the same line of pixels.

```python
# First, create a blurred version of (grayscale)image
image_blur = ski.filters.gaussian(image_gray, sigma=3)

# like before, plot the pixels profile along "Y"
image_blur_pixels_slice = image_blur[Y, :]
image_blur_pixels_slice = ski.img_as_ubyte(image_blur_pixels_slice)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(image_blur_pixels_slice, 'red')
ax.set_ylim(255, 0)
ax.set_ylabel('L')
ax.set_xlabel('X')
```

```python
fig, ax = plt.subplots()
ax.imshow(image_blur, cmap='gray')
```

#
```python
# 
```

## 🔧 Command log
! git pull
! conda env list
! conda activate image_libraries
! jupyter lab 


## 📚 Resources

Resources will be added from the cheat sheet based on audience composition

- Scikit-image documentation: https://scikit-image.org/docs/dev/user_guide/


## 🧠📚 Final tips and tops
