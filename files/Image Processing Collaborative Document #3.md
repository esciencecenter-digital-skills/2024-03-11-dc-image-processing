![](https://i.imgur.com/iywjz8s.png)


# Image Processing Collaborative Document #3

11-03-2024 to 14-03-2024 Image Processing.

Welcome to The Workshop Collaborative Document #3 13-03-2024.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [tinyurl.com/2024-03-dc-image-processing-3](<https://codimd.carpentries.org/s/QV6dtCxX1#>)

Collaborative Document day 1: [link](<https://tinyurl.com/2024-03-dc-image-processing-1>)

Collaborative Document day 2: [link](<https://tinyurl.com/2024-03-dc-image-processing-2>)

Collaborative Document day 3: [tinyurl.com/2024-03-dc-image-processing-3](<https://codimd.carpentries.org/s/QV6dtCxX1#>)

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

Thijs Vroegh, 

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
|Name/ Organization||
| ---------------- |:-------------------------------- |
| Stripped data | Stripped data |



## 💻❄️ :snowflake: 	⛄ :snowman:🔬❄️ :snowflake: ⛄ :snowman::blue_heart:🔧 Ice Breaker |    

|Name| Artist who impressed you at 10 / why, changes in opinion |
|-----|:------------|
|Stripped data| Stripped data



## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Extra questions 3 and 4 (not again?)
Name/ What do you use to work with images now?/ Happiness with it

## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Extra questions 5 and 6 (not again?)
Name/ Pitfalls of your current methods/ Advantages of your methods

## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Extra questions 7 and 8 
Name/ Have you used an image generation software/ which one

## 🗓️ Agenda
| Times Day One | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro     |
|          9:20 | Image Basics |
|          10:10| Working with Skimage | 
|         10:30| Coffee            |
|    10:40 | Drawing     |              
|         11:40 | Bitwise operations     |
|         12:30 |    Summary and Wrap-up    |
|         12: 40| Urgent feedback collection and updates  |
|     12:50     | Optional extra tutoring  |
|         13:00 | END                                             |



|  Times Day Two | Topic                        |
| -------:|:----------------------------- |
|  9:00 | Welcome back + Questions  |
|  9:10 | Historgrams   |
| 10:00 | Blurring              |
| 10:15 | Thresholding Images          |
| 11:30 |    Wrap up and summary                   |
| 11:45 | Feedback collection            |
| 12:00 | Extra tutoring and/or networking         |
| 13:00 | End day 2             |




| Times Day Three | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro  |
|          9:25 | Thresholding Images I           |
| 10:30 | Coffee break                          |
|          10:45 | Thresholding Images II continued         |
| 11:15 |  Connected components analysis                    |
|         13:00 | END                                             |

|  Times Day Four | Topic                        |
| -------:|:----------------------------- |
|  9:00 | Welcome back  |
| 9:30 | Optional Challenge                         |
| 10:30 | Coffee break                          |
| 10:45 | Optional Challenge continued                        |
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

We would especially love extra materials comments - what do you think?
 
- Evaluator name (optional) | Evaluation

- If the pre-course survey would work next time, it might be nice to edit the subjects based on peoples skill level. 
-
-
-
-
-
-
-
 

## 🔧 Exercises
Exercises:

### Thresholding
#### Exercise 1: practice with simple thresholding (15 min)

Now, it is your turn to practice. Suppose we want to use simple thresholding to select only the coloured shapes (in this particular case we consider grayish to be a colour, too) from the image `data/shapes-02.jpg`.

First, plot the grayscale histogram as in the Creating Histogram episode and examine the distribution of grayscale values in the image. What do you think would be a good value for the threshold t?


**Your awnser to "what is a good threshold t?":** 
<Stripped data>



#### Exercise 1, continued (10 min)

Next, create a mask to turn the pixels above the threshold `t` on and pixels below the threshold `t` off. Note that unlike the image with a white background we used above, here the peak for the background colour is at a lower gray level than the shapes. Therefore, change the comparison operator less `<` to greater `>` to create the appropriate mask. Then apply the mask to the image and view the thresholded image. If everything works as it should, your output should show only the coloured shapes on a black background.



#### Exercise 2: ignoring more of the images – brainstorming (10 min)

Let us take a closer look at the binary masks produced by the `measure_root_mass` function. You may have noticed in the section on automatic thresholding that the thresholded image does include regions of the image aside of the plant root: the numbered labels and the white circles in each image are preserved during the thresholding, because their grayscale values are above the threshold. Therefore, our calculated root mass ratios include the white pixels of the label and white circle that are not part of the plant root. Those extra pixels affect how accurate the root mass calculation is!

How might we remove the labels and circles before calculating the ratio, so that our results are more accurate? Think about some options given what we have learned so far.

**Answers:**
<Stripped data>
    * Apply a threshold 


**Discussion**
An option is indeed to use the drawing examples and put a rectangle over the labels. However it will be hard to find the right coordinates, and there might not even be a single set of coordinates that works for all images. (This could have been prevented by the one taking the images, by having a clear separation and a constant position for the labels relative to the roots).

Another option is to use an extra threshold (close to 1, white) to filter out the labels.

For example, all of the following measures could have made the images easier to process, by helping us predict and/or detect where the label is in the image and subsequently mask it from further processing:

    Using labels with a consistent size and shape
    Placing all the labels in the same position, relative to the sample
    Using a non-white label, with non-black writing
    


#### Exercise 3: thresholding a bacteria colony image (15 min)

In the images directory `data/`, you will find an image named `colonies-01.tif`.

This is one of the images you will be working with in the morphometric challenge at the end of the workshop.

1. Plot and inspect the grayscale histogram of the image to determine a good threshold value for the image.
2. Create a binary mask that leaves the pixels in the bacteria colonies “on” while turning the rest of the pixels in the image “off”.

**Potential answer:**
    <strippd data>

### Connected component analysis

#### Exercise 1: How many objects are in that image

1. Using the `connected_components` function, find two ways of outputting the number of objects found.
2. Does this number correspond with your expectation? Why/why not?
3. Play around with the `sigma` and `thresh` parameters.
    a. How do these parameters influence the number of objects found?
    b. OPTIONAL: Can you find a set of parameters that will give you the expected number of objects?

Put your green sign online when done.

#### Solution:

<stripped data>

#### Exercise 2: Plot a histogram of the object area distribution

Plot a histogram of the object area distribution.

Similar to how we determined a “good” threshold in the _Thresholding_ episode, it is often helpful to inspect the histogram of an object property. For example, we want to look at the distribution of the object areas.

Create and examine a histogram of the object areas obtained with `ski.measure.regionprops`.
What does the histogram tell you about the objects?

#### Solution:

The histogram can be plotted with
<stripped data>

![](https://codimd.carpentries.org/uploads/upload_dcb7b7671efb2b44514fcae5f8383e0e.png)



#### Exercise 3: Filter objects by area

Now we would like to use a minimum area criterion to obtain a more accurate count of the objects in the image.

Find a way to calculate the number of objects by only counting objects above a certain area.

#### Complete solution (stripped data (removed))

## 🧠 Collaborative Notes

### 7. Thresholding


```python
# import necessary libraries
import glob

import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

%matplotlib widget
```

```python
# load the image
shapes01 = iio.imread(uri="data/shapes-01.jpg")

fig, ax = plt.subplots()
ax.imshow(shapes01)

```
We will filter out the background from the original image by automatically creating a mask that will only contain the shapes that we see in the image.

```python
# convert the image to grayscale
gray_shapes = ski.color.rgb2gray(shapes01)

# blur the image to denoise
blurred_shapes = ski.filters.gaussian(gray_shapes, sigma=1.0)

fig, ax = plt.subplots()
ax.imshow(blurred_shapes, cmap="gray")
```

We will create a histogram of the blurred, gray image so that we can determine the threshold value. 

The blurring helps to get rid of tiny noise that will otherwise interfer with the actual objects that you want to pick up. 


```python
# create a histogram of the blurred grayscale image
histogram, bin_edges = np.histogram(blurred_shapes, bins=256, range=(0.0, 1.0))

fig, ax = plt.subplots()
ax.plot(bin_edges[0:-1], histogram)
ax.set_title("Grayscale Histogram")
ax.set_xlabel("grayscale value")
ax.set_ylabel("pixels")
ax.set_xlim(0, 1.0)
```
It should look like this:
![](https://datacarpentry.org/image-processing/fig/shapes-01-histogram.png)

You can see a very high peak towards 1.0, that is the background, which is the majority of the pixels in the images. The other, smaller peaks are the different shapes in the image. 

Now we want to cut off all the pixels that are the background and keep the pixels that are the shapes. So we select the threshold to be just next to the last small peak at `0.8`.

Comparing the numpy array `binary_mask` with the integer threshold, will 

```python
# create a mask based on the threshold
t = 0.8
binary_mask = blurred_shapes < t

fig, ax = plt.subplots()
ax.imshow(binary_mask, cmap="gray")

```

```python
# We apply the mask to a copy of the original image
selection = shapes01.copy()
selection[~binary_mask] = 0

fig, ax = plt.subplots()
ax.imshow(selection)
```

**Now we did exercise 1, see above**

If you need to mask out thousands of images, it is not very practical to plot and look at every histogram for every image. scikit image has nice methods implemented for automated thresholding.


```python
maize_roots = iio.imread(uri="data/maize-root-cluster.jpg")

fig, ax = plt.subplots()
ax.imshow(maize_roots)
```
let's plot the histogram just for illustration

```python
# convert the image to grayscale
gray_image = ski.color.rgb2gray(maize_roots)

# blur the image to denoise
blurred_image = ski.filters.gaussian(gray_image, sigma=1.0)

# show the histogram of the blurred image
histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
fig, ax = plt.subplots()
ax.plot(bin_edges[0:-1], histogram)
ax.set_title("Graylevel histogram")
ax.set_xlabel("gray value")
ax.set_ylabel("pixel count")
ax.set_xlim(0, 1.0)
```

The peaks in this histogram are not as well separated as in the previous toy example. We will use Otsu's method which will automatically detect two major peaks and select a threshold in between.


```python
# performin automatic thresholding
t = ski.filters.threshold_otsu(blurred_image)
print(f"Found automatic threshold t = {t}")
```
Because the backgroun is black, we want to remove everything that is below the threshold.

```python
# create a binary mask with the threshold found by otsu's method
binary_mask = blurred_image > t

fig, ax = plt.subplots()
ax.imshow(binary_mask, cmap="gray")
```

```python
# apply the binary mask to select the foreground
selection = maize_roots.copy()
selection[~binary_mask] = 0

fig, ax = plt.subplots()
ax.imshow(selection)
```

#### Application: measuring root mass
In the data folder of our repository we have various images of roots and we would like to have an estimate of the root mass of each of these images.

For this we will implement a function that we can apply multiple times, so that we don't have to retype the same code for every image, or change the image path in the code. 

We will count the number of pixels in the image that make up the roots and devide it by the total amount of pixels to get an estimate of the root mass.

```python
def measure_root_mass(filepath, sigma=1.0):
    """
    You could put a docstring here that will tell the user how to use this function.
    
    filepath: str, path to the image file
    sigma: ...
    """
    # read the original image, converting to grayscale on the fly
    image = iio.imread(uri=filepath, mode="L")
    
    # blur before thresholding
    blurred_image = ski.filters.gaussian(image, sigma=sigma)
    
    # perform automatic thresholding to produce a binary image
    t = ski.filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > t
    
    # determine root mass ratio
    root_pixels = np.count_nonzero(binary_mask)
    w = binary_mask.shape[1]
    h = binary_mask.shape[0]
    density = root_pixels / (w * h)
    
    return density 
```

Now we can use this function and apply it to one of our images:
```python
measure_root_mass(filepath="data/trial-016.jpg", sigma=1.5)
```

or apply it to all files by first finding all files that start with "trial_" . For this we use the glob library and a so called *regular expression* syntax.

```python
all_files = glob.glob("data/trial-*.jpg")
all_files
```

Now we can loop through these files and apply our function:

```python
for filepath in all_files:
    density = measure_root_mass(filepath=filepath, sigma=1.5)
    # output in format suitable for .csv
    print(filepath, density, sep=",")
```

We are now taking into account the labels as well , in **exercise 2** we discuss how you could get rid of the labels.

We will now change our original function to also filter out the labels:

```python
def enhanced_root_mass(filepath, sigma=1.0):
    # read the original image, converting to grayscale on the fly
    image = iio.imread(uri=filepath, mode="L")
    
    # blur before thresholding
    blurred_image = ski.filters.gaussian(image, sigma=sigma)
    
    #####
    # perform binary thresholding to mask the white label and circle
    binary_mask = blurred_image < 0.95
    
    #perform automatic thresholding using only the pixels with value True in the binary mask
    t = ski.filters.threshold_otsu(blurred_image(binary_mask))
    
    #update binary mask to identify pixels which are both less than 0.95 and greater than t
    binary_mask = (blurred_image < 0.95) & (blurred_image > t)
    
    #####
    
    # determine root mass ratio
    root_pixels = np.count_nonzero(binary_mask)
    w = binary_mask.shape[1]
    h = binary_mask.shape[0]
    density = root_pixels / (w * h)
    
    return density
```

Now we can again apply the function:

```python
all_files = glob.glob("data/trial-*.jpg")
for filename in all_files:
    density = enhanced_root_mass(filename=filename, sigma=1.5)
    # output in format suitable for .csv
    print(filename, density, sep=",")
```


*Question:* the white circle is probably meant to measure the mass. How could you use that to determine the root mass? You could think of detecting the circle size by counting the number of pixels and dividing the root mass by that. But again here we have the problem that the circles are not very consistently placed. There are ways to detect certain shapes, but we will not go into that here. It would have been better if the person that took the images would have placed the circle consistently in the lowerleft corner for example. 

Now we did **exercise 3** to apply the things we have learned so far in a slightly harder type of picture, the bacterial colony in a petridish image that we have seen earlier.

### 8. Connected Component Analysis (CCA)

#### Objects

In the _Thresholding_ episode we have covered dividing an image into foreground and background pixels. In the shapes example image, we considered the coloured shapes as foreground _objects_ on a white background.

![](https://codimd.carpentries.org/uploads/upload_41b456262e991fefae2098e9c37dbdbe.png)

By looking at the mask image, one can count the objects that are present in the image. But how did we actually do that, how did we decide which lump of pixels constitutes a single object?

#### Pixel Neighborhoods

In order to decide which pixels belong to the same object, one can exploit their neighborhood: pixels that are directly next to each other and belong to the foreground class can be considered to belong to the same object.

Let’s discuss the concept of pixel neighborhoods in more detail. Consider the following mask “image” with 8 rows, and 8 columns. For the purpose of illustration, the digit `0` is used to represent background pixels, and the letter `X` is used to represent object pixels foreground).

```
0 0 0 0 0 0 0 0
0 X X 0 0 0 0 0
0 X X 0 0 0 0 0
0 0 0 X X X 0 0
0 0 0 X X X X 0
0 0 0 0 0 0 0 0
```

The pixels are organised in a rectangular grid. In order to understand pixel neighborhoods we will introduce the concept of “jumps” between pixels. The jumps follow two rules: First rule is that one jump is only allowed along the column, or the row. Diagonal jumps are not allowed. So, from a centre pixel, denoted with o, only the pixels indicated with a 1 are reachable:

```
- 1 -
1 o 1
- 1 -
```

The pixels on the diagonal (from `o`) are not reachable with a single jump, which is denoted by the `-`. The pixels reachable with a single jump form the **1-jump** neighborhood.

The second rule states that in a sequence of jumps, one may only jump in row and column direction once -> they have to be _orthogonal_. An example of a sequence of orthogonal jumps is shown below. Starting from `o` the first jump goes along the row to the right. The second jump then goes along the column direction up. After this, the sequence cannot be continued as a jump has already been made in both row and column direction.

All pixels reachable with one, or two jumps form the **2-jump** neighborhood. The grid below illustrates the pixels reachable from the centre pixel **o** with a single jump, highlighted with a **1**, and the pixels reachable with 2 jumps with a **2**.

```
2 1 2
1 o 1
2 1 2
```

We want to revisit our example image mask from above and apply the two different neighborhood rules. With a single jump connectivity for each pixel, we get two resulting objects, highlighted in the image with `A`’s and `B`’s.

```
0 0 0 0 0 0 0 0
0 A A 0 0 0 0 0
0 A A 0 0 0 0 0
0 0 0 B B B 0 0
0 0 0 B B B B 0
0 0 0 0 0 0 0 0
```

In the 1-jump version, only pixels that have direct neighbors along rows or columns are considered connected. Diagonal connections are not included in the 1-jump neighborhood. With two jumps, however, we only get a single object `A` because pixels are also considered connected along the diagonals.

```
0 0 0 0 0 0 0 0
0 A A 0 0 0 0 0
0 A A 0 0 0 0 0
0 0 0 A A A 0 0
0 0 0 A A A A 0
0 0 0 0 0 0 0 0
```

We have just introduced how you can reach different neighboring pixels by performing one or more orthogonal jumps. We have used the terms 1-jump and 2-jump neighborhood. There is also a different way of referring to these neighborhoods: the 4- and 8-neighborhood. With a single jump you can reach four pixels from a given starting pixel. Hence, the 1-jump neighborhood corresponds to the 4-neighborhood. When two orthogonal jumps are allowed, eight pixels can be reached, so the 2-jump neighborhood corresponds to the 8-neighborhood.

#### Connected Component Analysis

In order to find the objects in an image, we want to employ an operation that is called Connected Component Analysis (CCA). This operation takes a binary image as an input. Usually, the `False` value in this image is associated with background pixels, and the `True` value indicates foreground, or object pixels. Such an image can be produced, e.g., with thresholding. Given a thresholded image, the connected component analysis produces a new _labeled_ image with integer pixel values. Pixels with the same value, belong to the same object. scikit-image provides connected component analysis in the function `ski.measure.label()`. Let us add this function to the already familiar steps of thresholding an image.

```python
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

%matplotlib widget
```

In this episode, we will use the `ski.measure.label` function to perform the CCA.

Next, we define a reusable Python function `connected_components`:

```python
def connected_components(filename, sigma=1.0, t=0.5, connectivity=2):
    # load the image
    image = iio.imread(filename)
    # convert the image to grayscale
    gray_image = ski.color.rgb2gray(image)
    # denoise the image with a Gaussian filter
    blurred_image = ski.filters.gaussian(gray_image, sigma=sigma)
    # mask the image according to threshold
    binary_mask = blurred_image < t
    # perform connected component analysis
    labeled_image, count = ski.measure.label(binary_mask,
                                                 connectivity=connectivity, return_num=True)
    return labeled_image, count
```

The first four lines of code are familiar from the Thresholding episode.

Then we call the `ski.measure.label` function. This function has one positional argument where we pass the `binary_mask`, i.e., the binary image to work on. With the optional argument connectivity, we specify the neighborhood in units of orthogonal jumps. For example, by setting `connectivity=2` we will consider the 2-jump neighborhood introduced above. The function returns a `labeled_image` where each pixel has a unique value corresponding to the object it belongs to. In addition, we pass the optional parameter `return_num=True` to return the maximum label index as `count`.

We can call the above function `connected_components` and display the labeled image like so:

```python
labeled_image, count = connected_components(filename="data/shapes-01.jpg", sigma=2.0, t=0.9, connectivity=2)

fig, ax = plt.subplots()
ax.imshow(labeled_image)
ax.set_axis_off();
```

Let's convert `labeled_image` into an actual RGB image:

```python
# convert the label image to color image
colored_label_image = ski.color.label2rgb(labeled_image, bg_label=0)

fig, ax = plt.subplots()
ax.imshow(colored_label_image)
ax.set_axis_off();
```

You might wonder why the connected component analysis with `sigma=2.0`, and `threshold=0.9` finds 11 objects, whereas we would expect only 7 objects. Where are the four additional objects? With a bit of detective work, we can spot some small objects in the image, for example, near the left border.

![](https://codimd.carpentries.org/uploads/upload_f3939fa56c588bfa008fea801550783d.png)

For us it is clear that these small spots are artifacts and not objects we are interested in. But how can we tell the computer? One way to calibrate the algorithm is to adjust the parameters for blurring (`sigma`) and thresholding (`t`), but you may have noticed during the above exercise that it is quite hard to find a combination that produces the right output number. In some cases, background noise gets picked up as an object. And with other parameters, some of the foreground objects get broken up or disappear completely. Therefore, we need other criteria to describe desired properties of the objects that are found.

#### Morphometrics - Describe object features with numbers

Morphometrics is concerned with the quantitative analysis of objects and considers properties such as size and shape. For the example of the images with the shapes, our intuition tells us that the objects should be of a certain size or area. So we could use a minimum area as a criterion for when an object should be detected. To apply such a criterion, we need a way to calculate the area of objects found by connected components.

The scikit-image library provides the function `ski.measure.regionprops` to measure the properties of labeled regions. It returns a list of `RegionProperties` that describe each connected region in the images. The properties can be accessed using the attributes of the `RegionProperties` data type. Here we will use the properties `"area"` and `"label"`. You can explore the scikit-image documentation to learn about other properties available.

We can get a list of areas of the labeled objects as follows:

```python
# compute object features and extract object areas
object_features = ski.measure.regionprops(labeled_image)
object_areas = [objf["area"] for objf in object_features]
object_areas
```

This will produce the output

```
[318542, 1, 523204, 496613, 517331, 143, 256215, 1, 68, 338784, 265755]
```

## 🔧 Command log

## 📚 Resources

Resources will be added from the cheat sheet based on audience composition


## 🧠📚 Final tips and tops