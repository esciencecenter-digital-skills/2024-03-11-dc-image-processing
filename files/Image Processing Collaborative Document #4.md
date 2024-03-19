
![](https://i.imgur.com/iywjz8s.png)


# Image Processing Collaborative Document #4

11-03-2024 to 14-03-2024 Image Processing.

Welcome to The Workshop Collaborative Document #4 14-03-2024.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [tinyurl.com/2024-03-dc-image-processing-4](<https://codimd.carpentries.org/s/QzhGE3HTu#>)

Collaborative Document day 1: [link](<https://tinyurl.com/2024-03-dc-image-processing-1>)

Collaborative Document day 2: [link](<https://tinyurl.com/2024-03-dc-image-processing-2)>)

Collaborative Document day 3: [link](<https://codimd.carpentries.org/s/QV6dtCxX1#>)

Collaborative Document day 4: [tinyurl.com/2024-03-dc-image-processing-4](<https://codimd.carpentries.org/s/QzhGE3HTu>)

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

## 👩‍💻👩‍💼👩🏿🔬👲🏾🔬🧑‍🚀🧙‍♂️🔧 Roll Call
| Name       | Organization     | Topic of Research/work | 
| -------------------- | ------------------------------------ | ---------------------------------------------------------------------------------- | --- | ------- | --------- | ----------------------------- |
| Stripped data      | Stripped data     | Stripped data                                            |     |         |           |                               |



## 💻❄️ :snowflake: 	⛄ :snowman:🔬❄️ :snowflake: ⛄ :snowman::blue_heart:🔧 Ice Breaker
| Name  | something you want to know about image processing |
| ----- | ------------------------------------------------- |
| Stripped data| Stripped data                                                 |
|  |                                                   |

### Answers to extra questions

#### How to apply Gaussian blur on ‘3D’ images (z-stacks)?

You can apply a Gaussian filter to a 3D image (a z-stack) using scikit-image in Python by following these steps:

```python
import imageio.v3 as iio
import matplotlib.pyplot as plt
import skimage as ski

# Step 1: Load your 3D image data
# Assuming you have a 3D image stored in a numpy array, you can load it using imageio.v3.imread.
image = iio.imread('your_image_file_path.tif')

# Step 2: Apply the Gaussian filter to the image
sigma = 1.0
# Apply the Gaussian filter to each slice (2D image) along the z-axis.
filtered_image = np.zeros_like(image)
for z in range(image.shape[0]):
    filtered_image[z] = ski.filters.gaussian(image[z], sigma=sigma)
```

## 💻👩‍💼🔬👨‍🔬👩🏿🔬🚀🔧 Extra questions 1 and 2 (again?)
Name/ Type of Images you work on/ Types of interest

## 💻👩‍💼🔬👨‍🔬👲🏾🔬🚀🔧 Extra questions 3 and 4 (not again?)
Name/ What do you use to work with images now?/ Happiness with it

## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Extra questions 5 and 6 (not again?)
Name/ Pitfalls of your current methods/ Advantages of your methods

## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Extra questions 7 and 8 (not again?)
Name/ Have you used an image generation software/ which one

## 🗓️ Agenda
| Times Day One | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro    |
|          9:20 | Image Basics  |
|          10:10| Working with Skimage | 
|         10:30| Coffee            |
|    10:40 | Drawing  |              
|         11:40 | Bitwise operations        |
|         12:30 |    Summary and Wrap-up     |
|         12: 40| Urgent feedback collection and updates |
|     12:50     | Optional extra tutoring     |
|         13:00 | END                                             |



|  Times Day Two | Topic                        |
| -------:|:----------------------------- |
|  9:00 | Welcome back + Questions  |
|  9:10 | Historgrams    |
| 10:00 | Blurring               |
| 10:15 | Thresholding Images            |
| 11:30 |    Wrap up and summary                   |
| 11:45 | Feedback collection            |
| 12:00 | Extra tutoring and/or networking         |
| 13:00 | End day 2             |




| Times Day Three | Topic                                           |
| -------------:|:-------------------------- |
|          9:00 | Welcome and Intro    |
|          9:25 | Thresholding Images I
| 10:30 | Coffee break                          |
|          10:45 | Thresholding Images II continued          |
| 11:15 |  Connected components analysis                    |
|         13:00 | END                                             |

| Times Day Four | Topic                                           |
| --------------:|:----------------------------------------------- |
|           9:00 | Welcome back                     |
|           9:10 | End CCA + start capstone                       |
|          10:30 | Coffee break                                    |
|          10:45 | Optional Challenge continued            |
|          11:45 | Bonus lecture: transformations, affine |
|          12:45 | Final feedback collection and notices           |
|          13:00 | End course                                      |
 

## 🎓🏢 Evaluation logistics
* At the end of the day you should write evaluations into the colaborative document.


## 🏢 Location logistics
* This version is online! We are using Zoom. If you have trouble with Zoom please contact us at training@esciencecenter.nl

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🫠 🔥Extra materials🔥:

You will probably want to go in your own repo and open the notebook, but it should be about here: https://github.com/esciencecenter-digital-skills/image-processing/blob/main/episodes/extra_materials/advanced_exercises.ipynb


## 🎓🔧Evaluations

 
-Evaluator name (optional)| Evaluation
 -
 - Your alias | Your comments (especially on extra materials)
 -
 -
 -
 -
 -
 -

## 💰🎓🤑 💶Evaluations about the fees 💰💶: 
-Evaluator name (optional)| Evaluation
 -
 
 
-Evaluator name (optional)| Evaluation
 -
 - Your alias | Your comments
 -
 -
 -
 -
 -
 

## 🔧 Exercises

### Exercise 1 of the day:
We might also want to exclude (mask) the small objects when plotting the labeled image (from last exercise).

Enhance the connected_components function such that it automatically removes objects that are below a certain area that is passed to the function as an optional parameter.

#### Your answers:


### Capstone challenge exercise

Write a Python program that uses scikit-image to count the number of bacteria colonies in each image, and for each, produce a new image that highlights the colonies. 

The image files can be found at data/colonies-01.tif, data/colonies-02.tif, and data/colonies-03.tif.

For each image make a new image that highlights colonies as below.
![](https://codimd.carpentries.org/uploads/upload_385f0961c33cb624b7eaa6806c17f8ce.png)


Additionally, print out the number of colonies for each image.

Use what you have learnt about histograms, thresholding and connected component analysis. Try to put your code into a re-usable function, so that it can be applied conveniently to any image file.

**Potential answer:**
```python
def count_colonies(image_filename):
    bacteria_image = iio.imread(image_filename)
    gray_bacteria = ski.color.rgb2gray(bacteria_image)
    blurred_image = ski.filters.gaussian(gray_bacteria, sigma=1.0)
    mask = blurred_image < 0.2
    labeled_image, count = ski.measure.label(mask, return_num=True)
    print(f"There are {count} colonies in {image_filename}")

    colored_label_image = ski.color.label2rgb(labeled_image, bg_label=0)
    summary_image = ski.color.gray2rgb(gray_bacteria)
    summary_image[mask] = colored_label_image[mask]
    fig, ax = plt.subplots()
    ax.imshow(summary_image)
```

then loop over the images
```python
for image_filename in ["data/colonies-01.tif", "data/colonies-02.tif", "data/colonies-03.tif"]:
    count_colonies(image_filename=image_filename)
```

you could also use the `glob.glob` module to loop over the files as we did earlier.


#### Further capstone challenge (optional):

Modify your function from the previous exercise for colony counting to (i) exclude objects smaller than a specified size and (ii) use an automated thresholding approach, e.g. Otsu, to mask the colonies.

```python
def count_colonies_enhanced(image_filename, sigma=1.0, min_colony_size=10, connectivity=2):
    
    bacteria_image = iio.imread(image_filename)
    gray_bacteria = ski.color.rgb2gray(bacteria_image)
    blurred_image = ski.filters.gaussian(gray_bacteria, sigma=sigma)
    
    # create mask excluding the very bright pixels outside the dish
    # we dont want to include these when calculating the automated threshold
    mask = blurred_image < 0.90
    # calculate an automated threshold value within the dish using the Otsu method
    t = ski.filters.threshold_otsu(blurred_image[mask])
    # update mask to select pixels both within the dish and less than t
    mask = np.logical_and(mask, blurred_image < t)
    # remove objects smaller than specified area
    mask = ski.morphology.remove_small_objects(mask, min_size=min_colony_size)
    
    labeled_image, count = ski.measure.label(mask, return_num=True)
    print(f"There are {count} colonies in {image_filename}")
    colored_label_image = ski.color.label2rgb(labeled_image, bg_label=0)
    summary_image = ski.color.gray2rgb(gray_bacteria)
    summary_image[mask] = colored_label_image[mask]
    fig, ax = plt.subplots()
    ax.imshow(summary_image)
```

### Bonus section
The exercises for this section are in the notebook in the repository:
`episodes/10_bonus/image_lecture_bonus_lecture_student.ipynb`



## 🧠 Collaborative Notes

### Continuation of the CCA. 
We saw that some objects were very small, so small we did not see them but the algorithm picked up on them. We will do an exercise related to this. 

There is more than one way to solve our problems. Disucssion of student awnser to exercise. Skimage has a morphology.remove_small_objects function. This was a nice choice- essentially what we want. Adding color can be a possible addition we might want before plotting. 

You can put this before or after you make a labeled image or do both before so you can compare. There is also an approach using enumerate in a for loop. 


### Capstone challenge

This is meant to bring together everything we learned and apply it to something that could be a real world example. 

You can do this as one big exercise in a new notebook. The idea is to analyze three images of bacteria (colonies -1, -2, -3 tiff files). See the "Capstone challenge exercise" section in the Exercises for more details.

### Bonus lecture
See slides in the repository:
`episodes/10_bonus/image_processing_bonus_lecture_v2024.pptx`


## 🔧 Command log

! git pull

## 📚 Resources

Resources will be added from the cheat sheet based on audience composition


## 🧠📚 Final tips and tops

