# Handwriting Recognition

The goal of this project is to be able to read Handwriting.

## Running the project
firstly, install the dependencies in the requirements.txt file (doesn't actually exist at the moment)

next...

## Verbosity Readings
each segmenter class has its own verbosity reading. 
0. No output whatsoever
1. Only outputting textual data
2. Show the output of each segmenter as opencv images
3. Good for demonstraitions. also shows visual representations of the major algorithmic steps.
4. Highest possible output level. Anthing that can be shown will be.

## How the project works
From my research, most ocr algorithms use some form of segmentation and classification.
based off of this, this project breaks the task of reading handwriting into 3 steps.

1. Line segmentation
2. Word / Character segmentation
3. Classification

I have tried many different methods throughout this project, some of which are beneficial in different ways to others,
so I have constructed the segmentation algorithm to use a "builder" type approach, where we can pick and choose what
algorithm we want to be using for each step. This also makes debugging much easier since we can code debug versions for each step, 
allowing consistant results.

I have also chosen to do as much of the work algorithmically rather than relying on some ai algorithm.
This is for two main reasons: fistly, that algorithms tend to be much faster to adapt and understand than complex ai,
and secondly, compiling a dataset for this sort of project sounded hellish and not fun.

### Line Segmentation
Although on the surface this seems like a pretty trivial task - just draw horizontal lines where the text is least dense,
in the case of handwriting, where writing can slope and lines can curve, a more complex algorithm needs to be used.

### Word segmentation
This step takes a line and divides it into sections that can then be run through a classification step.
For now, I think that the best approach to this is to carve the text into connected components. We can then 
analyse these components to find spaces in the text. I think that it would be a mistake to try and segment
these chunks further, as I think that oftentimes the links between characters themselves can contain information about
what the characters are.

### Chunk recognition
once all the segmentation has been completed, we can pass each chunk into some sort of machine learning algorithm
(potentially some 2dimensional varient of a hidden markov model, research ongoing)