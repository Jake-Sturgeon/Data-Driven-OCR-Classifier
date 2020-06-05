#OCR assignment report

## Feature Extraction (Max 200 Words)

[When the pages are initially loaded, I add noise to some of the images and then apply noise removal (which uses a gaussian filter to smooth the image then a filter which sets all the pixels below the mean value to 0 and all the value above to 255) to all the images in the test set. This is done to give some knowledge about how noise, and the process of removing it, can change the image data. After this, the images are converted into vectors and PCA is used to reduce the dimensions down to 10. I chose the 2nd to 11th principal components as the 1 component didn't make a difference during classification as the brightness of the images is removed doing noise removal. I also attempted divergence on 20 principal components but it didn't lead to any improvements.]

## Classifier (Max 200 Words)

[Initially, I run the same noise removal function for all the character images. Then turn all the images to feature vectors then apply PCA using the same mean and principal components from the training stage to reduce the data to 10 dimensions.
I then use a weighted KNN (K - Nearest Neighbour) classifier to classify each character from the page data. The current K is the sqrt of the amount of data in each page. A better K would be to work out the regression of each page and increase K accordingly but I never got this to work.]

## Error Correction (Max 200 Words)

[I tried several different methods for error correction but every time it just made more errors.
My first method used a python library called PyEnchant. This was good as it provided a function to check whether or not a word was correct as well as a function which gives a list of suggestions in order of most probable. This meant I could filter the words in length and I also used a distance measure called the "Hamming distance" to give each suggestion further weighting. However, there was no way to stop nouns from changing and even though some words were being correctly changed, the suggestions didn't take into account that some letters look similar. Therefore, if I had more time, I would try to implement some sort of 'simularity' function for each letter to give further weighting to each suggestion.
My second method used an algorithm from http://norvig.com/spell-correct.html which provided a very small/compact spell checker. However, the suggestions provided were sometimes shorter in length than the word given which meant information was being lost. It also required another large text file call big.txt]

## Performance

The percentage errors (to 1 decimal place) for the development data are
as follows:
‐ Page 1: [96.8%]
‐ Page 2: [95.4%]
‐ Page 3: [88.1%]
‐ Page 4: [79.0%]
‐ Page 5: [70.6%]
‐ Page 6: [63.6%]

## Other information (Optional, Max 100 words)

[Optional: highlight any significant aspects of your system that are
NOT covered in the sections above]
