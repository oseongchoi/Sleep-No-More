# Sleep-No-More
A mini project of AI that alerts if the user fall in sleep during study in front of laptop.

This type of problem can be solved in many different ways though, I've planned :
 - The model has an end-to-end manner.  
   The only annotation of the given video is whether the user is sleeping or not.
 - The input data is 4-dimensional tensor(width * height * channel * time).  
   Some continuous frames of video will be sent to the model,  
   and the model will predict whether the user in the video is sleeping or studying.
   
FYI, The sleep no more is the name of play in NYC.  
(https://mckittrickhotel.com/sleep-no-more/)  
The best play in my lifetime :)