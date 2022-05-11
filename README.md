# CS4245
CS4245 Seminar Computer Vision by Deep Learning (2021/22 Q4)

TODOs:
- [x] Download dataset & be sure that there is enough data (30k images)
- [ ] Look into GAN architecture (DC and Style GAN)
- [ ] Start Training
- [ ] Setup Kaggle?
- [ ] Setup Blog Post

Possible improvements:
We can use other cartoon items as long as there is enough data


Use kaggle for now instead of GCP (https://www.kaggle.com/code/jonathanborg/cs4245-cv-by-dl)
- This will be done once initial training is complete - for clean up purposes and less development on local machines

Dataset
https://drive.google.com/drive/folders/1bXXeEzARYWsvUwbW3SA0meulCR3nIhDb
Download dataset then move all images into a subfolder (the name of the subfolder can be anything) so the final structure looks like this:
> data
    > faces
        > SUBFOLDER
            > img1.png
            > img2.png 
This is done so the torchvision ImageFolder dataloader properly loads the images
