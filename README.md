# Bias mitigation

##  Research project
### MSc Data Science with AI
### University of Exeter

Repository with the necessary code to replicate the bias mitigation project for [https://huggingface.co/stabilityai/stable-diffusion-2-1-base](Stability's Stable Diffusion Model). It includes the results of the images generated with the different experiments.

The list of codes, separated by purpose and project stages.

### utils.py 
Read OpenCV's pre-trained models. Facedetector, in conjunction with the gender_net from caffemodel to determine the gender of the individuals depicted as {\em Male} or a {\em Female}. And DeepFace's analyze function to identify the ethnicity from the following categories: {\em Asian, Indian, White, Black, Middle Eastern, and Latino Hispanic}. Code with the functions to detect age and race from a folder of images.

Adaptated code from: [https://dev.to/ethand91/simple-age-and-gender-detection-using-python-and-opencv-319h]

### SDM_bias_analysis.ipynb
Analysis over the results of SDM generation to measure the level of bias considering the classifications identified and mentioned in the previous paragraph.

Adaptated code from Diffusers Colab Notebook: [https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb]

#### dreambooth.ipynb
Code to re-train, fine-tune a SDM model by using dreambooth.
The parameters that were modified for each experiment are:
--lr_scheduler="linear","constant","polynomial" \
--num_class_images=20, 300 \
--max_train_steps=100, 1000

Adapted from: [https://colab.research.google.com/drive/1QUjLK6oUB_F4FsIDYusaHx-Yl7mL-Lae?usp=sharing]

#### gender_race_detection.ipynb
Code to set a folder of folders, the code goes inside each of the folder of images to detect gender and race for every image. Creates a dataframe with the results and save it as a csv file.


Note: The main codes are in ipynb format because they were run on Google Colab as they require GPUs to run. GoogleColab provides the necessary resources to run the models involved in this project.
