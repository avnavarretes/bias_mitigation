# Bias mitigation

##  Research project
### MSc Data Science with AI
### University of Exeter

Repository with the necessary code to replicate the bias mitigation project for https://huggingface.co/stabilityai/stable-diffusion-2-1-base Stability's Stable Diffusion Model. It includes the results of the images generated with the different experiments.

The list of codes, separated by purpose and project stages.

## utils.py 
Read OpenCV's pre-trained models. Facedetector, in conjunction with the gender_net from caffemodel to determine the gender of the individuals depicted as {\em Male} or a {\em Female}. And DeepFace's analyze function to identify the ethnicity from the following categories: {\em Asian, Indian, White, Black, Middle Eastern, and Latino Hispanic}. Code with the functions to detect age and race from a folder of images.

Adaptated code from: [https://dev.to/ethand91/simple-age-and-gender-detection-using-python-and-opencv-319h]

## SDM_bias_analysis.ipynb
Analysis over the results of SDM generation to measure the level of bias considering the classifications identified and mentioned in the previous paragraph. 

For each image generated, a detection function is used to detect gender and ethnicity. After the generation of num_iterations images for each of the prompts recorded in the prompts_info file, the images generated will be saved in a new folder and a csv file is created with the results of the detection functions (prompts_results_SDM.csv)

After this, the percentage of outcomes for each gender-race group is computed for each prompt and this will be the input of the next generation phase. Each under-represented group for each promop will be used to create a new set of images to be used to re-train a new model. Fow example, if Female-Black turns to have a small participation on the 100 images generated of the prompt "A photo of a CEO of an important company", the code will generate 10 images of "A photo of a black female CEO of an important company", and it would be the same for every case. This new set of images generated will be stored in a new folder

Part of this code was adaptated from Diffusers Colab Notebook: [https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb]

## dreambooth_fine_tuning.ipynb
Code to re-train, fine-tune a SDM model by using dreambooth. 
The input of the code would be a prompt and a set images related to this prompt which es gonna be the images used to fine_tune the model
You have to choose a configuration of hyper parameters to run the fine-tuning.
The parameters and values that were modified for this experiment are:
--lr_scheduler="linear","constant","polynomial" \
--num_class_images=20, 300 \
--max_train_steps=100, 1000

Adapted from: [https://colab.research.google.com/drive/1QUjLK6oUB_F4FsIDYusaHx-Yl7mL-Lae?usp=sharing]

#### post_tuning_gender_race_detection.ipynb
Code to set a folder of folders, the code goes inside each of the folder of images to detect gender and race for every image. Creates a dataframe with the results and save it as a csv file.


Note: The main codes are in ipynb format because they were run on Google Colab as they require GPUs to run. GoogleColab provides the necessary resources to run the models involved in this project.
