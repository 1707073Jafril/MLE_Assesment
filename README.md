# MLE_Assesment

**File Description :**
  - AssesmentFinal.ipynb : It is a clear and concise notebook and data processing , modeling , testing , evaluation are shown here
  - CV(1).pdf : It is the unseen pdf for testing the model
  - ResumeDataset.csv : is the dataset
  - preprocess_model.py : is the raw code of the previous notebook
  - script.py : is the script to predict on unseen pdf and save the result
  - finalized_model.sav : is the saved trained model and it can be used directly
  - category.csv  : is the output file containg file and predicted category
  

**About the model**\
Finally we chose a traditional machine learning and SVM based model as our final model . We chose **LinearSVC()** because :
  - It can handle high dimensional data
  - Faster than Neural Network based model
  - Obtained maxmum accuracy & F1 Score
  - Work good in non linear decision boundaries
\
\
**F1 Score of LinearSVC()**
**0.99**
\
\
**Model Overfit?**
- No, the model passed the manual test also!

\
**Run the Script**
- Modify the pdf location
- Run : python script.py in linux machine or windows
- It will show the model description and will create a csv file containing unseen csv file and predicted category!

\
**Or Run the Notebook in Colab , Kaggle or in your machine**
