# SMS Spam Classifier

This is a machine learning project that classifies SMS messages as either spam or ham (not spam). It uses a dataset of SMS messages and applies text preprocessing techniques, followed by training a machine learning model to classify new messages.

## Requirements

Before running the project, ensure that you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `nltk`

You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## Explanation
- **Data Loading:** The dataset is loaded into a pandas DataFrame.
- **Text Preprocessing:** The ham and spam labels are mapped to 0 and 1, respectively.
- **Vectorization:** The text data is converted into numerical form using the CountVectorizer from scikit-learn.
- **Model Training:** A Naive Bayes classifier (MultinomialNB) is used to train the model.
- **Evaluation:** The model is evaluated using accuracy and the classification report.
- **Prediction:** The trained model is used to classify an example SMS message.

  
## How to Use
- Download the spam.csv dataset.
- Place it in the same directory as this script.
- Run the Python script.
- The accuracy of the model and the classification report will be displayed.
- You can use the model to predict whether an SMS message is spam or ham.

  
## 🎨 GUI Interface

The Disease Prediction System features an enhanced Tkinter-based GUI:
- Framed Layout: Symptoms appear inside a scrollable frame.
- Scrollable Symptom List: Users can scroll through symptoms.
- Styled Submit Button: Large, clear Submit button.
- Popup Alert: Displays the predicted disease.

##🏆 Machine Learning Model
✅ **Data Preprocessing**
- Dataset: The model is trained on a structured medical dataset containing diseases and their corresponding symptoms.
- Feature Selection: Symptoms are used as input features, while the disease name is the output label.
- Label Encoding: Disease names are converted to numeric values using LabelEncoder.
✅ **Model Training**

The system utilizes a Random Forest Classifier:

```bash
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

✅ **Cross-Validation**

To ensure robust performance, the model is evaluated using Stratified K-Fold Cross-Validation:

```bash
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Mean Accuracy: {np.mean(scores):.4f}")
```

## ⚙ Future Enhancements

🔹 Additional ML Models: Implement models like SVM, XGBoost for comparison.🔹 Web-Based UI: Convert Tkinter GUI into a Flask or Streamlit Web App.🔹 Larger Dataset: Improve accuracy by adding more symptoms and diseases.🔹 Chatbot Integration: Enable users to interact via a chat-based interface.

## 👨‍💻 Contributing
- Contributions are welcome! Follow these steps:
- Fork the repository
- Create a new branch
- Commit changes
- Push to GitHub
- Create a pull request

## 🛡 License

This project is licensed under the MIT License. You are free to use and modify the code.

## 🤝 Contact

📧 Email: [amaltrivedi3904stella@gmail.com]🔗 LinkedIn: [[Your LinkedIn Profile](https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer/)]📂 GitHub: [[Your GitHub Profile](https://github.com/amalprasadtrivedi)]

## ⭐ Acknowledgments

A big thank you to the Scikit-Learn and Tkinter community for their fantastic libraries! 🎉

