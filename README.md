# 🌜 Sleep Disorder Detection App

![Sleep Disorder Detection](https://github.com/RobinMillford/Sleep-Disorder-Detection-App/blob/main/Prediction1.png)
![Sleep Disorder Detection1](https://github.com/RobinMillford/Sleep-Disorder-Detection-App/blob/main/Analysis.png)

## Overview

Welcome to the Sleep Disorder Detection App! This app uses a machine learning model to detect potential sleep disorders based on user input features such as age, gender, occupation, sleep duration, and more. It also provides helpful tips and insights based on the predictions.

### Features
- Predicts potential sleep disorders: No Disorder, Sleep Apnea, Insomnia
- Provides personalized sleep tips based on predictions
- Offers insights and analytics from the sleep data
- User-friendly interface with interactive elements and visualizations

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/RobinMillford/sleep-disorder-detection-app.git
    cd Sleep-Disorder-Detection-App
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App Locally

1. **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Open your browser and navigate to:**
    ```
    http://localhost:8501
    ```

### Usage

1. **Navigate to the Prediction page:**
   - Enter your details such as age, gender, occupation, sleep duration, etc.
   - Click the "🔮 Predict" button to see the results and tips based on the prediction.

2. **Explore the Analytics page:**
   - View various visual insights and analytics derived from the sleep data.

## Deployment

The app is deployed on Streamlit Cloud and can be accessed [here](https://sleep-disorder-detection-app.streamlit.app/).

## Contributing

We welcome contributions to improve the app! Here’s how you can get started:

1. **Fork the repository:**
    Click on the 'Fork' button on the upper right corner of the repository page.

2. **Clone your forked repository:**
    ```bash
    git clone https://github.com/RobinMillford/sleep-disorder-detection-app.git
    cd Sleep-Disorder-Detection-App
    ```

3. **Create a new branch for your feature or bug fix:**
    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make your changes and commit them:**
    ```bash
    git add .
    git commit -m "Description of your changes"
    ```

5. **Push your changes to your forked repository:**
    ```bash
    git push origin feature/your-feature-name
    ```

6. **Create a Pull Request:**
    - Go to the original repository.
    - Click on the 'Pull Requests' tab and then on the 'New Pull Request' button.
    - Select your feature branch and submit the pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://www.streamlit.io) - The framework used to build the web app.
- [Unsplash](https://unsplash.com) - For the background image used in the app.
- [Scikit-learn](https://scikit-learn.org/stable/) - For the machine learning model.

Enjoy detecting sleep disorders and making data-driven improvements to your sleep habits! 🌙💤
