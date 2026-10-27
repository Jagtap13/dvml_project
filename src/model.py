
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def train_model():
    # Load sample dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build and train the model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200),
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"Model accuracy: {accuracy:.4f}")
