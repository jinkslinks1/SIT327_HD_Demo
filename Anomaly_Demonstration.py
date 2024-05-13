import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from fim import eclat
#All Imports for the the following anomaly detection Program

 #"""Loads the IoT-23 dataset with specified headers and data types, handling any missing values appropriately."""#
def load_data(filepath):
    
    headers = ['timestamp', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents', 'label', 'detailed-label']
    dtypes = {
        'timestamp': 'float', 'uid': 'string', 'id.orig_h': 'string', 'id.orig_p': 'Int64', 'id.resp_h': 'string', 'id.resp_p': 'Int64', 
        'proto': 'string', 'service': 'string', 'duration': 'float', 'orig_bytes': 'Int64', 'resp_bytes': 'Int64', 
        'conn_state': 'string', 'local_orig': 'boolean', 'local_resp': 'boolean', 'missed_bytes': 'Int64', 'history': 'string', 
        'orig_pkts': 'Int64', 'orig_ip_bytes': 'Int64', 'resp_pkts': 'Int64', 'resp_ip_bytes': 'Int64', 'tunnel_parents': 'string', 
        'label': 'string', 'detailed-label': 'string'
    }
    data = pd.read_csv(filepath, delimiter='\t', header=None, skiprows=8, na_values=['(empty)', '-'], keep_default_na=False, names=headers, dtype=dtypes)
    data[['tunnel_parents', 'label', 'detailed-label']] = data['tunnel_parents'].str.split(expand=True)
    return data

#"""Encodes categorical features using LabelEncoder to transform them into numerical format suitable for modeling."""#
def preprocess_features(df):
    le = LabelEncoder()
    for column in ['uid', 'id.orig_h', 'id.resp_h', 'proto', 'service', 'conn_state', 'history', 'tunnel_parents']:
        print(f"Encoding {column}")
        df[column] = le.fit_transform(df[column].astype(str))
    return df

#"""Trains multiple models and evaluates them, displaying confusion matrices to assess each model's performance."""#
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    #"""Train models and evaluate them using confusion matrices."""#
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    results = {}
    for name, model in models.items():
        pipeline = make_pipeline(SimpleImputer(strategy='mean'), model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[name] = confusion_matrix(y_test, preds)
    return results
#"""Visualizes confusion matrices for each trained model, helping to highlight the accuracy and misclassifications."""#
def plot_confusion_matrices(results):
    #"""Plot confusion matrices for each model."""#
    fig, axes = plt.subplots(nrows=1, ncols=len(results), figsize=(15, 5))
    for ax, (name, matrix) in zip(axes.flat, results.items()):
        sns.heatmap(matrix, annot=True, fmt="d", ax=ax)
        ax.set_title(name)
    plt.show()

#"""Preprocesses the dataset by imputing missing values and encoding categorical variables, preparing it for analysis."""#
def preprocess_data(data):
    #""" Fill missing values for both categorical and numerical data. """#
    # Numerical imputation
    num_imputer = SimpleImputer(strategy='mean')
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

    # Categorical imputation (most frequent strategy or a placeholder like 'missing')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = ['proto', 'service', 'conn_state', 'history', 'tunnel_parents', 'uid', 'id.orig_h', 'id.resp_h']
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols].astype(str))

    # Continue with encoding if necessary
    le = LabelEncoder()
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
    
    data.drop(['local_orig', 'local_resp'], axis=1, inplace=True)

    return data


#"""Trains a set of machine learning models using the preprocessed data, ready for evaluation."""#
def train_models(X_train, y_train):
    #""" Train multiple models with imputation and return them. """#
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier()
    }
    results = {}
    for name, model in models.items():
        pipeline = make_pipeline(SimpleImputer(strategy='mean'), model)
        pipeline.fit(X_train, y_train)
        results[name] = pipeline
    return results

#"""Validates each trained model against the test set, outputting classification reports to compare performance metrics."""#
def validate_models(models, X_test, y_test):
   # """ Validate models and print performance metrics. """#
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
#"""Performs ECLAT algorithm to discover frequent itemsets in the dataset, which can indicate common patterns of botnet activity."""#
def eclat_analysis(data):

    transactions = []
    for index, row in data.iterrows():
        # Create a list of non-null values for each row
        transaction = [str(val) for val in row.values if pd.notnull(val)]
        transactions.append(transaction)
    patterns = eclat(transactions, target='m', supp=2)
    print("Frequent patterns and support:")
    for pattern, support in patterns:
        print(pattern, support)
    return patterns  # Return the patterns for further use

#"""Executes ECLAT analysis and visualizes the frequent patterns to provide insights into potential security threats."""#
def eclat_analysis_and_visualization(data):
    transactions = []
    for index, row in data.iterrows():
        transaction = [str(val) for val in row.values if pd.notnull(val)]
        transactions.append(transaction)
    patterns = eclat(transactions, target='m', supp=2, zmin=2)  # Ensure minimum item size for patterns

    # Visualization
    if patterns:
        plt.figure(figsize=(10, 8))
        plt.title("Frequent Patterns Plot")
        labels = [' '.join(pat[0]) for pat in patterns]
        supports = [pat[1] for pat in patterns]
        sns.barplot(x=supports, y=labels)
        plt.xlabel('Support')
        plt.show()
    else:
        print("No significant patterns found.")

    return patterns

        
#"""Calculates and plots the frequency and importance of different features within the dataset."""#       
def calculate_feature_importance(df):
    table_data = []
    for feature_name in df.columns:
        feature_cnts = df[feature_name].value_counts().to_dict()
        max_count = max(feature_cnts.values(), default=0)
        feature_v = max(feature_cnts, key=feature_cnts.get, default=None)
        table_data.append([feature_name, feature_v, max_count, (max_count / len(df)) * 100])

    # Convert list to DataFrame
    df_features = pd.DataFrame(table_data, columns=['Feature', 'Value', 'Occurrences', 'Freq'])
    
    # Plot the frequency of most frequent values
    df_features.plot(x='Feature', y='Freq', kind='bar', title='Feature Frequency')
    plt.show()

    return df_features

#"""Calculates Gini impurity for features to assess how each contributes to the classification decisions, important for feature selection."""#
def gini_impurity(df_features, df):
    """Calculate Gini impurity for each feature based on how much each feature contributes to class impurity."""
    features_with_impure_score = []
    for index, row in df_features.iterrows():
        feature_name = row['Feature']
        if feature_name in df.columns:
            datas = df[df[feature_name] == row['Value']]
            if not datas.empty:
                p = datas['label'].value_counts(normalize=True)
                gini = 1 - (p ** 2).sum()
                features_with_impure_score.append([feature_name, row['Value'], gini])
    
    # Convert list to DataFrame
    df_gini = pd.DataFrame(features_with_impure_score, columns=['Feature', 'Value', 'Gini'])
    return df_gini
#"""Applies K-means clustering on Gini impurity scores to identify clusters of features with similar properties."""#
def perform_kmeans_clustering(df_gini):
    X = df_gini[['Gini']].to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
    df_gini['Cluster'] = kmeans.labels_
    
    return df_gini

#"""Visualizes the frequent patterns identified by the ECLAT algorithm, highlighting common itemsets in network traffic."""#
def visualize_frequent_patterns(patterns):
    """Visualize frequent patterns using a horizontal bar chart."""
    # Assuming 'patterns' is a list of tuples like (pattern, support)
    labels = [' '.join(pat[0]) for pat in patterns]
    supports = [pat[1] for pat in patterns]

    plt.figure(figsize=(10, 8))
    plt.barh(labels, supports, color='skyblue')
    plt.xlabel('Support')
    plt.ylabel('Patterns')
    plt.title('Support of Frequent Network Traffic Patterns')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest supports on top
    plt.show()
    
#"""Trains a RandomForest model to extract feature importance, visualizing it to understand which features most influence model decisions."""#
def calculate_and_plot_feature_importance(X_train, y_train):
    # Train a RandomForest model to extract feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    features = X_train.columns

    # Creating a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plotting feature importances
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importance Plot")
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.show()

    return feature_importance_df


def main():
    # Load data
    filepath = 'ul-feature-selection-for-botnet-detection/data/conn.log.labeled'
    data = load_data(filepath)
    print("Initial Data Loaded:")
    print(data.head())

    # Preprocess the data
    data = preprocess_data(data)
    print("Data after preprocessing (checking for missing values):")
    print(data.isnull().sum())

    # Display basic data info after preprocessing
    print("Data types after preprocessing:")
    print(data.dtypes)
    

    # Prepare data for model training and feature importance analysis
    X = data.drop(['label', 'detailed-label'], axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # Print key features from the dataset
    keys = ['id.orig_p', 'history', 'id.orig_h', 'label', 'orig_pkts', 'proto', 'label', 'id.resp_p', 'detailed-label', 'conn_state', 'id.resp_h']
    print("\nKey features from the data:")
    print(data[keys].head())
    
      # Summarize the feature set for the audience
    table_data = []
    for feature_name in data.columns:
        feature_cnts = data[feature_name].value_counts().to_dict()
        max_count = max(feature_cnts.values(), default=0)
        feature_v = max(feature_cnts, key=feature_cnts.get, default=None)
        table_data.append([feature_name, feature_v, max_count, (max_count / len(data)) * 100])
        # Plotting feature summary table
    fig, ax = plt.subplots()
    print("Feature Summary Table")
    table = ax.table(cellText=table_data, colLabels=['Feature', 'Value', 'Appearances', 'Freq'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2, 1.5)
    ax.axis('off')
    plt.show()
    
        # ECLAT Pattern Analysis
    print("\nECLAT Pattern Analysis:")
    patterns = eclat_analysis(data)
    
    
    # Gini impurity and clustering of features based on impurity
    df_features = calculate_feature_importance(data)  # Ensure this function returns a DataFrame with 'Feature' and 'Value' columns
    df_gini = gini_impurity(df_features, data)
    df_clustered = perform_kmeans_clustering(df_gini)
    print("Features clustered by Gini Impurity:")
    print(df_clustered[['Feature', 'Cluster']])

    # Plot for Gini impurity scores
    if not df_gini.empty:
        plt.figure(figsize=(10, 5))
        plt.title("Gini Impurity Scores Plot")
        sns.barplot(x='Gini', y='Feature', data=df_gini.sort_values('Gini', ascending=False))
        plt.show()
    else:
        print("No Gini data to plot.")

  

    # Calculate and plot feature importance using RandomForest
    print("\nCalculating and Plotting Feature Importance:")
    feature_importance_df = calculate_and_plot_feature_importance(X_train, y_train)
    print("Feature Importance Results:")
    print(feature_importance_df)

    # ECLAT Pattern Analysis and Visualization
    print("\nECLAT Pattern Analysis and Visualization:")
    patterns = eclat_analysis_and_visualization(data)

    # Model Training and Evaluation
    print("\nModel Training and Evaluation:")
    models = train_models(X_train, y_train)
    validate_models(models, X_test, y_test)





    
    
if __name__ == '__main__':
    main()



