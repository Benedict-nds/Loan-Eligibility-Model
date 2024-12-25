# import pandas as pd
# import numpy as np
# df = pd.read_csv("Backend2\Loandataset.csv")
# df.head()
# df.info()
# # Drop rows where 'term_in_months' is NaN
# df1 = df.dropna(subset=['term_in_months'])

# # Check the new shape of the dataframe
# print(df1.shape)
# df1.columns
# X = df1[['loan_amount', 'rate_of_interest', 'Interest_rate_spread','Upfront_charges', 'term_in_months', 'property_value', 'income','Credit_Score']].values
# y = df1['Status'].values
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE  # Requires 'imblearn' package
# import pandas as pd
# import numpy as np
# import joblib  # For saving the model

# # Assuming X and y are already defined

# # Check class imbalance
# print("Class distribution before balancing:")
# print(pd.Series(y).value_counts())

# # Handle class imbalance (if any)
# smote = SMOTE(random_state=42)
# X, y = smote.fit_resample(X, y)

# print("Class distribution after balancing:")
# print(pd.Series(y).value_counts())

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert back to DataFrame if necessary
# if isinstance(X, pd.DataFrame):
#     X_train = pd.DataFrame(X_train, columns=X.columns)
#     X_test = pd.DataFrame(X_test, columns=X.columns)

# # Check for duplicates between train and test sets before scaling
# if isinstance(X, pd.DataFrame):
#     duplicates = X_train.merge(X_test, how='inner')
#     print("Number of duplicates between train and test:", len(duplicates))
# else:
#     print("Duplicate check skipped as X is not a pandas DataFrame.")

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Define the model
# model = Sequential([
#     Input(shape=(X_train.shape[1],)),
#     Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Add early stopping
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=3,  # Stop after 3 epochs of no improvement
#     restore_best_weights=True
# )

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")

# # Generate predictions
# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# # Classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred_binary))

# # Display predicted vs actual values for the first 5 examples
# print("Predicted vs Actual values (first 5 examples):")
# print(np.hstack((y_pred_binary[:5], y_test[:5].reshape(-1, 1))))

# # Save the model to a .pkl file
# model_filename = "trained_model.pkl"
# joblib.dump(model, model_filename)
# print(f"Model saved as {model_filename}")

# # Load the model back to verify
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully")

# Updated code below
#############################################


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE  # Requires 'imblearn' package
# import pandas as pd
# import numpy as np
# import joblib  # For saving the model

# # Assuming X and y are already defined

# # Check class imbalance
# print("Class distribution before balancing:")
# print(pd.Series(y).value_counts())

# # Handle class imbalance (if any)
# smote = SMOTE(random_state=42)
# X, y = smote.fit_resample(X, y)

# print("Class distribution after balancing:")
# print(pd.Series(y).value_counts())

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert back to DataFrame if necessary
# if isinstance(X, pd.DataFrame):
#     X_train = pd.DataFrame(X_train, columns=X.columns)
#     X_test = pd.DataFrame(X_test, columns=X.columns)

# # Check for duplicates between train and test sets before scaling
# if isinstance(X, pd.DataFrame):
#     duplicates = X_train.merge(X_test, how='inner')
#     print("Number of duplicates between train and test:", len(duplicates))
# else:
#     print("Duplicate check skipped as X is not a pandas DataFrame.")

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Define the model
# model = Sequential([
#     Input(shape=(X_train.shape[1],)),
#     Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Add early stopping
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=3,  # Stop after 3 epochs of no improvement
#     restore_best_weights=True
# )

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")

# # Generate predictions
# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# # Classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred_binary))

# # Display predicted vs actual values for the first 5 examples
# print("Predicted vs Actual values (first 5 examples):")
# print(np.hstack((y_pred_binary[:5], y_test[:5].reshape(-1, 1))))

# from tensorflow.keras.models import load_model

# # Save the model in HDF5 format
# model.save('loan_model.h5')
# print("Model has been saved successfully to 'loan_model.h5'.")

# # Load the model
# model = load_model('loan_model.h5')
# print("Model has been loaded successfully from 'loan_model.h5'.")


##################################################
import pandas as pd
import numpy as np
df = pd.read_csv("Loandataset.csv")
df.head()
df.info()
# Drop rows where 'term_in_months' is NaN
df1 = df.dropna(subset=['term_in_months'])

# Check the new shape of the dataframe
print(df1.shape)
df1.columns
X = df1[['loan_amount', 'rate_of_interest', 'Interest_rate_spread','Upfront_charges', 'term_in_months', 'property_value', 'income','Credit_Score']].values
y = df1['Status'].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Requires 'imblearn' package
import joblib  # For saving the scaler

# Assuming X and y are already defined

# Check class imbalance
print("Class distribution before balancing:")
print(pd.Series(y).value_counts())

# Handle class imbalance (if any)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

print("Class distribution after balancing:")
print(pd.Series(y).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert back to DataFrame if necessary
if isinstance(X, pd.DataFrame):
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

# Check for duplicates between train and test sets before scaling
if isinstance(X, pd.DataFrame):
    duplicates = X_train.merge(X_test, how='inner')
    print("Number of duplicates between train and test:", len(duplicates))
else:
    print("Duplicate check skipped as X is not a pandas DataFrame.")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = 'Instances\scalar.pkl'  # Ensure the directory exists
joblib.dump(scaler, scaler_path)
print(f"Scaler has been saved successfully to '{scaler_path}'.")

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop after 3 epochs of no improvement
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Display predicted vs actual values for the first 5 examples
print("Predicted vs Actual values (first 5 examples):")
print(np.hstack((y_pred_binary[:5], y_test[:5].reshape(-1, 1))))

from tensorflow.keras.models import load_model

# Save the model in HDF5 format
model_path = 'Instances/loan_model.h5'  # Ensure the directory exists
model.save(model_path)
print(f"Model has been saved successfully to '{model_path}'.")

# Load the model
model = load_model(model_path)
print(f"Model has been loaded successfully from '{model_path}'.")
