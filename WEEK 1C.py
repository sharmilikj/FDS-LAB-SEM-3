import pandas as pd
structured_data = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})

print("Structured Data:\n", structured_data)

# Unstructured data example
unstructured_data = "This is an example of unstructured data. It can be a piece of text, an image, or a video file."

print("\nUnstructured Data:\n", unstructured_data)

# Semi-structured data example (JSON)
semi_structured_data = {
    'ID': 1,
    'Name': 'Alice',
    'Attributes': {
        'Height': 165,
        'Weight': 68
    }
}

print("\nSemi-structured Data:\n", semi_structured_data)