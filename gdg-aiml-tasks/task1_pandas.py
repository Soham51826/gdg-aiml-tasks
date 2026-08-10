import pandas as pd

data = {
    'Student ID': [101, 102, 103, 104, 105, 106],
    'Name': ['Aditi', 'Rahul', 'Priya', 'Arjun', 'Sneha', 'Karan'],
    'Math': [85, 78, None, 92, 75, 89],
    'Science': [90, None, 85, 88, 80, 91],
    'English': [88, 81, 79, None, 76, 90],
    'Attendance (%)': [95, 89, 92, 97, None, 96]
}

df = pd.DataFrame(data)
df.to_csv('students.csv', index=False)

df = pd.read_csv('students.csv')

print(df.head())
print(df.isnull().sum())

cols = ['Math', 'Science', 'English', 'Attendance (%)']
for c in cols:
    df[c] = df[c].fillna(df[c].mean())

print(df[['Math', 'Science', 'English']].mean())
print(df)