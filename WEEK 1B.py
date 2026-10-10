import matplotlib.pyplot as plt

# Data
roles = [
    'Data Analyst',
    'Data Engineer',
    'Data Scientist',
    'ML Engineer',
    'Business Analyst'
]

counts = [300, 500, 450, 200, 150]

# Create bar plot
plt.bar(roles, counts)

# Add title and labels
plt.title('Distribution of Data Science Roles')
plt.xlabel('Role')
plt.ylabel('Count')

# Display the graph
plt.show()