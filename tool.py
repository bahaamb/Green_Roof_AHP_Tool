import numpy as np
import matplotlib.pyplot as plt

def ahp_calculator(factors):
    num_factors = len(factors)

    # Create the pairwise comparison matrix
    pairwise_matrix = np.ones((num_factors, num_factors))
    for i in range(num_factors):
        for j in range(i + 1, num_factors):
            pairwise_matrix[i, j] = float(input(f"Rate the importance of {factors[i]} compared to {factors[j]} (1-9): "))
            pairwise_matrix[j, i] = 1 / pairwise_matrix[i, j]

    # Normalize the pairwise matrix
    column_sums = np.sum(pairwise_matrix, axis=0)
    normalized_matrix = pairwise_matrix / column_sums

    # Calculate the priorities
    priorities = np.mean(normalized_matrix, axis=1)

    return priorities, pairwise_matrix

# Define the factors
factors = [
    'Initial cost',
    'Ease of maintenance',
    'Beginner friendliness',
    'Water saving',
    'Productivity',
    'Heat gain reduction'
]

# Run the AHP calculator to estimate priorities and get pairwise matrix
priorities, pairwise_matrix = ahp_calculator(factors)

# Define the ranks for three comparison types: SB, DWC, and NFT
SB = [89, 50, 100, 0, 74.4, 92]
DWC = [87, 60, 50, 4, 97.6, 100]
NFT = [0, 90, 20, 65, 100, 60]

# Calculate the final index results
final_index_SB = np.dot(SB, priorities)
final_index_DWC = np.dot(DWC, priorities)
final_index_NFT = np.dot(NFT, priorities)

# Determine the system with the highest index
max_index = max(final_index_SB, final_index_DWC, final_index_NFT)
recommendation = ""
if max_index == final_index_SB:
    recommendation = "SB"
elif max_index == final_index_DWC:
    recommendation = "DWC"
else:
    recommendation = "NFT"

# Prepare data for the chart
systems = ['SB', 'DWC', 'NFT']
percentages = [final_index_SB, final_index_DWC, final_index_NFT]

# Plotting the figure
fig, ax = plt.subplots()
bars = ax.bar(systems, percentages)


# Highlighting the most suitable system
highlighted_box = systems.index(recommendation)
bars[highlighted_box].set_color('red')

# Adding percentages labels on the bars
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            systems[i]+"="+ f'{percentages[i]:.2f}%', ha='center', va='bottom')

# Adding AHP weighting results on the right side of the figure
ax.text(1.1, 0.8, 'AHP Weighting Results:', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
for i in range(len(factors)):
    ax.text(1.1, 0.7 - (0.1 * i), f'{factors[i]}: {priorities[i]:.2f}', transform=ax.transAxes, fontsize=10, va='top')

# Set plot title and axis labels
#plt.title('System Evaluation')
plt.xlabel(systems)
plt.ylabel(percentages)

# Add a label indicating the suitable system
suitable_system_label = ax.text(0.5, -0.1, f"Recommended System: {recommendation}", transform=ax.transAxes, fontsize=12, ha='center', color='red')

# Create a table illustrating the pairwise matrix
column_labels = ['Factors'] + factors
cell_text = []
for i in range(len(factors)):
    cell_text.append([factors[i]] + [f"{pairwise_matrix[i, j]:.1f}" for j in range(len(factors))])

table = ax.table(cellText=cell_text, colLabels=column_labels, loc='top', cellLoc='center')

# Set column widths
#table.set_column_widths([0.2] + [0.1] * len(factors))

# Style the table
table.auto_set_font_size(True)
table.set_fontsize(10)
table.scale(2, 1.5)

for i in range(len(factors)):
    table[(i+1, i+1)].set_facecolor('black')
    table[(i+1, i+1)].set_text_props(weight='bold', color='black')

# Hide axis ticks and labels
ax.axis('off')


# Adjust the spacing between the table and the chart
plt.subplots_adjust(top=1)  # Increase the top value as needed



# Display the figure
plt.show()
