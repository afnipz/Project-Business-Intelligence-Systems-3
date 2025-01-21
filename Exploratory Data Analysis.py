df_updated = pd.read_csv('update_heart_failure.csv')
df_updated.head()  # Menampilkan beberapa baris pertama dari data yang telah diupdate

df_updated.columns

# Menampilkan dimensi DataFrame (jumlah baris dan kolom)
df.shape

# Korelasi antar fitur dengan target (Exited)
correlation_matrix = df_updated.corr()

# Heatmap untuk melihat korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasi antar Fitur')
plt.show()

# Fokus pada korelasi dengan kolom target ''DEATH_EVENT''
correlation_with_target = correlation_matrix['DEATH_EVENT'].sort_values(ascending=False)

print("Korelasi dengan target (DEATH_EVENT):\n", correlation_with_target)

# Mengatur palet warna
sns.set_palette("pastel")

# Pisahkan fitur numerik
numerical_features = df_updated.select_dtypes(include=np.number).columns.tolist()

# Memplot histogram
fig, axes = plt.subplots(7, 2, figsize=(12, 24))  # Sesuaikan jumlah subplot dengan jumlah fitur numerik
axes = axes.flatten()

# Iterasi dan Plot Histogram
for i, feature in enumerate(numerical_features):
    if i < len(axes):  # Pastikan index tidak melebihi batas axes
        hist_plot = sns.histplot(df_updated[feature], bins=15, kde=False, ax=axes[i], color=sns.color_palette("pastel")[i % len(sns.color_palette("pastel"))])
        axes[i].set_title(feature, fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        # Menambahkan keterangan angka di atas setiap bar
        for patch in hist_plot.patches:
            height = patch.get_height()
            if height > 0:  # Jika ada nilai
                axes[i].text(
                    patch.get_x() + patch.get_width() / 2,
                    height + 0.5,  # Posisi sedikit di atas bar
                    int(height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black"
                )
    else:
        break

# Menyusun tata letak dan menampilkan judul utama
plt.tight_layout()
plt.suptitle('Histogram', fontsize=16, y=1.02)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Get all columns except DEATH_EVENT
columns_to_plot = [col for col in df_updated.columns if col != 'DEATH_EVENT']

# Calculate number of rows needed (3 charts per row)
rows = (len(columns_to_plot) + 2) // 3

# Create subplots with 3 columns
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
axes = axes.flatten()

# Define colors to match the example
colors = ['#8de0d0', '#fff68f']  # [mint green, pale yellow]

for i, column in enumerate(columns_to_plot):
    # Create bins for numerical data
    if df_updated[column].dtype in ['float64', 'int64']:
        bins = np.linspace(df_updated[column].min(), df_updated[column].max(), 6)
        df_updated[f'{column}_binned'] = pd.cut(df_updated[column], bins=bins)
        column_data = f'{column}_binned'
    else:
        column_data = column

    # Calculate distributions
    death_dist = df_updated[df_updated['DEATH_EVENT'] == 1][column_data].value_counts()
    no_death_dist = df_updated[df_updated['DEATH_EVENT'] == 0][column_data].value_counts()

    # Combine distributions
    combined_dist = pd.DataFrame({
        'Died': death_dist,
        'Survived': no_death_dist
    }).fillna(0)

    # Calculate totals
    total_died = combined_dist['Died'].sum()
    total_survived = combined_dist['Survived'].sum()

    # Create the pie chart
    wedges, _, autotexts = axes[i].pie([total_died, total_survived],
                                     labels=['', ''],
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90)

    # Create legend
    legend_labels = [f'Died', f'Survived']

    axes[i].legend(wedges, legend_labels,
                  title=column,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

    axes[i].set_title(f'Distribution of {column}', fontsize=10, pad=20)

# Remove any extra subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distribution of Each Attribute by Death Event', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Clean up temporary binned columns
for column in columns_to_plot:
    if df_updated[column].dtype in ['float64', 'int64']:
        df_updated.drop(f'{column}_binned', axis=1, inplace=True, errors='ignore')

