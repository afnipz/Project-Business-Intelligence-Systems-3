# Pisahkan fitur dan target dari dataset
# df_updated adalah dataset yang telah diproses sebelumnya
X = df_updated.drop('DEATH_EVENT', axis=1)  # Fitur
y = df_updated['DEATH_EVENT']  # Target

df_updated.shape

# Periksa distribusi awal target
y.value_counts()

# Visualisasi distribusi awal
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette='viridis')
plt.title('Distribusi Target Sebelum SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()

# Terapkan SMOTE untuk menyeimbangkan data latih
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Periksa distribusi setelah SMOTE
y_balanced.value_counts()

# Visualisasi distribusi setelah SMOTE
plt.figure(figsize=(8, 5))
sns.countplot(x=y_balanced, palette='viridis')
plt.title('Distribusi Target Setelah SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()


# Split data dengan ukuran test set tetap (15%)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.15, random_state=42)

# Latih Random Forest untuk menentukan fitur penting
rf_temp = RandomForestClassifier(random_state=42, max_depth=3)
rf_temp.fit(X_train, y_train)

# Pilih 7 fitur terpenting
feature_importances = pd.Series(rf_temp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
important_features = list(feature_importances.nlargest(7).index)

# Data hanya menggunakan fitur yang dipilih
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# Bagian 4: Hyperparameter Tuning dan Pelatihan Model
# Definisi distribusi parameter
param_dist = {'n_estimators': randint(100, 500),
              'max_depth': randint(5, 20)}

# Randomized search
rand_search = RandomizedSearchCV(rf_temp,
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5,
                                 random_state=42)
rand_search.fit(X_train_selected, y_train)

# Model terbaik
best_rf_model = rand_search.best_estimator_
y_pred = best_rf_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model is {accuracy:.5f}")

# Bagian 5: Analisis Fitur dan Visualisasi Model
# Simpan skor dari setiap pohon dalam Random Forest
scores = [tree.score(X_train_selected, y_train) for tree in best_rf_model.estimators_]
best_tree_index = np.argmax(scores)
best_tree = best_rf_model.estimators_[best_tree_index]
print(f"Best tree index: {best_tree_index}")

# Get feature importances
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train_selected.columns).sort_values(ascending=False)

# Create the plot
print(feature_importances)
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()


from sklearn.tree import export_graphviz
import graphviz

# Export pohon ke format DOT
dot_data = export_graphviz(
    best_tree,
    feature_names=X_train_selected.columns,
    filled=True,
    class_names=['Tidak Meninggal', 'Meninggal'],
    max_depth=3,
    impurity=False,
    proportion=True
)

# Memodifikasi cabang agar memiliki label True/False
lines = dot_data.split("\n")
new_lines = []
counter = 1  # Counter untuk melacak urutan cabang

for line in lines:
    if "->" in line:  # Cabang ditemukan
        if "label=" not in line:  # Tambahkan label jika belum ada
            label = "False" if counter % 2 == 0 else "True"  # Set label berdasarkan urutan
            if "[style=" in line:  # Jika atribut style ada, tambahkan label setelahnya
                line = line.replace("[style=", f"[label={label}, style=")
            else:
                line = line.replace(";", f"[label={label}];")
        counter += 1  # Increment counter untuk cabang berikutnya
    new_lines.append(line)

# Gabungkan kembali menjadi string DOT
updated_dot_data = "\n".join(new_lines)

# Tampilkan graph
graph = graphviz.Source(updated_dot_data)
display(graph)


# Aturan dalam bentuk teks
tree_rules = export_text(best_tree, feature_names=list(X_train_selected.columns), max_depth=3)
print("Aturan dari pohon terbaik hingga kedalaman tertentu:")
print(tree_rules)

