# # Menghitung jumlah duplikasi data
# num_duplicates = df.duplicated().sum()

# print("Jumlah nilai duplikasi data:", num_duplicates)

# # Menghapus duplikasi data dan menyimpan hasilnya
# df = df.drop_duplicates()

print("\nHasil setelah menghapus duplikasi data:\n")
df.shape

# Memeriksa nilai yang hilang di DataFrame
print("Nilai yang hilang sebelum imputasi:\n", df.isnull().sum())

# Menangani nilai yang hilang (contoh: isi dengan rata-rata untuk numerik, modus untuk kategorikal)
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

# Mengisi nilai yang hilang untuk kolom numerik dengan rata-rata
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Mengisi nilai yang hilang untuk kolom kategorikal dengan modus
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Memverifikasi apakah nilai yang hilang sudah ditangani
print("\nNilai yang hilang setelah imputasi:\n", df.isnull().sum())

# Menampilkan DataFrame
df.head()

# Loop melalui setiap kolom numerik dalam DataFrame
for column in df.select_dtypes(include=np.number):
    if column == 'DEATH_EVENT':
        continue  # Lewati kolom 'DEATH_EVENT'

    # Buat box plot untuk kolom tersebut
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot untuk {column}")
    plt.show()

    # Hitung jumlah outlier menggunakan IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    num_outliers = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
    # Hapus outlier dari DataFrame
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Jumlah outlier dalam kolom '{column}': {num_outliers}")

# Menampilkan dimensi DataFrame (jumlah baris dan kolom)
df.shape

# Menampilkan DataFrame setelah outlier dihapus
print("\nDataFrame setelah outlier dihapus:\n")
df.head()

# Simpan DataFrame yang telah di-preprocessing ke dalam file CSV
df.to_csv('update_heart_failure.csv', index=False)
