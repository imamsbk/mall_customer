import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#membaca dataset
df = pd.read_csv("Mall_Customers.csv")

#data preaparation
df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
},inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)


st.header('Isi Dataset')
st.write(X)


#membuat elbow, untuk menentukan cluster yang optimal
clusters=[]
for i in range (1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig,ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title("Mencari Elbow")
ax.set_xlabel("clusters")
ax.set_ylabel("inertia")

#panah elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3', color='blue', lw=2))

##st.set_option('deprecation.showPyplotGlobalUse', False)
##elbo_plot = st.pyplot()

st.pyplot(fig)

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster : ", 2, 10, 3, 1)

def k_means(n_clust): 
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    ##kmeans = KMeans(n_clusters=4).fit(X)
    ##X['Labels'] = kmeans.labels_

    # Membuat scatter plot
#    plt.figure(figsize=(10, 8))
#    sns.scatterplot(
#        x='Income', y='Score', hue='Labels',
#        data=X, palette=sns.color_palette('hls', n_clust),
#        size='Labels', sizes=(50, 200)
#    )

    # Menambahkan anotasi
 #   for label in X['Labels'].unique():
#        mean_income = X[X['Labels'] == label]['Income'].mean()
  #      mean_score = X[X['Labels'] == label]['Score'].mean()
 #       plt.annotate(
#            f'Cluster {label}',
 #           (mean_income, mean_score),
#            horizontalalignment='center',
#            verticalalignment='center',
#            size=15, weight='bold', color='black'
#       )
    
#    st.header('Cluster Plot')
#    st.pyplot()
#    st.write(X)


    # Membuat plot
    fig, ax = plt.subplots(figsize=(10, 8))  # Buat objek figure dan axis
    sns.scatterplot(
        x='Income', y='Score', hue='Labels',
        data=X, palette=sns.color_palette('hls', n_clust),
        size='Labels', sizes=(50, 200), ax=ax
    )

    # Menambahkan anotasi
    for label in X['Labels'].unique():
        mean_income = X[X['Labels'] == label]['Income'].mean()
        mean_score = X[X['Labels'] == label]['Score'].mean()
        ax.annotate(
            f'Cluster {label}',
            (mean_income, mean_score),
            horizontalalignment='center',
            verticalalignment='center',
            size=15, weight='bold', color='black'
        )

    # Menampilkan plot di Streamlit
    st.header('Cluster Plot')
    st.pyplot(fig)  # Menampilkan plot menggunakan objek `fig`
    st.write(X)  # Menampilkan DataFrame di Streamlit



k_means(clust)