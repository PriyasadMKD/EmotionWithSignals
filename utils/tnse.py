from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.weight': 'bold'})

def tsne_function(X,y,N,save_name,titles=None):

    np.random.seed(42)
    

    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1] 
    df_subset['pca-three'] = pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure(figsize=(8,6))

    plt.rcParams.update({'font.size': 14,'font.weight':'bold'})

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        alpha=1
    )
    L=plt.legend()
    plt.title(titles)
    L.get_texts()[1].set_text('scale value <= 3')
    L.get_texts()[2].set_text('scale value > 3')
    # L.get_texts()[3].set_text('scale value > 3')

    plt.savefig(save_name,bbox_inches='tight',dpi=3000)


objects = []

with (open("../data/TSNE_GENERAL_FOLD1.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

data = objects[0]

arousal = np.concatenate(data['a'], axis=0)
valence = np.concatenate(data['v'], axis=0)
dominance = np.concatenate(data['d'], axis=0)

print(arousal.shape,valence.shape,dominance.shape)

labels = data['l']
# /home/n10370986/Dreamer/data/segmented/Subject3/Subject_3_Video_15_3_1_1_Segment55.0_EEG_STIM.npy\n
lat_list = [item for sublist in labels for item in sublist]

arousal_labels = np.zeros((len(lat_list)))
valence_labels = np.zeros((len(lat_list)))
dominance_labels = np.zeros((len(lat_list)))

for i in range(len(lat_list)):
    ar_la,val_lab,dom_lab = map(int,lat_list[i].strip().split("/")[-1].split("_")[4:7])
    # print(ar_la,val_lab,dom_lab)
    
    
    # arousal_labels[i] = 1 if ar_la > 3 else 0
    # valence_labels[i] = 1 if val_lab > 3 else 0
    # dominance_labels[i] = 1 if dom_lab > 3 else 0

    if ar_la > 3 :
        arousal_labels[i] = 1
    # elif ar_la == 3 :
    #     arousal_labels[i] = 1
    else:
        arousal_labels[i] = 0

    if val_lab > 3 :
        valence_labels[i] = 1
    # elif val_lab == 3 :
    #     valence_labels[i] = 1
    else:
        valence_labels[i] = 0

    if dom_lab > 3 :
        dominance_labels[i] = 1
    # elif dom_lab == 3 :
    #     dominance_labels[i] = 1
    else:
        dominance_labels[i] = 0

    
tsne_function(arousal,arousal_labels,arousal_labels.shape[0],'Arousal_TSNE.pdf','Arousal')
tsne_function(valence,valence_labels,valence_labels.shape[0],'Valence_TSNE.pdf','Valence')
tsne_function(dominance,dominance_labels,dominance_labels.shape[0],'Dominance_TSNE.pdf','Dominance')








# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.targetprint(X.shape, y.shape)[out] (70000, 784) (70000,)