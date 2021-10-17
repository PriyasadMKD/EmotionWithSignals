import pickle5 as pickle
import os

def read_pickle(file_name):

    if not os.path.exists(file_name):
        return 
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    whole_data = objects[0]['val_acc']

    maimum = 0
    
    epoch = None

    i=0
    for item in whole_data:

        if sum(item)/len(item) > maimum:
            maimum = sum(item)/len(item)
            epoch=i
        i+=1
    
    print(maimum,epoch)

    sn = "../weights/"+file_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(maimum)+'.pkl'

    if os.path.exists(sn):
        print(sn)


def read_pickle1(file_name):

    if not os.path.exists(file_name):
        return 
    objects = []
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    # print(objects[0].keys())
    whole_data = objects[0]['val_acc']

    maimum = 0
    
    epoch = None

    i=0
    # print(whole_data)
    for item in whole_data:
        # print(item)
        if item > maimum:
            maimum = item
            epoch=i
        i+=1
    
    print(maimum,epoch)

    sn = "../weights/"+file_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(maimum)+'.pkl'

    if os.path.exists(sn):
        print(sn)

# Emotion_Fold_0_Conv_Channel_Adj1.pkl
# Emotion_Fold_0_Conv_Channel.pkl
# Emotion_Fold_0_Conv_NumHead6.pkl
# Emotion_Fold_0_Conv_NumHead5.pkl
# Emotion_Fold_0_Conv_NumHead3.pkl
# Emotion_Fold_0_Conv_NumHead4.pkl

# Emotion_Subject_0_Conv_Channel_Arousal_LOSO.pkl

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Arousal_LOSO.pkl')

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Valence_LOSO.pkl')

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Dominance_LOSO.pkl')

# Emotion_Fold_3_Conv_Channel_Valence.pkl

# Emotion_Fold_8_Conv_Channel_ValenceN.pkl

# Emotion_Subject_4_Conv_Channel_Valence_LOSON.pkl

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Arousal_LOSON.pkl')

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Valence_LOSON.pkl')

# for i in range(23):
#     read_pickle1('/home/n10370986/Dreamer/history/Emotion_Subject_'+str(i)+'_Conv_Channel_Dominance_LOSON.pkl')

# Emotion_Fold_2_Conv_Channel_Time2.pkl           Emotion_Fold_8_Conv_Channel_Adj4.pkl                Emotion_Subject_3_Conv_Channel_Dominance_LOSON.pkl
# Emotion_Fold_2_Conv_Channel_Time44.pkl

# Emotion_Fold_0_Conv_Channel_DominanceN.pkl

for i in range(10):
    read_pickle1('/home/n10370986/Dreamer/history/Emotion_Fold_'+str(i)+'_Conv_Channel_DominanceN.pkl')

# for i in range(10):
#     read_pickle('/home/n10370986/Dreamer/history/Emotion_Fold_'+str(i)+'_Conv_Channel_Time44.pkl')

# for i in range/'/home/n10370986/Dreamer/history/Emotion_Fold_'+str(i)+'_Conv_Channel_DominanceR.pkl')



# for i in range(10):
    # read_pickle('/home/n10370986/Dreamer/history/Emotion_Fold_'+str(i)+'_Conv_NumHead3.pkl')


