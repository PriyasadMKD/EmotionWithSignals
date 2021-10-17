import numpy as np


transform_dict = {}
center_dict = {}

transform_dict['AF3'] = [0.74294722,0,0,0.74294722,24.042455,34.521165]
center_dict['AF3'] = [111.152,36.785999]

transform_dict['AF4'] = [0.74294722,0,0,0.74294722,112.25218,34.933359]
center_dict['AF4'] = [111.152,36.785999]

transform_dict['FC5'] = [0.74294722,0,0,0.74294722,43.053555,76.788325]
# transform_dict['FC5'] = [0.74294722,0,0,0.74294722,26.053555,55.788325]
center_dict['FC5'] = [52.964001,80.286003]

transform_dict['FC6'] = [0.74294722,0,0,0.74294722,179.70234,76.788325]
# transform_dict['FC6'] = [0.74294722,0,0,0.74294722,196.70234,55.788325]
center_dict['FC6'] = [52.964001,80.286003]

transform_dict['T8'] = [0.74294724,0,0,0.74294724,69.346924,38.500589]
center_dict['T8'] = [269.77701,149.77699]

transform_dict['T7'] = [0.74294724,0,0,0.74294724,7.7449738,38.482083]
center_dict['T7'] = [30.1299,149.705]

transform_dict['P8'] = [0.74294724,0,0,0.74294724,64.238535,56.404316]
center_dict['P8'] = [246.69701,219.427]

transform_dict['O2'] = [0.74294724,0,0,0.74294724,48.286589,67.727486]
center_dict['O2'] = [187.847,263.47699]

transform_dict['P7'] = [0.74294724,0,0,0.74294724,11.604977,56.081884]
center_dict['P7'] = [53.164001,219.86099]

transform_dict['F7'] = [0.74294724,0,0,0.74294724,13.614542,20.637738]
center_dict['F7'] = [52.964001,80.286003]

transform_dict['F8'] = [0.74294724,0,0,0.74294724,63.482519,20.637738]
center_dict['F8'] = [246.963,80.286003]

transform_dict['O1'] = [0.74294724,0,0,0.74294724,28.797877,67.806662]
center_dict['O1'] = [112.031,263.785]

transform_dict['F4'] = [0.74294724,0,0,0.74294724,41.192569,32.908336]
# transform_dict['F4'] = [0.74294724,0,0,0.74294724,51.192569,22.908336]
center_dict['F4'] = [199.15199,89.119202]

transform_dict['F3'] = [0.74294724,0,0,0.74294724,36.001401,32.908336]
# transform_dict['F3'] = [0.74294724,0,0,0.74294724,26.001401,22.908336]
center_dict['F3'] = [101.152,89.119202]


node_list = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", 'T8', 'FC6', 'F4', "F8", 'AF4']
adj_matrix = np.array([
                        [1,1,1,1,1,0,0,0,0,0,0,1,0,1],
                        [1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,0,0,0,0,0,0,0,0,1],
                        [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                        [0,0,0,1,1,1,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                        [1,0,0,0,0,0,0,0,0,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,1,1,1,1,1],
                        [1,0,1,0,0,0,0,0,0,1,1,1,1,1]])



attention_scores = np.load('save_2021-09-09.npy')

# code_snippets = open('model_code.svg',"r").readlines()

for i in range(attention_scores.shape[0]):

    array = np.transpose(attention_scores[i], (2, 0, 1))
    
    for j in range(2):

        temp_code_snippets = open('model_code.svg',"r").readlines()

        save_file_name = "Figure_"+str(i)+"_Head_"+str(j)+".svg"

        array1 = np.asarray(array[j])

        new_negihbor_weighte = np.zeros((14,14))

        
        for p in range(14):

            node = node_list[p]
            neighbor_scores = array1[p]
            median_value = 1/np.count_nonzero(neighbor_scores)
            # print(median_value)
            for m in range(neighbor_scores.shape[0]):
                if neighbor_scores[m] != 0:
                    neighbor_scores[m] = neighbor_scores[m]-median_value
            
            mina = np.amin(neighbor_scores)
            maxa = np.amax(neighbor_scores)
            
            for q in range(14):
                neigh_node = node_list[q]
                is_neighbor = adj_matrix[p][q]

                if is_neighbor==1:

                    score1 = (((array1[p][q] - mina)/(maxa-mina))*(1-0.2)+0.2)*100
                    new_negihbor_weighte[p][q] = score1

# ==================================================

        for k in range(14):
            node = node_list[k]
            # neighbor_scores = array1[k]
            # # print(neighbor_scores.shape)
            # # print(neighbor_scores)
            # # print(np.amin(neighbor_scores),np.amax(neighbor_scores),np.count_nonzero(neighbor_scores))
            
            # median_value = 1/np.count_nonzero(neighbor_scores)
            # # print(median_value)
            # for m in range(neighbor_scores.shape[0]):
            #     if neighbor_scores[m] != 0:
            #         neighbor_scores[m] = neighbor_scores[m]-median_value
            # # print(neighbor_scores)
            # mina = np.amin(neighbor_scores)
            # maxa = np.amax(neighbor_scores)
            # break
            for l in range(14):
                neigh_node = node_list[l]
                is_neighbor = adj_matrix[k][l]

                if is_neighbor==1:

                    # score1 = (((array1[k][l] - mina)/(maxa-mina))*(1-0.2)+0.2)*100
                    # score2 = (((array1[l][k] - mina)/(maxa-mina))*(1-0.2)+0.2)*100
                    
                    score1 = new_negihbor_weighte[k][l]
                    score2 = new_negihbor_weighte[l][k]

                    conv_score1 = 1 * round(score1/1)
                    conv_score2 = 1 * round(score2/1)

                    color = 100-(conv_score1+conv_score2)//2

                    current_node_transform = transform_dict[node]
                    current_node_center = center_dict[node]

                    neighbor_node_transform = transform_dict[neigh_node]
                    neighbor_node_center = center_dict[neigh_node]

                    if l ==k:
                        alter_distance = 0
                    else:
                        alter_distance = 0
                    
                    # if k == 6:
                    #     alter_distance_y = 5
                    #     alter_distance = 0
                    # elif k ==7:
                    #     alter_distance_y = -5
                    #     alter_distance = 0
                    # else:
                    #     alter_distance_y = 0
#                     newX = a * oldX + c * oldY + e = 3 * 10 - 1 * 10 + 30 = 50
#   newY = b * oldX + d * oldY + f = 1 * 10 + 3 * 10 + 40 = 80

                    x1 = current_node_transform[0]*current_node_center[0] + current_node_transform[2]*current_node_center[1]+current_node_transform[4] - alter_distance
                    y1 = current_node_transform[1]*current_node_center[0] + current_node_transform[3]*current_node_center[1]+current_node_transform[5]

                    x2 = neighbor_node_transform[0]*neighbor_node_center[0] + neighbor_node_transform[2]*neighbor_node_center[1]+neighbor_node_transform[4] + alter_distance
                    y2 = neighbor_node_transform[1]*neighbor_node_center[0] + neighbor_node_transform[3]*neighbor_node_center[1]+neighbor_node_transform[5]

                    line = "<line x1='"+str(x1)+"' y1='"+str(y1)+"' x2='"+str(x2)+"' y2='"+str(y2)+"' style='stroke:hsl(0,100%,"+str(color)+"%);stroke-width:1' />"
        
                    temp_code_snippets.insert(-1,line)
        
        with open(save_file_name, "w") as outfile:
            outfile.write("\n".join(temp_code_snippets))
        

        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF

        drawing = svg2rlg(save_file_name)
        renderPDF.drawToFile(drawing, save_file_name.replace("svg","pdf"))

    #     break
    # break