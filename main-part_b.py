"""
Introduction to Machine Learning - Programming Assignment
Exercise 04
December 2020
Yotam Leibovitz
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
from mlxtend.data import loadlocal_mnist
from scipy.spatial.distance import cdist

#########################################################################
#                Load Dataset                                           #
#########################################################################
w_1 = np.array([[0.55463477]
, [0.19577211]
, [0.936441  ]
, [0.38479091]
, [0.19677097]
, [0.82336811]
, [0.33158275]
, [0.62521739]
, [0.08745383]
, [0.67046417]
, [0.41819745]
, [0.61595693]
, [0.03132848]
, [0.09501537]
, [0.26546122]
, [0.95649817]
, [0.51923139]
, [0.68559747]
, [0.09714508]
, [0.9044466 ]
, [0.26965886]
, [0.17211266]
, [0.92461366]
, [0.8448675 ]
, [0.96600273]
, [0.08628544]
, [0.95714248]
, [0.04467034]
, [0.01862983]
, [0.35837411]
, [0.81113483]
, [0.90727734]
, [0.37709382]
, [0.67045779]
, [0.47055848]
, [0.78924226]
, [0.98587121]
, [0.12065573]
, [0.81669892]
, [0.04982993]
, [0.3865741 ]
, [0.35674243]
, [0.19263358]
, [0.66992687]
, [0.45032482]
, [0.703376  ]
, [0.63286995]
, [0.28011731]
, [0.21833201]
, [0.1330583 ]
, [0.49508074]
, [0.10172299]
, [0.31876385]
, [0.54970262]
, [0.70507804]
, [0.41627166]
, [0.2238479 ]
, [0.50409028]
, [0.82870748]
, [0.1484074 ]
, [0.41274679]
, [0.83148813]
, [0.32795232]
, [0.58349126]
, [0.3762341 ]
, [0.7648062 ]
, [0.15230003]
, [0.95327941]
, [0.48942812]
, [0.97287748]
, [0.49637794]
, [0.49948571]
, [0.52961629]
, [0.99165184]
, [0.04860502]
, [0.80647053]
, [0.80476793]
, [0.02962515]
, [0.78876108]
, [0.22449345]
, [0.05261216]
, [0.55125087]
, [0.59509736]
, [0.35199417]
, [0.15502186]
, [0.32581559]
, [0.98984298]
, [0.86025048]
, [0.56832136]
, [0.35868092]
, [0.06842832]
, [0.08351363]
, [0.67480479]
, [0.57169386]
, [0.28158964]
, [0.99824093]
, [0.65544107]
, [0.62384062]
, [0.52579594]
, [0.76389976]
, [0.44097448]
, [0.98918585]
, [0.21252039]
, [0.96082409]
, [0.48574106]
, [0.58427481]
, [0.91310976]
, [0.77290531]
, [0.30391949]
, [0.55228007]
, [0.8848888 ]
, [0.50672985]
, [0.14690017]
, [0.56567161]
, [0.51305175]
, [0.01125748]
, [0.886218  ]
, [0.30157318]
, [0.70871679]
, [0.46249974]
, [0.86497819]
, [0.7208612 ]
, [0.24203131]
, [0.452416  ]
, [0.74591176]
, [0.61575202]
, [0.52039291]
, [0.87938345]
, [0.52974238]
, [0.49842037]
, [0.84020221]
, [0.48352017]
, [0.03263206]
, [0.66297276]
, [0.3889738 ]
, [0.08896625]
, [0.71355851]
, [0.47931509]
, [0.46319053]
, [0.32219175]
, [0.07710412]
, [0.38317827]
, [0.97074542]
, [0.87506854]
, [0.44903118]
, [0.58185684]
, [0.43775317]
, [0.01336597]
, [0.38478315]
, [0.82876199]
, [0.96317886]
, [0.66347266]
, [0.82684797]
, [0.38458808]
, [0.8318038 ]
, [0.06298232]
, [0.562545  ]
, [0.08167758]
, [0.15829588]
, [0.01386699]
, [0.8554915 ]
, [0.52545632]
, [0.8714424 ]
, [0.27333779]
, [0.25441205]
, [0.82620045]
, [0.24703886]
, [0.48282114]
, [0.18095159]
, [0.32738999]
, [0.96053498]
, [0.85938459]
, [0.40257892]
, [0.11176613]
, [0.36490566]
, [0.63672592]
, [0.4843965 ]
, [0.0872216 ]
, [0.21698289]
, [0.87893777]
, [0.82566438]
, [0.87993715]
, [0.92347364]
, [0.57852246]
, [0.08587464]
, [0.41961883]
, [0.35950243]
, [0.82288031]
, [0.25440034]
, [0.30924535]
, [0.05777284]
, [0.25768447]
, [0.41014147]
, [0.78752736]
, [0.22013783]
, [0.60951023]
, [0.51280509]
, [0.64652954]
, [0.62754864]
, [0.87032815]
, [0.82631802]
, [0.91357269]
, [0.13642169]
, [0.29995637]
, [0.1918368 ]
, [0.36073072]
, [0.95416144]
, [0.80597158]
, [0.22264956]
, [0.94973395]
, [0.18145105]
, [0.56560715]
, [0.78671488]
, [0.2466018 ]
, [0.67065481]
, [0.92592903]
, [0.61664265]
, [0.66672478]
, [0.95712536]
, [0.73899712]
, [0.08815014]
, [0.02386564]
, [0.37784508]
, [0.74982907]
, [0.78615496]
, [0.95754457]
, [0.91080212]
, [0.96817466]
, [0.17503505]
, [0.08108005]
, [0.12307448]
, [0.6189344 ]
, [0.33774837]
, [0.51273709]
, [0.0377686 ]
, [0.74643605]
, [0.34044552]
, [0.51147809]
, [0.72079597]
, [0.84536047]
, [0.90452447]
, [0.40468455]
, [0.70296284]
, [0.44681838]
, [0.69541945]
, [0.0746279 ]
, [0.95703719]
, [0.95456116]
, [0.31731269]
, [0.61108903]
, [0.01530975]
, [0.65496891]
, [0.13639614]
, [0.16414241]
, [0.42607707]
, [0.11419045]
, [0.09293472]
, [0.51861409]
, [0.13011083]
, [0.70497576]
, [0.11856778]
, [0.62936458]
, [0.83809407]
, [0.38118685]
, [0.95584199]
, [0.50268524]
, [0.00421859]
, [0.51047076]
, [0.25663461]
, [0.49997098]
, [0.43018534]
, [0.43280907]
, [0.44385925]
, [0.67017497]
, [0.29546003]
, [0.96430188]
, [0.09790328]
, [0.05631272]
, [0.15405202]
, [0.00628463]
, [0.11930883]
, [0.45448903]
, [0.59571581]
, [0.43887758]
, [0.49467148]
, [0.69540201]
, [0.74600473]
, [0.40164483]
, [0.99416958]
, [0.58833255]
, [0.34021446]
, [0.30955788]
, [0.61752376]
, [0.65719506]
, [0.55785525]
, [0.95633285]
, [0.82540014]
, [0.47202121]
, [0.49989406]
, [0.5707004 ]
, [0.50081565]
, [0.24150957]
, [0.91259583]
, [0.97821912]
, [0.84416133]
, [0.73795295]
, [0.25828617]
, [0.76729534]
, [0.44217364]
, [0.43409658]
, [0.36076295]
, [0.14860842]
, [0.81796096]
, [0.17444456]
, [0.72966816]
, [0.56419362]
, [0.42915443]
, [0.42377453]
, [0.15492692]
, [0.38368162]
, [0.90701963]
, [0.56161902]
, [0.96425066]
, [0.84544321]
, [0.65702946]
, [0.60256654]
, [0.8494562 ]
, [0.51326873]
, [0.5426688 ]
, [0.36558798]
, [0.18795975]
, [0.50746961]
, [0.65834118]
, [0.9461925 ]
, [0.54049968]
, [0.83318377]
, [0.50156971]
, [0.21009992]
, [0.83364608]
, [0.98053356]
, [0.61013462]
, [0.7294427 ]
, [0.24976506]
, [0.07633897]
, [0.66350484]
, [0.77025221]
, [0.32940827]
, [0.51473807]
, [0.50119478]
, [0.2362605 ]
, [0.15100843]
, [0.42120492]
, [0.9896505 ]
, [0.17658176]
, [0.33119302]
, [0.68409358]
, [0.83966989]
, [0.91696298]
, [0.09792092]
, [0.21787648]
, [0.09528269]
, [0.00102941]
, [0.23544208]
, [0.17341508]
, [0.90307208]
, [0.21267368]
, [0.1356887 ]
, [0.34912395]
, [0.76446697]
, [0.26609996]
, [0.92007279]
, [0.05313265]
, [0.97376978]
, [0.1590194 ]
, [0.46310125]
, [0.78814037]
, [0.34115101]
, [0.91302994]
, [0.1448836 ]
, [0.64617614]
, [0.70487739]
, [0.51279885]
, [0.40133111]
, [0.35851414]
, [0.93657398]
, [0.97940099]
, [0.04250272]
, [0.3620179 ]
, [0.18248978]
, [0.5879654 ]
, [0.84194928]
, [0.45292475]
, [0.05866083]
, [0.320118  ]
, [0.85735879]
, [0.55017408]
, [0.097253  ]
, [0.92570494]
, [0.48159344]
, [0.45940742]
, [0.17407948]
, [0.4879785 ]
, [0.33652138]
, [0.90339304]
, [0.56196496]
, [0.66488186]
, [0.32747761]
, [0.43040498]
, [0.78800067]
, [0.99091363]
, [0.25537899]
, [0.27015082]
, [0.38447617]
, [0.86938343]
, [0.51450246]
, [0.77827713]
, [0.9854625 ]
, [0.19915148]
, [0.43908168]
, [0.2344202 ]
, [0.53477602]
, [0.5340183 ]
, [0.18134317]
, [0.07231332]
, [0.6456096 ]
, [0.92965685]
, [0.77173254]
, [0.37305389]
, [0.65260173]
, [0.7196976 ]
, [0.73871093]
, [0.86417671]
, [0.99517621]
, [0.4422133 ]
, [0.49869226]
, [0.76959782]
, [0.82993087]
, [0.73114022]
, [0.763854  ]
, [0.5406093 ]
, [0.81418719]
, [0.21753321]
, [0.94149182]
, [0.77634195]
, [0.44958803]
, [0.22603893]
, [0.09555945]
, [0.74967548]
, [0.27838934]
, [0.04639446]
, [0.17703584]
, [0.07218884]
, [0.71035641]
, [0.9275567 ]
, [0.75428184]
, [0.4197339 ]
, [0.10073277]
, [0.59698555]
, [0.6418407 ]
, [0.9368383 ]
, [0.92229028]
, [0.6604828 ]
, [0.34567859]
, [0.37096559]
, [0.9170764 ]
, [0.09937856]
, [0.36086054]
, [0.79821093]
, [0.25026491]
, [0.79836327]
, [0.64373208]
, [0.71634185]
, [0.22210033]
, [0.06847859]
, [0.60085462]
, [0.16331659]
, [0.89539764]
, [0.70312385]
, [0.66893855]
, [0.97413399]
, [0.031424  ]
, [0.6530488 ]
, [0.16040073]
, [0.53296306]
, [0.63683367]
, [0.14141776]
, [0.4741332 ]
, [0.71862822]
, [0.9239145 ]
, [0.23847405]
, [0.37442146]
, [0.44930216]
, [0.68782593]
, [0.64109055]
, [0.21943474]
, [0.21378309]
, [0.72549092]
, [0.11821172]
, [0.99691603]
, [0.78209789]
, [0.72038895]
, [0.2173807 ]
, [0.9382999 ]
, [0.75226351]
, [0.98063186]
, [0.89912748]
, [0.84137989]
, [0.7127591 ]
, [0.84743184]
, [0.99443434]
, [0.76075598]
, [0.41935721]
, [0.72299162]
, [0.0112254 ]
, [0.87139009]
, [0.60687452]
, [0.07810383]
, [0.69646224]
, [0.33703116]
, [0.93220187]
, [0.47201296]
, [0.54761248]
, [0.68985085]
, [0.62649842]
, [0.63212131]
, [0.56859233]
, [0.46140192]
, [0.66089424]
, [0.92592231]
, [0.33001606]
, [0.64193856]
, [0.30025546]
, [0.0502641 ]
, [0.06384681]
, [0.70535529]
, [0.37546493]
, [0.28844573]
, [0.75884821]
, [0.75057826]
, [0.6989199 ]
, [0.48447075]
, [0.16423211]
, [0.58239331]
, [0.477046  ]
, [0.86137565]
, [0.01628231]
, [0.04786625]
, [0.59402837]
, [0.1225022 ]
, [0.17379309]
, [0.49054397]
, [0.58664016]
, [0.36823633]
, [0.77479861]
, [0.2440906 ]
, [0.8068202 ]
, [0.95091719]
, [0.03530636]
, [0.58538599]
, [0.48153974]
, [0.77466996]
, [0.2785666 ]
, [0.28737963]
, [0.71418898]
, [0.7265528 ]
, [0.13318593]
, [0.46385837]
, [0.64203005]
, [0.04884648]
, [0.34816704]
, [0.48261166]
, [0.55013023]
, [0.84180335]
, [0.33229389]
, [0.02914591]
, [0.16352828]
, [0.85193537]
, [0.93975295]
, [0.61286224]
, [0.78073984]
, [0.44616109]
, [0.67449532]
, [0.50564497]
, [0.02316613]
, [0.3854354 ]
, [0.44760453]
, [0.31915049]
, [0.97492598]
, [0.29550836]
, [0.83064052]
, [0.99832567]
, [0.56597536]
, [0.50314347]
, [0.72925908]
, [0.86500542]
, [0.48261057]
, [0.43489104]
, [0.81184523]
, [0.7826833 ]
, [0.50056885]
, [0.28893562]
, [0.65292225]
, [0.68009233]
, [0.87660969]
, [0.89061105]
, [0.63055269]
, [0.13591556]
, [0.28345429]
, [0.21034265]
, [0.301531  ]
, [0.17742527]
, [0.40374687]
, [0.11050097]
, [0.64568909]
, [0.95838897]
, [0.09555158]
, [0.2693128 ]
, [0.24618392]
, [0.33548428]
, [0.42943561]
, [0.15494575]
, [0.7666283 ]
, [0.60702963]
, [0.05714719]
, [0.27957601]
, [0.82532435]
, [0.14655688]
, [0.27977411]
, [0.19321425]
, [0.92029275]
, [0.50317898]
, [0.10147546]
, [0.29700063]
, [0.03051254]
, [0.28533673]
, [0.39865547]
, [0.55958085]
, [0.88881875]
, [0.30127504]
, [0.24833392]
, [0.55609398]
, [0.15573885]
, [0.23290556]
, [0.06018822]
, [0.09056024]
, [0.48885318]
, [0.95396535]
, [0.02996541]
, [0.66000373]
, [0.15107521]
, [0.72449405]
, [0.31188184]
, [0.8843186 ]
, [0.56865592]
, [0.4145311 ]
, [0.55848983]
, [0.35855374]
, [0.34206725]
, [0.46018919]
, [0.01534231]
, [0.88649105]
, [0.92637166]
, [0.43158762]
, [0.97877004]
, [0.26071164]
, [0.20641812]
, [0.36864049]
, [0.62129349]
, [0.01046461]
, [0.78024873]
, [0.52228771]
, [0.73233651]
, [0.84823374]
, [0.06871747]
, [0.16132971]
, [0.42380442]
, [0.4243445 ]
, [0.26210916]
, [0.39817876]
, [0.14810985]
, [0.68897416]
, [0.39146342]
, [0.51548538]
, [0.40258735]
, [0.99492673]
, [0.50692013]
, [0.21147887]
, [0.63425987]
, [0.88197641]
, [0.92493568]
, [0.98065909]
, [0.2219757 ]
, [0.87974908]
, [0.41649968]
, [0.07417261]
, [0.07699027]
, [0.38882454]
, [0.69054794]
, [0.84988556]
, [0.4890829 ]
, [0.4130497 ]
, [0.14231303]
, [0.77349392]
, [0.16813075]
, [0.14488836]
, [0.54337345]
, [0.39236325]
, [0.17234516]
, [0.95916756]
, [0.74666799]
, [0.03353759]
, [0.23855781]
, [0.0180512 ]
, [0.81438223]
, [0.18416077]
, [0.04956137]
, [0.36879454]
, [0.43697058]
, [0.33777674]
, [0.17231448]
, [0.88026203]
, [0.5954544 ]
, [0.49674384]
, [0.28621359]
, [0.02007729]
, [0.71423747]
, [0.75371114]
, [0.76972257]
, [0.34954887]
, [0.52220646]
, [0.70741777]
, [0.77520786]
, [0.61018306]
, [0.38034239]
, [0.9800899 ]
, [0.06154887]
, [0.75691758]
, [0.51369468]
, [0.14407792]
, [0.3564953 ]
, [0.96655583]
, [0.85108447]
, [0.16373859]
, [0.25679893]
, [0.70288961]
, [0.01843677]
, [0.34549717]
, [0.50866181]
, [0.4233022 ]
, [0.27923931]
, [0.12109764]
, [0.94132555]
, [0.25282202]
, [0.14238608]
, [0.33734138]
, [0.52096326]
, [0.55361487]
, [0.23722576]
, [0.82239751]
, [0.91179282]
, [0.02354934]
, [0.69434075]
, [0.74283229]
, [0.77201062]
, [0.94607171]
, [0.48736148]
, [0.96858513]
, [0.59343844]
, [0.45056793]
, [0.94106809]
, [0.90699358]
, [0.23230729]
, [0.51440081]
, [0.69679149]
, [0.80127981]
, [0.76244335]
, [0.01219831]
, [0.02107346]
, [0.71918065]
, [0.74720634]
, [0.24388335]
, [0.64446649]
, [0.92726813]
, [0.05644929]
, [0.13219103]])

# load MNIST train and test data
X_src_1, t_src_1 = loadlocal_mnist(
            images_path=r'C:\Users\yotam\PycharmProjects\ATILT-project\MNIST\train-images.idx3-ubyte',
            labels_path=r'C:\Users\yotam\PycharmProjects\ATILT-project\MNIST\train-labels.idx1-ubyte')
X_src_2, t_src_2 = loadlocal_mnist(
            images_path=r'C:\Users\yotam\PycharmProjects\ATILT-project\MNIST\t10k-images.idx3-ubyte',
            labels_path=r'C:\Users\yotam\PycharmProjects\ATILT-project\MNIST\t10k-labels.idx1-ubyte')

# combine to one MNIST data set of 70000 samples
X_src = np.vstack((X_src_1, X_src_2)).astype('float64')
t_src = np.concatenate((t_src_1, t_src_2), axis=0).astype('int')


# fetch MNIST if not found locally
# mnist = fetch_openml('mnist_784')
# X_src = mnist['data'].astype('float64')
# t_src = mnist['target']

# Split the data into 3 different binary classification problems
index_01 = (t_src == 0) | (t_src == 1)
t_01 = t_src[index_01]  # target labels of the digit '0' or '1' only
X_01 = X_src[index_01]  # data samples of the digit '0' or '1' only

index_23 = (t_src == 2) | (t_src == 3)
t_23 = t_src[index_23]  # target labels of the digit '2' or '3' only
X_23 = X_src[index_23]  # data samples of the digit '2' or '3' only

index_45 = (t_src == 4) | (t_src == 5)
t_45 = t_src[index_45]  # target labels of the digit '4' or '5' only
X_45 = X_src[index_45]  # data samples of the digit '4' or '5' only

# convert labels to binary {-1, 1} (According to lecture notes 5)
t_01[t_01 == 0] = -1
t_01[t_01 == 1] = 1

t_23[t_23 == 2] = -1
t_23[t_23 == 3] = 1

t_45[t_45 == 4] = -1
t_45[t_45 == 5] = 1


# t_list = [t_0_1_only, t_2_3_only, t_4_5_only]
# X_list = [X_0_1_only, X_2_3_only, X_4_5_only]

# train test split
# shuffle the samples - important : make sure each run the permutation is different


X_01 = X_01.reshape((X_01.shape[0], -1))  # This line flattens the image into a vector of size 784
X_01_train, X_01_test, t_01_train, t_01_test = X_01[0:10346], X_01[10346:14780], t_01[0:10346], t_01[10346:14780]


X_23 = X_23.reshape((X_23.shape[0], -1))  # This line flattens the image into a vector of size 784
X_23_train, X_23_test, t_23_train, t_23_test = X_23[0:9891], X_23[9891:14131], t_23[0:9891], t_23[9891:14131]


X_45 = X_45.reshape((X_45.shape[0], -1))  # This line flattens the image into a vector of size 784
X_45_train, X_45_test, t_45_train, t_45_test = X_45[0:9195], X_45[9195:13137], t_45[0:9195], t_45[9195:13137]

# split again to obtain validation set (20% validation, 20% test)
# X_validation, X_test, t_validation, t_test = train_test_split(X_test, t_test, test_size=0.5)

# plt.imshow(X_2_3_only[0].reshape((28,28)))

#########################################################################
#                Preprocessing                                          #
#########################################################################
# TODO: try doing only shifting to mean zero without changing the variance + normalizing to unit norm
NORMALIZE_SAMPLES = True
STANDARDIZE_SAMPLES = True

# The next lines standardize the images
if NORMALIZE_SAMPLES:
    # normalize norms to 1
    normalizer = Normalizer()
    X_01 = normalizer.fit_transform(X_01)
    X_23 = normalizer.fit_transform(X_23)
    X_45 = normalizer.fit_transform(X_45)

    # normalize train-test data sets
    X_01_train = normalizer.fit_transform(X_01_train)
    X_01_test = normalizer.fit_transform(X_01_test)
    X_23_train = normalizer.fit_transform(X_23_train)
    X_23_test = normalizer.fit_transform(X_23_test)
    X_45_train = normalizer.fit_transform(X_45_train)
    X_45_test = normalizer.fit_transform(X_45_test)

if STANDARDIZE_SAMPLES:
    scaler = StandardScaler(with_std=False)  # translate and scale sample vectors to zero only
    # scaler = StandardScaler()  # translate and scale sample vectors to zero mean and unit variance

    # print(np.mean(X_01), np.std(X_01))
    # print(np.mean(X_23), np.std(X_23))
    # print(np.mean(X_45), np.std(X_45))

    # standardize full data sets
    X_01 = scaler.fit_transform(X_01)
    X_23 = scaler.fit_transform(X_23)
    X_45 = scaler.fit_transform(X_45)

    # print(np.mean(X_01), np.std(X_01))
    # print(np.mean(X_23), np.std(X_23))
    # print(np.mean(X_45), np.std(X_45))

    # standardize train-test data sets
    X_01_train = scaler.fit_transform(X_01_train)
    X_01_test = scaler.transform(X_01_test)  # translate and scale the test data with the train mean variance values
    X_23_train = scaler.fit_transform(X_23_train)
    X_23_test = scaler.transform(X_23_test)  # translate and scale the test data with the train mean variance values
    X_45_train = scaler.fit_transform(X_45_train)
    X_45_test = scaler.transform(X_45_test)  # translate and scale the test data with the train mean variance values


# add 1 at the of every sample for the bias term
X_01 = np.hstack((X_01, np.ones((X_01.shape[0], 1))))
X_23 = np.hstack((X_23, np.ones((X_23.shape[0], 1))))
X_45 = np.hstack((X_45, np.ones((X_45.shape[0], 1))))

X_01_train = np.hstack((X_01_train, np.ones((X_01_train.shape[0], 1))))
X_01_test = np.hstack((X_01_test, np.ones((X_01_test.shape[0], 1))))
X_23_train = np.hstack((X_23_train, np.ones((X_23_train.shape[0], 1))))
X_23_test = np.hstack((X_23_test, np.ones((X_23_test.shape[0], 1))))
X_45_train = np.hstack((X_45_train, np.ones((X_45_train.shape[0], 1))))
X_45_test = np.hstack((X_45_test, np.ones((X_45_test.shape[0], 1))))

# convert from chars to int
t_01 = t_01.astype(int)
t_23 = t_23.astype(int)
t_45 = t_45.astype(int)

t_01_train = t_01_train.astype(int)
t_01_test = t_01_test.astype(int)
t_23_train = t_23_train.astype(int)
t_23_test = t_23_test.astype(int)
t_45_train = t_45_train.astype(int)
t_45_test = t_45_test.astype(int)

# convert from shape = (m, ) to shape = (m,1)
t_01 = t_01.reshape(t_01.shape[0], 1)
t_23 = t_23.reshape(t_23.shape[0], 1)
t_45 = t_45.reshape(t_45.shape[0], 1)

t_01_train = t_01_train.reshape(t_01_train.shape[0], 1)
t_01_test = t_01_test.reshape(t_01_test.shape[0], 1)
t_23_train = t_23_train.reshape(t_23_train.shape[0], 1)
t_23_test = t_23_test.reshape(t_23_test.shape[0], 1)
t_45_train = t_45_train.reshape(t_45_train.shape[0], 1)
t_45_test = t_45_test.reshape(t_45_test.shape[0], 1)

#########################################################################
#                Auxiliary Functions and Classes                        #
#########################################################################
def reshuffle_train_test(_X, _t):
    seed = np.random.randint(1, 2 ** 30)
    random_state = check_random_state(seed)
    permutation = random_state.permutation(X.shape[0])
    _X = _X[permutation]
    _t = _t[permutation]
    _X = _X.reshape((_X.shape[0], -1))  # This line flattens the image into a vector of size 784
    return train_test_split(_X, _t, test_size=0.3)



def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def get_datetime():
    return time.asctime(time.localtime()).replace(' ', '_').replace(':', "-")


def compute_Y(X, w):
    """
    Y is the output of the classifier - vector of size (m,1) of the estimated class of each sample
    Y = sign(w@X) TODO: try sigmoid instead of sign (and then thresholding at y=0.5)
    """
    Y = np.sign(np.dot(X, w))
    return Y

    # for i in range(A.shape[0]):
    #     # prevent overflow by subtracting the max value from each entry in row i
    #     A[i, :] = A[i, :] - A[i, :].max()
    #     A[i, :] = np.exp(A[i, :])
    #     Y[i, :] = A[i, :] / A[i, :].sum()


def compute_loss(X, t, w):  # TODO: try to change to square loss since it's hessian is easier to obtain
    """
    Log loss function

    X : matrix with m rows and n columns, where m is the number of samples and n is the dimension of the samples
        (the k-th sample is the k-th row of X). X can also be a single sample (when m=1)
    t : vector of the labels (size mx1). t can also be a single label (when m=1)
    w : vector of the weights (size nx1)
    """
    A = np.dot(X, w)
    m = t.shape[0]
    return (1 / m) * np.sum(np.log(1 + np.exp(-1 * A * t)))


def compute_accuracy(X, t, w):
    Y = compute_Y(X, w)
    correct_classifications = (Y == t).sum()
    accuracy = correct_classifications / X.shape[0]
    return accuracy


def step(w, grad_E, learning_rate):
    # print("norm w = ", np.linalg.norm(w))
    w_new = w - learning_rate * grad_E
    return w_new


def compute_gradient(X, t, w):  # TODO: try to change to square loss since it's hessian is easier to obtain
    """
        Gradient of log loss function at the current w
    """
    # TODO : print to console the max gradient in every run
    A = np.dot(X, w)
    m = t.shape[0]
    C = -1 * t * (1 / (1 + np.exp(A * t)))
    return (1 / m) * np.dot(X.T, C)


def hessian_eigenvalues(X, t, w):  # TODO: add function that returns the eigenvalues of the hessian of the loss function
    pass


def estimate_beta(X):  # TODO: add function that returns the estimated beta of the beta-smooth loss function
    # calculate the max and mean of square norm of the samples
    X_norm2 = np.power(np.linalg.norm(X, axis=1), 2)
    X_norm2_max = np.max(X_norm2)
    X_norm2_mean = np.mean(X_norm2)
    print("max norm X sqaure is ", X_norm2_max)
    print("Beta = mean norm X square is ", X_norm2_mean)
    return X_norm2_mean


def max_x_norm(X):  # TODO: add function that returns the estimated beta of the beta-smooth loss function
    # calculate the max and mean of square norm of the samples
    X_norm = np.linalg.norm(X, axis=1)
    X_norm_max = np.max(X_norm)
    print("R = max norm X is ", X_norm_max)
    return X_norm_max


def estimate_alpha(X):  # TODO: add function that returns the estimated alpha of the alpha strongly convex loss function
    pass


def estimate_gamma(X):  # TODO: add a function which runs SVM on the dataset to estimate the mergin gamma, and then
    pass                #  use JL lemma from Theorem 6.4


class GradientDescentOptimizer:
    def __init__(self, X, t, w_init, learning_rate, threshold, X_test=None, t_test=None, learning_rate_decay=0,
                 X_validation=None, t_validation=None):  # TODO: try adding validation set to the training process
        self.X = X
        self.t = t
        self.w = w_init
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.X_test = X_test
        self.t_test = t_test
        self.learning_rate_decay = learning_rate_decay

    def optimize(self):
        tt0 = tic()  # evaluate runtime
        # TODO: add running time evaluation of the optimize function for each optimizer
        _step_num = 0
        _train_loss_list = np.array(())
        _test_loss_list = np.array(())
        true_grad = np.ones(self.w.shape)

        # recent_mean_loss_diff = 100  # set initial accuracy difference to 100
        # gradient descent
        # TODO: try to set the stopping criteria to ||gradE||^2 < threshold
        # while np.abs(recent_mean_loss_diff) >= threshold:
        # while np.linalg.norm(true_grad) >= self.threshold:

        _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))
        if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

        # stopping_condition_var = 100
        # while stopping_condition_var >= self.threshold:
        for i in range(NUMBER_OF_ITERATIONS):
            _step_num += 1

            # update learning rate if learning rate decay is not 0
            if self.learning_rate_decay == 1:  # 1/sqrt(t) decay
                self.learning_rate = self.init_learning_rate / np.sqrt(_step_num)
            if self.learning_rate_decay == 2:  # 1/t decay
                self.learning_rate = self.init_learning_rate / (_step_num + 1)
            if self.learning_rate_decay != 0 :
                if PRINT_LEARNING_RATE:
                    print(self.learning_rate)

            # compute new loss
            _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))

            # update weights
            grad_E = self.compute_gradient()
            self.w = step(self.w, grad_E, self.learning_rate)



            # compute test loss
            if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

            if _step_num % 1000 == 0:
                print('*********** GD : STEP NUM EXCEEDS ', _step_num, '!!! ***********')

            if _step_num > 10000:  # prevent infinite loop
                print("*** STEP NUM EXCEEDS 10000 !!! - STOPPING ***")
                break

            if _step_num > 3:
                # compute mean of accuracy difference for the last 3 steps
                recent_mean_loss_diff = np.mean(_train_loss_list[-3:] - _train_loss_list[-4:-1])
                # TODO: try taking the mean diff over the last k steps (k != 3)

            # compute true grad used only for the stopping condition
            """
            this line is added only for comparison between the running times of each optimizer - because the
            reg_GD and SGD optimizers need to compute also the true grad for the stopping condition, this extra
            computation need to be even out across all optimizers to perform a fair comparison  
            """
            if COMPARE_RUNTIMES:
                true_grad = compute_gradient(self.X, self.t, self.w)
            else:
                true_grad = grad_E

            if PRINT_TRUE_GRAD:
                print(np.linalg.norm(true_grad))

            if ZERO_ORDER_STOPPING_CONDITION is True:
                stopping_condition_var = np.abs(recent_mean_loss_diff)
            else: # first order stopping condition - default
                stopping_condition_var = np.linalg.norm(true_grad)



        # compute final accuracy
        # train_acc = compute_accuracy(X_train, t_train, w)
        # test_acc = compute_accuracy(X_test, t_test, w)
        # validation_acc = val_accuracy_list[-1]
        runtime = toc(tt0)


        if self.X_test is not None and self.t_test is not None:
            return self.w, _train_loss_list, _step_num, runtime, _test_loss_list

        return self.w, _train_loss_list, _step_num, runtime

    def compute_gradient(self):  # TODO: try to change to square loss since it's hessian is easier to obtain
        """
            Gradient of log loss function at the current w
        """
        A = np.dot(self.X, self.w)
        m = self.t.shape[0]
        C = -1 * self.t * (1 / (1 + np.exp(A * self.t)))
        return (1 / m) * np.dot(self.X.T, C)


class ConstrainedGradientDescentOptimizer:
    def __init__(self, X, t, w_init, learning_rate, threshold, B, X_test=None, t_test=None, learning_rate_decay=0,
                 X_validation=None, t_validation=None):
        self.X = X
        self.t = t
        self.w = w_init
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.B = B  # the upper bound on norm of w
        self.X_test = X_test
        self.t_test = t_test
        self.learning_rate_decay = learning_rate_decay

    def optimize(self):
        tt0 = tic()  # evaluate runtime
        _step_num = 0
        _train_loss_list = np.array(())
        _test_loss_list = np.array(())
        w_bar = np.zeros(self.w.shape)
        true_grad = np.ones(self.w.shape)
        recent_mean_loss_diff = 100  # set initial accuracy difference to 100

        # gradient descent
        # TODO: try to set the stopping criteria to ||gradE||^2 < threshold
        # while np.abs(recent_mean_loss_diff) >= threshold:
        # while np.linalg.norm(true_grad) >= self.threshold:
        _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))
        if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

        # stopping_condition_var = 100
        # while stopping_condition_var >= self.threshold:
        for i in range(NUMBER_OF_ITERATIONS):
            _step_num += 1

            # update weights
            grad_E = self.compute_gradient()
            self.w = step(self.w, grad_E, self.learning_rate)

            # update learning rate if learning rate decay is not 0
            if self.learning_rate_decay == 1:  # 1/sqrt(t) decay
                self.learning_rate = self.init_learning_rate / np.sqrt(_step_num)
            if self.learning_rate_decay == 2:  # 1/t decay
                self.learning_rate = self.init_learning_rate / (_step_num + 1)
            if self.learning_rate_decay != 0:
                if PRINT_LEARNING_RATE:
                    print(self.learning_rate)

            # project w to ||w||<B
            if np.linalg.norm(self.w) > self.B:
                self.w = self.w * (self.B / np.linalg.norm(self.w))

            # compute new loss
            _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))

            w_bar += self.w

            # compute test loss
            if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

            if _step_num % 1000 == 0:
                print('*********** Cons_GD : STEP NUM EXCEEDS ', _step_num, '!!! ***********')

            if _step_num > 10000:  # prevent infinite loop
                print("*** STEP NUM EXCEEDS 10000 !!! - STOPPING ***")
                break

            if _step_num > 3:
                # compute mean of accuracy difference for the last 3 steps
                recent_mean_loss_diff = np.mean(_train_loss_list[-3:] - _train_loss_list[-4:-1])

            # compute true grad used only for the stopping condition
            """
            this line is added only for comparison between the running times of each optimizer - because the
            reg_GD and SGD optimizers need to compute also the true grad for the stopping condition, this extra
            computation need to be even out across all optimizers to perform a fair comparison  
            """
            if COMPARE_RUNTIMES:
                true_grad = compute_gradient(self.X, self.t, self.w)
            else:
                true_grad = grad_E

            if PRINT_TRUE_GRAD:
                print(np.linalg.norm(true_grad))

            if ZERO_ORDER_STOPPING_CONDITION is True:
                stopping_condition_var = np.abs(recent_mean_loss_diff)
            else: # first order stopping condition - default
                stopping_condition_var = np.linalg.norm(true_grad)


        # compute final accuracy
        # train_acc = compute_accuracy(X_train, t_train, w)
        # test_acc = compute_accuracy(X_test, t_test, w)
        # validation_acc = val_accuracy_list[-1]
        runtime = toc(tt0)
        w_bar = w_bar / _step_num

        if self.X_test is not None and self.t_test is not None:
            return w_bar, self.w, _train_loss_list, _step_num, runtime, _test_loss_list

        return w_bar, self.w, _train_loss_list, _step_num, runtime

    def compute_gradient(self):
        """
            Gradient of log loss function at the current w
        """
        A = np.dot(self.X, self.w)
        m = self.t.shape[0]
        C = -1 * self.t * (1 / (1 + np.exp(A * self.t)))
        return (1 / m) * np.dot(self.X.T, C)


class RegularizedGradientDescentOptimizer:
    def __init__(self, X, t, w_init, learning_rate, threshold, lambda_reg, X_test=None, t_test=None,
                 learning_rate_decay=0, X_validation=None, t_validation=None):
        self.X = X
        self.t = t
        self.w = w_init
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.lambda_reg = lambda_reg  # the upper bound on norm of w
        self.X_test = X_test
        self.t_test = t_test
        self.learning_rate_decay = learning_rate_decay

    def optimize(self):
        tt0 = tic()  # evaluate runtime
        _step_num = 0
        _train_loss_list = np.array(())
        _test_loss_list = np.array(())
        recent_mean_loss_diff = 100  # set initial accuracy difference to 100
        true_grad = np.ones(self.w.shape)

        # gradient descent
        # TODO: try to set the stopping criteria to ||gradE||^2 < threshold
        # while np.abs(recent_mean_loss_diff) >= threshold:
        # while np.linalg.norm(true_grad) >= self.threshold:
        _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))
        if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

        # stopping_condition_var = 100
        # while stopping_condition_var >= self.threshold:
        for i in range(NUMBER_OF_ITERATIONS):
            _step_num += 1

            # update weights
            grad_E_reg = self.compute_reg_gradient()
            self.w = step(self.w, grad_E_reg, self.learning_rate)

            # update learning rate if learning rate decay is not 0
            if self.learning_rate_decay == 1:  # 1/sqrt(t) decay
                self.learning_rate = self.init_learning_rate / np.sqrt(_step_num)
            if self.learning_rate_decay == 2:  # 1/t decay
                self.learning_rate = self.init_learning_rate / (_step_num + 1)
            if self.learning_rate_decay != 0:
                if PRINT_LEARNING_RATE:
                    print(self.learning_rate)

            # compute new loss
            _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))
            # TODO: assert that need to use the original loss func and not the regularized loss

            # compute test loss
            if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

            if _step_num % 1000 == 0:
                print('*********** REG_GD : STEP NUM EXCEEDS ', _step_num, '!!! ***********')

            if _step_num > 10000:  # prevent infinite loop
                print("*** STEP NUM EXCEEDS 10000 !!! - STOPPING ***")
                break

            if _step_num > 3:
                # compute mean of accuracy difference for the last 3 steps
                recent_mean_loss_diff = np.mean(_train_loss_list[-3:] - _train_loss_list[-4:-1])

            # compute true grad used only for the stopping condition
            true_grad = compute_gradient(self.X, self.t, self.w)

            if PRINT_TRUE_GRAD:
                print(np.linalg.norm(true_grad))

            if ZERO_ORDER_STOPPING_CONDITION is True:
                stopping_condition_var = np.abs(recent_mean_loss_diff)
            else: # first order stopping condition - default
                stopping_condition_var = np.linalg.norm(grad_E_reg)  # not the true grad or else does not converge

        # compute final accuracy
        # train_acc = compute_accuracy(X_train, t_train, w)
        # test_acc = compute_accuracy(X_test, t_test, w)
        # validation_acc = val_accuracy_list[-1]
        runtime = toc(tt0)

        if self.X_test is not None and self.t_test is not None:
            return self.w, _train_loss_list, _step_num, runtime, _test_loss_list

        return self.w, _train_loss_list, _step_num, runtime

    def compute_reg_gradient(self):
        """
            Gradient of log loss function with regularization at the current w
        """
        A = np.dot(self.X, self.w)
        m = self.t.shape[0]
        C = -1 * self.t * (1 / (1 + np.exp(A * self.t)))
        return (1 / m) * np.dot(self.X.T, C) + self.lambda_reg * self.w  # add regularization term

    # def compute_regularized_loss(self, X, t, w, lambda_reg):
    #     # TODO: try to change to square loss since it's hessian is easier to obtain
    #     """
    #     Log loss function with L2 regularization
    #
    #     X : matrix with m rows and n columns, where m is the number of samples and n is the dimension of the samples
    #         (the k-th sample is the k-th row of X). X can also be a single sample (when m=1)
    #     t : vector of the labels (size mx1). t can also be a single label (when m=1)
    #     w : vector of the weights (size nx1)
    #     """
    #     A = np.dot(X, w)
    #     m = t.shape[0]
    #     return (1 / m) * np.sum(np.log(1 + np.exp(-1 * A * t))) + 0.5 * lambda_reg * (np.linalg.norm(w))**2


class SGDOptimizer:
    def __init__(self, X, t, w_init, learning_rate, threshold, B, X_test=None, t_test=None, learning_rate_decay=0,
                 X_validation=None, t_validation=None):
        self.X = X
        self.t = t
        self.w = w_init
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.B = B
        self.X_test = X_test
        self.t_test = t_test
        self.learning_rate_decay = learning_rate_decay

    def compute_sgd_gradient(self, x_j, t_j):
        """
            An estimator for the gradient of log loss function at the current w, by taking only a single sample
            x_j : a single sample from X (row vector of size 1Xn)
            t_j : the label of the x_j sample (scalar)
        """
        a = np.dot(x_j.T, self.w)
        return -1 * t_j * (1 / (1 + np.exp(a * t_j))) * x_j

    def optimize(self):
        tt0 = tic() # evaluate runtime

        # reshuffle the samples each run
        _seed = np.random.randint(1, 2 ** 30)
        _random_state = check_random_state(_seed)
        _permutation = _random_state.permutation(self.X.shape[0])
        self.X = self.X[_permutation]
        self.t = self.t[_permutation]


        _step_num = 0
        _train_loss_list = np.array(())
        _test_loss_list = np.array(())
        recent_mean_loss_diff = 100  # set initial accuracy difference to 100
        w_bar = np.zeros(self.w.shape)
        m = self.X.shape[0]
        true_grad = np.ones(self.w.shape)

        # gradient descent
        # TODO: try setting the number of steps to exactly m like in lecture 8 SGD
        # TODO: try to set the stopping criteria to ||gradE||^2 < threshold
        # while np.abs(recent_mean_loss_diff) >= threshold:
        # while np.linalg.norm(true_grad) >= self.threshold:
        _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))
        if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

        # stopping_condition_var = 0 # just for accuracy threshold
        # while stopping_condition_var <= self.threshold:     # just for accuracy threshold

        # stopping_condition_var = 100
        # while stopping_condition_var >= self.threshold:
        for i in range(NUMBER_OF_ITERATIONS):
            _step_num += 1
            j = (_step_num - 1) % m

            # if step_num is a multiple of m shuffle the samples and start over
            if _step_num % m == 0:
                _seed = np.random.randint(1, 2 ** 30)
                _random_state = check_random_state(_seed)
                _permutation = _random_state.permutation(self.X.shape[0])
                self.X = self.X[_permutation]
                self.t = self.t[_permutation]
                j = 0

                # # perform 1-epoch (1 iteration equals to 1 epoch of SGD)
                # for j in range(self.X.shape[0]):
                #     x_j = self.X[j, :]
                #     t_j = self.t[j]
                #     # update weights
                #     grad_E = self.compute_estimated_gradient(x_j, t_j)
                #     self.w = step(self.w, grad_E, self.learning_rate)

            # update learning rate if learning rate decay is not 0
            if self.learning_rate_decay == 1:  # 1/sqrt(t) decay
                self.learning_rate = self.init_learning_rate / np.sqrt(_step_num)
            if self.learning_rate_decay == 2:  # 1/t decay
                self.learning_rate = self.init_learning_rate / (_step_num + 1)
            if self.learning_rate_decay != 0 :
                if PRINT_LEARNING_RATE:
                    print(self.learning_rate)

            x_j = self.X[j, :].reshape(-1,1)
            t_j = self.t[j]
            # update weights
            grad_E_sgd = self.compute_sgd_gradient(x_j, t_j)
            self.w = step(self.w, grad_E_sgd, self.learning_rate)

            # project w to ||w||<B
            if np.linalg.norm(self.w) > self.B:
                self.w = self.w * (self.B / np.linalg.norm(self.w))

            # compute new loss
            _train_loss_list = np.append(_train_loss_list, compute_loss(self.X, self.t, self.w))

            # compute test loss
            if self.X_test is not None and self.t_test is not None:
                _test_loss_list = np.append(_test_loss_list, compute_loss(self.X_test, self.t_test, self.w))

            w_bar += self.w

            # TODO: assert that we need to use the original loss func and not the regularized loss

            if _step_num%1000 == 0:
                print('*********** SGD : STEP NUM EXCEEDS ', _step_num, '!!! ***********')

            if _step_num > 3*m:  # prevent infinite loop
                print("*** STEP NUM EXCEEDS {0} !!! - STOPPING ***".format(3*m))
                break

            if _step_num > 3:
                # compute mean of accuracy difference for the last 3 steps
                recent_mean_loss_diff = np.mean(_train_loss_list[-3:] - _train_loss_list[-4:-1])

            # compute true grad used only for the stopping condition
            true_grad = compute_gradient(self.X, self.t, self.w)
            # acc = compute_accuracy(self.X, self.t, self.w)
            # print(acc)


            if PRINT_TRUE_GRAD:
                print(np.linalg.norm(true_grad))

            if ZERO_ORDER_STOPPING_CONDITION is True:
                stopping_condition_var = np.abs(recent_mean_loss_diff)
            else: # first order stopping condition - default
                stopping_condition_var = np.linalg.norm(true_grad)
                # stopping_condition_var = np.linalg.norm(acc)

        # compute final accuracy
        # train_acc = compute_accuracy(X_train, t_train, w)
        # test_acc = compute_accuracy(X_test, t_test, w)
        # validation_acc = val_accuracy_list[-1]

        w_bar = w_bar / _step_num
        runtime = toc(tt0)

        if self.X_test is not None and self.t_test is not None:
            return w_bar, self.w, _train_loss_list, _step_num, runtime, _test_loss_list

        return w_bar, self.w, _train_loss_list, _step_num, runtime
        # return w_bar, self.w, _train_loss_list, m


#########################################################################
#                Main                                                   #
#########################################################################
if __name__ == "__main__":
    log_to_file = True
    log_file_path = r"C:\Users\yotam\PycharmProjects\ATILT-project"
    COMPARE_RUNTIMES = False  # TODO: to compare runtime need to run SGD for T steps without computing the true grad
                              #  and compare it to running T rounds of the other optimizers
    PRINT_LEARNING_RATE = False
    PRINT_TRUE_GRAD = False
    NUMBER_OF_SGD_RUNS = 50

    #########################################################################
    #                Hyper-Parameters                                       #
    #########################################################################
    # Datasets
    USE_MNIST_01 = False
    USE_MNIST_23 = False
    USE_MNIST_45 = True
    THEORETICAL_HYPER_PARAMS = False
    ZERO_ORDER_STOPPING_CONDITION = False
    W_INIT_NORMALIZE = False
    TRAIN_SET_SIZE = None  # None == All samples
    PLOT_TEST_AND_TRAIN = False
    COMBINED_PLOT = True
    NUMBER_OF_ITERATIONS = 50

    thresholds = np.array([1, 1, 1]) * 0.01   # identical for each optimizer - can vary between data sets
    # thresholds = np.array([1, 1, 1]) * 0.99   # JUST FOR SGD
    thm_B_vals = [30, 30, 30]                 # assumption based on GD results
    empirical_B_vals = thm_B_vals             # property of the data set

    # learning rate may vary per optimizer per data set (total of 4x3 - row order is GD, consGD, regGD, SGD)
    empirical_learning_rates = np.array(([90, 25, 40],    # GD learning rates
                                         [90, 40, 30],       # consGD learning rates
                                         [20, 10, 10],                # regGD learning rates
                                         [40, 50, 60]))               # SGD learning rates


    empirical_lambdas_reg = np.array([1, 1, 1]) * 0.1  # used only by reg_GD optimizer
    empirical_learning_rate_decays = np.array([0, 0, 2, 1])  # learning rate decay : 0 = constant learning rate,
    # 1 = 1/sqrt(t) decay, 2 = 1/(t+1) decay

    if TRAIN_SET_SIZE is not None:
        X_01_train = X_01_train[0:TRAIN_SET_SIZE, :]
        X_23_train = X_23_train[0:TRAIN_SET_SIZE, :]
        X_45_train = X_45_train[0:TRAIN_SET_SIZE, :]
        t_01_train = t_01_train[0:TRAIN_SET_SIZE]
        t_23_train = t_23_train[0:TRAIN_SET_SIZE]
        t_45_train = t_45_train[0:TRAIN_SET_SIZE]

    #########################################################################
    #                Obtaining Theoretical Hyper-Parameters                 #
    #########################################################################
    # compute theoretical hyper-parameters on the full data sets

    delta = 0.01  # error percentage for regGD
    T = 20  # number of iterations for consGD TODO: change this according to results
    X_list = [X_01, X_23, X_45]
    X_names = ['MNIST_01', 'MNIST_23', 'MNIST_45']

    estimated_betas = []
    estimated_R = []
    estimated_L = []
    m_list = []
    thm_lambdas_reg = []
    thm_learning_rates = np.zeros((4, 3))
    thm_learning_rate_decays = np.array([0, 0, 2, 1])
    if THEORETICAL_HYPER_PARAMS:
        for X, B, i, X_name in zip(X_list, thm_B_vals, range(3), X_names):
            # Estimate beta
            beta = estimate_beta(X)
            estimated_betas.append(beta)

            # Estimate R and L and D
            L = max_x_norm(X)  # for log loss L = max norm x
            R = B
            D = 2*B
            # R = max_x_norm(X)
            # estimated_R.append(R)
            # L = R
            # estimated_L.append(L)

            # Estimate lambda
            m = X.shape[0]
            m_list.append(m)

            thm_lambda_reg = np.sqrt((8 * (L ** 2)) / (delta * m * (B ** 2)))
            thm_lambdas_reg.append(thm_lambda_reg)

            # Estimate theoretical learning rates
            thm_lr_gd = 1 / beta  #  TODO: take the thm_lr_gd equal to thm_lr_consgd
            thm_lr_consgd = R / (L * np.sqrt(T))
            thm_lr_reggd = 2 / thm_lambda_reg  # with decay factor = 2
            thm_lr_sgd = D / L  # with decay factor = 1

            thm_learning_rates[:, i] = np.array([thm_lr_gd, thm_lr_consgd, thm_lr_reggd, thm_lr_sgd])

            print("******", X_name, " - Theoretical hyper-parameters ******")
            print('beta = ', beta)
            print('R = ', R)
            print('L = ', L)
            print('m = ', m)
            print('lambda = ', thm_lambda_reg)
            print('thm_lr_gd = ', thm_lr_gd)
            print('thm_lr_consgd = ', thm_lr_consgd)
            print('thm_lr_reggd = ', thm_lr_reggd)
            print('thm_lr_sgd = ', thm_lr_sgd)
            print("\n\n\n")

        print(thm_learning_rates)


    # print(np.mean((X_01), axis=0))
    # print(np.std(X_01, axis=0))
    # print(np.mean((X_23), axis=0))
    # print(np.std(X_23, axis=0))
    # print(np.mean((X_45), axis=0))
    # print(np.std(X_45, axis=0))

    #########################################################################
    #                Set Hyper-Parameters                                   #
    #########################################################################
    # set hyper-parameters
    if THEORETICAL_HYPER_PARAMS:
        learning_rates = thm_learning_rates
        lambdas_reg = thm_lambdas_reg
        learning_rate_decays = thm_learning_rate_decays
        B_vals = thm_B_vals
    else:
        learning_rates = empirical_learning_rates
        lambdas_reg = empirical_lambdas_reg
        learning_rate_decays = empirical_learning_rate_decays
        B_vals = empirical_B_vals

    #########################################################################
    #                Initial Weights                                        #
    #########################################################################
    # create initial random weights

    w_init = w_1  # TODO: try changing w initial value
    # if W_INIT_NORMALIZE:
    #     w_init = w_init / np.linalg.norm(w_init)  # normalize initial weights vector to norm = 1 (in accordance with the
    # standardization of the data)

    #########################################################################
    #                Part A                                                 #
    #########################################################################
    datasets = zip([X_01, X_23, X_45],
                   [t_01, t_23, t_45],
                   ['MNIST_01', 'MNIST_23', 'MNIST_45'],
                   learning_rates.T,  # need to run over the columns of learning_rates in the for loop
                   thresholds,
                   B_vals,
                   lambdas_reg)

    #######################################################################
    #                A.1 - GD                                               #
    #########################################################################
    # optimizer_name = 'GD'
    # for X, t, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45':
    #         continue
    #
    #     # run the GD optimizer only
    #     gd_optimizer = GradientDescentOptimizer(X, t, w_init, learning_rate[0], threshold,
    #                                             learning_rate_decay=learning_rate_decays[0])
    #     w_hat_gd, loss_list_gd, step_num_gd, runtime_gd = gd_optimizer.optimize()
    #     train_acc_gd = compute_accuracy(X, t, w_hat_gd)
    #
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part A.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {10}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n\n\n".format(
    #         step_num_gd, train_acc_gd, learning_rate[0], threshold, dataset_name, np.linalg.norm(w_hat_gd),
    #         optimizer_name, B, figure_filename, runtime_gd, learning_rate_decays[0])
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     ax.plot(np.arange(0, step_num_gd + 1, 1), loss_list_gd, linewidth=2, c='C0')
    #     plt.legend([optimizer_name + ' train loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()

    #########################################################################
    #                A.1 - Constrained GD                                   #
    #########################################################################
    # optimizer_name = 'Constrained GD'
    # for X, t, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45':
    #         continue
    #
    #     # run the Constrained GD optimizer only
    #     cons_gd_optimizer = ConstrainedGradientDescentOptimizer(X, t, w_init, learning_rate[1], threshold, B,
    #                                                             learning_rate_decay=learning_rate_decays[1])
    #     w_hat_consgd, w_final_consgd, loss_list_consgd, step_num_consgd, runtime_consgd = cons_gd_optimizer.optimize()
    #     train_acc_consgd = compute_accuracy(X, t, w_hat_consgd)
    #
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part A.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {10}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n\n\n".format(
    #         step_num_consgd, train_acc_consgd, learning_rate[1], threshold, dataset_name,
    #         np.linalg.norm(w_hat_consgd), optimizer_name, B, figure_filename, runtime_consgd, learning_rate_decays[1])
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\nEmpirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     ax.plot(np.arange(0, step_num_consgd + 1, 1), loss_list_consgd, linewidth=2, c='C0')
    #     plt.legend([optimizer_name + ' train loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()

    #########################################################################
    #                A.1 - Regularized GD                                   #
    #########################################################################
    # optimizer_name = 'Regularized GD'
    # for X, t, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45':
    #         continue
    #
    #     # run the Regularized GD optimizer only
    #     reg_gd_optimizer = RegularizedGradientDescentOptimizer(X, t, w_init, learning_rate[2], threshold, lambda_reg,
    #                                                            learning_rate_decay=learning_rate_decays[2])
    #     w_hat_reggd, loss_list_reggd, step_num_reggd, runtime_reggd = reg_gd_optimizer.optimize()
    #     train_acc_reggd = compute_accuracy(X, t, w_hat_reggd)
    #
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part A.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {11}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "   lambda = {10}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n\n\n".format(
    #         step_num_reggd, train_acc_reggd, learning_rate[2], threshold, dataset_name, np.linalg.norm(w_hat_reggd),
    #         optimizer_name, B, figure_filename, runtime_reggd, lambda_reg, learning_rate_decays[2])
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     ax.plot(np.arange(0, step_num_reggd + 1, 1), loss_list_reggd, linewidth=2, c='C0')
    #     plt.legend([optimizer_name + ' train loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()

    #########################################################################
    #                A.1 - SGD                                              #
    #########################################################################
    # optimizer_name = ' SGD'
    # step_num_list = []
    # accuracy_list = []
    #
    # for X, t, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45':
    #         continue
    #
    #     for run_num in range(NUMBER_OF_SGD_RUNS):
    #         # run the SGD optimizer only
    #         sgd_optimizer = SGDOptimizer(X, t, w_init, learning_rate[3], threshold, B,
    #                                      learning_rate_decay=learning_rate_decays[3])
    #
    #         w_hat_sgd, w_final_sgd, loss_list_sgd, step_num_sgd, runtime_sgd = sgd_optimizer.optimize()
    #         train_acc_sgd = compute_accuracy(X, t, w_final_sgd)
    #
    #         # plot file name
    #         # time.sleep(1)  # make sure filenames are unique
    #         figure_filename = 'ATILT_project_' + get_datetime()
    #
    #         # print log
    #         log_str = "**** Part A.1 - {6} ***\n" \
    #                   "Figure filename: {8}\n" \
    #                   "Dataset: {4}\n" \
    #                   "Optimizer: {6}\n" \
    #                   "Hyper-parameters :\n" \
    #                   "   learning rate = {2}\n" \
    #                   "   learning rate decay = {10}\n" \
    #                   "   threshold = {3}\n" \
    #                   "   B = {7}\n" \
    #                   "Runtime : {9}\n" \
    #                   "L2 norm of the minimizer w_hat = {5}\n" \
    #                   "Final training accuracy = {1}\n" \
    #                   "Number of steps until convergence = {0} steps\n\n\n".format(
    #             step_num_sgd, train_acc_sgd, learning_rate[3], threshold, dataset_name, np.linalg.norm(w_final_sgd),
    #             optimizer_name, B, figure_filename, runtime_sgd, learning_rate_decays[3])
    #
    #         # print to console
    #         print(log_str)
    #
    #         # save results to log file
    #         if log_to_file:
    #             with open(log_file_path + "\log.txt", 'a') as log_file:
    #                 log_file.write(log_str)
    #
    #         # plot the results
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         if THEORETICAL_HYPER_PARAMS:
    #             ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                    '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #         else:
    #             ax.set(title='Part A - ' + optimizer_name + ' Optimizer - ' + dataset_name + '\n  Loss per Iteration' +
    #                    '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #         ax.plot(np.arange(0, step_num_sgd + 1, 1), loss_list_sgd, linewidth=2, c='C0')
    #         plt.legend([optimizer_name + ' train loss'])
    #         fig.tight_layout()
    #         plt.savefig('plots/' + figure_filename + '.png')
    #         # plt.show()
    #         plt.close(fig)
    #
    #         step_num_list.append(step_num_sgd)
    #         accuracy_list.append(train_acc_sgd)
    #
    #         # SGD tests
    #         print('distance between w_bar_sgd and w_hat_sgd = ', np.linalg.norm(w_hat_sgd - w_final_sgd))
    #         print('train_acc_sgd with w_bar = ', compute_accuracy(X, t, w_hat_sgd))
    #         print('train_acc_sgd with w_final = ', compute_accuracy(X, t, w_final_sgd))
    #         print('train_loss_sgd with w_bar = ', compute_loss(X, t, w_hat_sgd))
    #         print('train_loss_sgd with w_final = ', compute_loss(X, t, w_final_sgd))
    #
    #     print("num of iterations list:\n", step_num_list)
    #     print("mean num of iterations = ", np.mean(step_num_list))
    #     print("std of num of iterations = ", np.std(step_num_list))
    #     print("accuracy list:\n", accuracy_list)
    #     print("mean accuracy = ", np.mean(accuracy_list))


    #########################################################################
    #                A.2 - Comparison between all 4 optimizers              #
    #########################################################################
    # for X, t, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45':
    #         continue
    #
    #     # run the 4 different optimizers
    #     # GD
    #     gd_optimizer = GradientDescentOptimizer(X, t, w_init, learning_rate[0], threshold,
    #                                             learning_rate_decay=learning_rate_decays[0])
    #     w_hat_gd, loss_list_gd, step_num_gd, runtime_gd = gd_optimizer.optimize()
    #     train_acc_gd = compute_accuracy(X, t, w_hat_gd)
    #
    #     # Constrained GD
    #     cons_gd_optimizer = ConstrainedGradientDescentOptimizer(X, t, w_init, learning_rate[1], threshold, B,
    #                                                             learning_rate_decay=learning_rate_decays[1])
    #     w_hat_consgd, w_final_consgd, loss_list_consgd, step_num_consgd, runtime_consgd = cons_gd_optimizer.optimize()
    #     train_acc_consgd = compute_accuracy(X, t, w_hat_consgd)
    #
    #     # Regularized GD
    #     reg_gd_optimizer = RegularizedGradientDescentOptimizer(X, t, w_init, learning_rate[2], threshold, lambda_reg,
    #                                                            learning_rate_decay=learning_rate_decays[2])
    #     w_hat_reggd, loss_list_reggd, step_num_reggd, runtime_reggd = reg_gd_optimizer.optimize()
    #     train_acc_reggd = compute_accuracy(X, t, w_hat_reggd)
    #
    #     # SGD
    #     sgd_optimizer = SGDOptimizer(X, t, w_init, learning_rate[3], threshold, B,
    #                                  learning_rate_decay=learning_rate_decays[3])
    #     w_hat_sgd, w_final_sgd, loss_list_sgd, step_num_sgd, runtime_sgd = sgd_optimizer.optimize()
    #     train_acc_sgd = compute_accuracy(X, t, w_hat_sgd)
    #
    #     # aux lists
    #     losses = [loss_list_gd, loss_list_consgd, loss_list_reggd, loss_list_sgd]
    #     step_nums = [step_num_gd, step_num_consgd, step_num_reggd, step_num_sgd]
    #     optimizer_names = ['GD', 'Constrained GD', 'Regularized GD', 'SGD']
    #     train_accuracies = [train_acc_gd, train_acc_consgd, train_acc_reggd, train_acc_sgd]
    #     w_hats = [w_hat_gd, w_hat_consgd, w_hat_reggd, w_hat_sgd]
    #     runtimes = [runtime_gd, runtime_consgd, runtime_reggd, runtime_sgd]
    #
    #     # file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     for step_num, train_acc, w_hat, optimizer_name, runtime, learning_rate_decay in zip(step_nums, train_accuracies,
    #                                                                    w_hats, optimizer_names, runtimes,
    #                                                                    learning_rate_decays):
    #         log_str = "**** Part A - MNIST binary classification results ***\n" \
    #                   "Figure filename: {9}\n" \
    #                   "Dataset: {4}\n" \
    #                   "Optimizer: {6}\n" \
    #                   "Hyper-parameters :\n" \
    #                   "   learning rate = {2}\n" \
    #                   "   learning rate decay = {11}\n" \
    #                   "   threshold = {3}\n" \
    #                   "   B = {7}\n" \
    #                   "   lambda = {8}\n" \
    #                   "Runtime : {10}\n" \
    #                   "L2 norm of the minimizer w_hat = {5}\n" \
    #                   "Final training accuracy = {1}\n" \
    #                   "Number of steps until convergence = {0} steps\n\n\n".format(
    #                     step_num, train_acc, learning_rate, threshold, dataset_name, np.linalg.norm(w_hat),
    #                     optimizer_name, B, lambda_reg, figure_filename, runtime, learning_rate_decay)
    #
    #         # print to console
    #         print(log_str)
    #
    #         # save results to log file
    #         if log_to_file:
    #             with open(log_file_path + "\log.txt", 'a') as log_file:
    #                 log_file.write(log_str)
    #
    #
    #     # TODO: normalize the different w_hats to norm 1 and examine the distance between them (meed to have approx.
    #     #  the same angle)
    #
    #     # plot the optimization error vs iteration for each optimizer
    #     # TODO: set the x-axis of all the plots to the same max value so the
    #     #  difference in convergence time will be visible
    #
    #     # 4 plots with the same x limits for comparison
    #     x_limit = max(step_nums)
    #     fig = plt.figure()
    #     for step_num, loss_list, optimizer_name, plt_color, i in zip(step_nums,
    #                                                                  losses,
    #                                                                  optimizer_names,
    #                                                                  ['C0', 'C2', 'C3', 'C6'],
    #                                                                  range(221, 225)):
    #         # fig = plt.figure()
    #         ax = fig.add_subplot(i)
    #         ax.set(title='Part A - ' + dataset_name + ' - ' + optimizer_name + '\n  Loss per iteration',
    #                xlabel='Iteration', ylabel='E(w)')
    #         ax.plot(np.arange(0, step_num + 1, 1), loss_list, linewidth=2, c=plt_color)
    #         plt.legend([optimizer_name + ' - Train loss'])
    #         ax.set_xlim(1, x_limit)
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '_' + optimizer_name + '.png')
    #     plt.show()
    #
    #     # combined plot of all optimizers
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set(title='Part A - ' + dataset_name + ' - Combined plot of all optimizers' +
    #                  '\nTrain loss per iteration', xlabel='Iteration', ylabel='E(w)')
    #     ax.plot(np.arange(0, step_nums[0] + 1, 1), losses[0], linewidth=2, c='C0')
    #     ax.plot(np.arange(0, step_nums[1] + 1, 1), losses[1], linewidth=2, c='C2')
    #     ax.plot(np.arange(0, step_nums[2] + 1, 1), losses[2], linewidth=2, c='C3')
    #     ax.plot(np.arange(0, step_nums[3] + 1, 1), losses[3], linewidth=2, c='C6')
    #     # ax.set_yscale("log")
    #     # ax.set_xscale("log")
    #     # ax.set_xlim(1, 50)
    #     plt.legend(optimizer_names)
    #     # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '_2.png')
    #     plt.show()

    #########################################################################
    #                Part B                                                 #
    #########################################################################
    # Datasets - train - test split
    datasets = zip([X_01_train, X_23_train, X_45_train],
                   [t_01_train, t_23_train, t_45_train],
                   [X_01_test, X_23_test, X_45_test],
                   [t_01_test, t_23_test, t_45_test],
                   ['MNIST_01_train', 'MNIST_23_train', 'MNIST_45_train'],
                   learning_rates.T,  # need to run over the columns of learning_rates in the for loop
                   thresholds,
                   B_vals,
                   lambdas_reg)

    # TODO: draw a curve of the mean test loss over multiple runs (say 10 or more) where each run the train-test split
    #  is different

    #########################################################################
    #                B.1 - GD                                               #
    #########################################################################
    # optimizer_name = 'GD'
    # for X_train, t_train, X_test, t_test, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01_train':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23_train':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45_train':
    #         continue
    #
    #     # run GD optimizer only
    #     gd_optimizer = GradientDescentOptimizer(X_train, t_train, w_init, learning_rate[0], threshold, X_test, t_test,
    #                                             learning_rate_decay=learning_rate_decays[0])
    #     w_hat_gd, train_loss_list_gd, step_num_gd, runtime_gd, test_loss_list_gd = gd_optimizer.optimize()
    #     train_acc_gd = compute_accuracy(X_train, t_train, w_hat_gd)
    #     test_acc_gd = compute_accuracy(X_test, t_test, w_hat_gd)
    #
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part B.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {11}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n" \
    #               "Train-test accuracy difference = {12}\n" \
    #               "Test loss = {13}\n" \
    #               "Final test accuracy = {10}\n\n\n".format(
    #         step_num_gd, train_acc_gd, learning_rate[0], threshold, dataset_name, np.linalg.norm(w_hat_gd),
    #         optimizer_name, B, figure_filename, runtime_gd, test_acc_gd, learning_rate_decays[0],
    #         train_acc_gd - test_acc_gd, compute_loss(X_test, t_test, w_hat_gd))
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                  '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                  '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                      '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                      '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     if PLOT_TEST_AND_TRAIN:
    #         ax.plot(np.arange(0, step_num_gd + 1, 1), test_loss_list_gd, linewidth=2, c='C3')
    #         ax.plot(np.arange(0, step_num_gd + 1, 1), train_loss_list_gd, linewidth=2, c='C0')
    #         plt.legend([optimizer_name + ' test loss', optimizer_name + ' train loss'])
    #     else:
    #         ax.plot(np.arange(0, step_num_gd + 1, 1), test_loss_list_gd, linewidth=2, c='C3')
    #         plt.legend([optimizer_name + ' test loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()

        #  TODO: add graph of the diff between test error to train error per iteration (generalization error)

    ########################################################################
    #               B.1 - Constrained GD                                   #
    ########################################################################
    # optimizer_name = 'Constrained GD'
    # for X_train, t_train, X_test, t_test, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01_train':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23_train':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45_train':
    #         continue
    #
    #     # run Constrained GD optimizer only
    #     cons_gd_optimizer = ConstrainedGradientDescentOptimizer(X_train, t_train, w_init, learning_rate[1], threshold,
    #                                                             B, X_test, t_test,
    #                                                             learning_rate_decay=learning_rate_decays[1])
    #     w_hat_consgd, w_final_consgd, train_loss_list_consgd, step_num_consgd, \
    #         runtime_consgd, test_loss_list_consgd = cons_gd_optimizer.optimize()
    #     train_acc_consgd = compute_accuracy(X_train, t_train, w_hat_consgd)
    #     test_acc_consgd = compute_accuracy(X_test, t_test, w_hat_consgd)
    #
    #
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part B.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {11}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n" \
    #               "Train-test accuracy difference = {12}\n" \
    #               "Test loss = {13}\n" \
    #               "Final test accuracy = {10}\n\n\n".format(
    #         step_num_consgd, train_acc_consgd, learning_rate[1], threshold, dataset_name,
    #         np.linalg.norm(w_hat_consgd), optimizer_name, B, figure_filename, runtime_consgd,
    #         test_acc_consgd, learning_rate_decays[1], train_acc_consgd - test_acc_consgd, compute_loss(X_test, t_test, w_hat_consgd))
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                      '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                      '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                      '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                      '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     if PLOT_TEST_AND_TRAIN:
    #         ax.plot(np.arange(0, step_num_consgd + 1, 1), test_loss_list_consgd, linewidth=2, c='C3')
    #         ax.plot(np.arange(0, step_num_consgd + 1, 1), train_loss_list_consgd, linewidth=2, c='C0')
    #         plt.legend([optimizer_name + ' test loss', optimizer_name + ' train loss'])
    #     else:
    #         ax.plot(np.arange(0, step_num_consgd + 1, 1), test_loss_list_consgd, linewidth=2, c='C3')
    #         plt.legend([optimizer_name + ' test loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()
    #
    #     # SGD tests
    #     print('distance between w_bar_sgd and w_hat_sgd = ', np.linalg.norm(w_hat_consgd - w_final_consgd))
    #     print('train_loss_sgd with w_bar = ', compute_loss(X_train, t_train, w_hat_consgd))
    #     print('train_loss_sgd with w_final = ', compute_loss(X_train, t_train, w_final_consgd))
    #     print('train_acc_sgd with w_bar = ', train_acc_consgd)
    #     print('train_acc_sgd with w_final = ', compute_accuracy(X_train, t_train, w_final_consgd))
    #     print('test_loss_sgd with w_bar = ', compute_loss(X_test, t_test, w_hat_consgd))
    #     print('test_loss_sgd with w_final = ', compute_loss(X_test, t_test, w_final_consgd))
    #     print('test_acc_sgd with w_bar = ', compute_accuracy(X_test, t_test, w_hat_consgd))
    #     print('test_acc_sgd with w_final = ', compute_accuracy(X_test, t_test, w_final_consgd))
    #     print('\n\n\n')

    ########################################################################
    #              B.1 - Regularized GD                                    #
    ########################################################################
    # optimizer_name = 'Regularized GD'
    # for X_train, t_train, X_test, t_test, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
    #     if not USE_MNIST_01 and dataset_name is 'MNIST_01_train':
    #         continue
    #     if not USE_MNIST_23 and dataset_name is 'MNIST_23_train':
    #         continue
    #     if not USE_MNIST_45 and dataset_name is 'MNIST_45_train':
    #         continue
    #
    #     # run GD optimizer only
    #     reg_gd_optimizer = RegularizedGradientDescentOptimizer(X_train, t_train, w_init, learning_rate[2], threshold,
    #                                                            lambda_reg, X_test, t_test,
    #                                                            learning_rate_decay=learning_rate_decays[2])
    #     w_hat_reggd, train_loss_list_reggd, step_num_reggd, \
    #         runtime_reggd, test_loss_list_reggd = reg_gd_optimizer.optimize()
    #     train_acc_reggd = compute_accuracy(X_train, t_train, w_hat_reggd)
    #     test_acc_reggd = compute_accuracy(X_test, t_test, w_hat_reggd)
    #     # plot file name
    #     figure_filename = 'ATILT_project_' + get_datetime()
    #
    #     # print log
    #     log_str = "**** Part B.1 - {6} ***\n" \
    #               "Figure filename: {8}\n" \
    #               "Dataset: {4}\n" \
    #               "Optimizer: {6}\n" \
    #               "Hyper-parameters :\n" \
    #               "   learning rate = {2}\n" \
    #               "   learning rate decay = {12}\n" \
    #               "   threshold = {3}\n" \
    #               "   B = {7}\n" \
    #               "   lambda = {11}\n" \
    #               "Runtime : {9}\n" \
    #               "L2 norm of the minimizer w_hat = {5}\n" \
    #               "Final training accuracy = {1}\n" \
    #               "Number of steps until convergence = {0} steps\n" \
    #               "Train-test accuracy difference = {13}\n" \
    #               "Test loss = {14}\n" \
    #               "Final test accuracy = {10}\n\n\n".format(
    #         step_num_reggd, train_acc_reggd, learning_rate[2], threshold, dataset_name, np.linalg.norm(w_hat_reggd),
    #         optimizer_name, B, figure_filename, runtime_reggd, test_acc_reggd, lambda_reg, learning_rate_decays[2],
    #         train_acc_reggd - test_acc_reggd, compute_loss(X_test, t_test, w_hat_reggd))
    #
    #     # print to console
    #     print(log_str)
    #
    #     # save results to log file
    #     if log_to_file:
    #         with open(log_file_path + "\log.txt", 'a') as log_file:
    #             log_file.write(log_str)
    #
    #     # plot the results
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     if THEORETICAL_HYPER_PARAMS:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                      '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                      '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     else:
    #         ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
    #                      '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
    #                      '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
    #     if PLOT_TEST_AND_TRAIN:
    #         ax.plot(np.arange(0, step_num_reggd + 1, 1), test_loss_list_reggd, linewidth=2, c='C3')
    #         ax.plot(np.arange(0, step_num_reggd + 1, 1), train_loss_list_reggd, linewidth=2, c='C0')
    #         plt.legend([optimizer_name + ' test loss', optimizer_name + ' train loss'])
    #     else:
    #         ax.plot(np.arange(0, step_num_reggd + 1, 1), test_loss_list_reggd, linewidth=2, c='C3')
    #         plt.legend([optimizer_name + ' test loss'])
    #     fig.tight_layout()
    #     plt.savefig('plots/' + figure_filename + '.png')
    #     plt.show()

    ########################################################################
    #               B.1 - SGD                                              #
    ########################################################################
    optimizer_name = ' SGD'
    step_num_list = []
    accuracy_train_list = []
    accuracy_test_list = []
    final_test_loss_list = []
    L2_norm_list = []

    for X_train, t_train, X_test, t_test, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
        if not USE_MNIST_01 and dataset_name is 'MNIST_01_train':
            continue
        if not USE_MNIST_23 and dataset_name is 'MNIST_23_train':
            continue
        if not USE_MNIST_45 and dataset_name is 'MNIST_45_train':
            continue

        for run_num in range(NUMBER_OF_SGD_RUNS):
            # run Constrained SGD optimizer only
            sgd_optimizer = SGDOptimizer(X_train, t_train, w_init, learning_rate[3], threshold, B, X_test, t_test,
                                         learning_rate_decay=learning_rate_decays[3])

            w_hat_sgd, w_final_sgd, train_loss_list_sgd, step_num_sgd, \
                runtime_sgd, test_loss_list_sgd = sgd_optimizer.optimize()
            train_acc_sgd = compute_accuracy(X_train, t_train, w_final_sgd)
            test_acc_sgd = compute_accuracy(X_test, t_test, w_final_sgd)

            # plot file name
            time.sleep(1)  # make sure filenames are unique
            figure_filename = 'ATILT_project_' + get_datetime()

            # print log
            log_str = "**** Part B.1 - {6} ***\n" \
                      "Figure filename: {8}\n" \
                      "Dataset: {4}\n" \
                      "Optimizer: {6}\n" \
                      "Hyper-parameters :\n" \
                      "   learning rate = {2}\n" \
                      "   learning rate decay = {11}\n" \
                      "   threshold = {3}\n" \
                      "   B = {7}\n" \
                      "Runtime : {9}\n" \
                      "L2 norm of the minimizer w_hat = {5}\n" \
                      "Final training accuracy = {1}\n" \
                      "Number of steps until convergence = {0} steps\n" \
                      "Train-test accuracy difference = {12}\n" \
                      "Test loss = {13}\n" \
                      "Final test accuracy = {10}\n\n\n".format(
                step_num_sgd, train_acc_sgd, learning_rate[3], threshold, dataset_name, np.linalg.norm(w_final_sgd),
                optimizer_name, B, figure_filename, runtime_sgd, test_acc_sgd, learning_rate_decays[3],
                train_acc_sgd - test_acc_sgd, compute_loss(X_test, t_test, w_final_sgd))

            # print to console
            print(log_str)

            # save results to log file
            if log_to_file:
                with open(log_file_path + "\log.txt", 'a') as log_file:
                    log_file.write(log_str)

            # plot the results
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if THEORETICAL_HYPER_PARAMS:
                ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
                             '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
                             '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
            else:
                ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
                             '\n  Loss per Iteration - m = ' + str(X_train.shape[0]) +
                             '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
            if PLOT_TEST_AND_TRAIN:
                ax.plot(np.arange(0, step_num_sgd + 1, 1), test_loss_list_sgd, linewidth=2, c='C3')
                ax.plot(np.arange(0, step_num_sgd + 1, 1), train_loss_list_sgd, linewidth=2, c='C0')
                plt.legend([optimizer_name + ' test loss', optimizer_name + ' train loss'])
            else:
                ax.plot(np.arange(0, step_num_sgd + 1, 1), test_loss_list_sgd, linewidth=2, c='C3')
                plt.legend([optimizer_name + ' test loss'])
            fig.tight_layout()
            plt.savefig('plots/' + figure_filename + '.png')
            # plt.show()
            plt.close(fig)

            step_num_list.append(step_num_sgd)
            accuracy_train_list.append(train_acc_sgd)
            accuracy_test_list.append(test_acc_sgd)
            final_test_loss_list.append(test_loss_list_sgd[-1])
            L2_norm_list.append(np.linalg.norm(w_final_sgd))

            # SGD tests
            print('distance between w_bar_sgd and w_hat_sgd = ', np.linalg.norm(w_hat_sgd - w_final_sgd))
            print('train_loss_sgd with w_bar = ', compute_loss(X_train, t_train, w_hat_sgd))
            print('train_loss_sgd with w_final = ', compute_loss(X_train, t_train, w_final_sgd))
            print('train_acc_sgd with w_bar = ', compute_accuracy(X_train, t_train, w_hat_sgd))
            print('train_acc_sgd with w_final = ', compute_accuracy(X_train, t_train, w_final_sgd))
            print('test_loss_sgd with w_bar = ', compute_loss(X_test, t_test, w_hat_sgd))
            print('test_loss_sgd with w_final = ', compute_loss(X_test, t_test, w_final_sgd))
            print('test_acc_sgd with w_bar = ', compute_accuracy(X_test, t_test, w_hat_sgd))
            print('test_acc_sgd with w_final = ', compute_accuracy(X_test, t_test, w_final_sgd))
            print('\n\n\n')

        print("num of iterations list:\n", step_num_list)
        print("mean num of iterations = ", np.mean(step_num_list))
        print("std of num of iterations = ", np.std(step_num_list))
        print("train accuracy list:\n", accuracy_train_list)
        print("train mean accuracy = ", np.mean(accuracy_train_list))
        print("test accuracy list:\n", accuracy_test_list)
        print("test mean accuracy = ", np.mean(accuracy_test_list))
        print("train - test accuracy diff = ", np.mean(accuracy_train_list) - np.mean(accuracy_test_list))
        print("mean test loss = ", np.mean(final_test_loss_list))
        print("mean L2 norm of minimizer = ", np.mean(L2_norm_list))

    ########################################################################
    #             B.2 - Comparison between all 4 optimizers                #
    ########################################################################
    for X_train, t_train, X_test, t_test, dataset_name, learning_rate, threshold, B, lambda_reg in datasets:
        if not USE_MNIST_01 and dataset_name is 'MNIST_01_train':
            continue
        if not USE_MNIST_23 and dataset_name is 'MNIST_23_train':
            continue
        if not USE_MNIST_45 and dataset_name is 'MNIST_45_train':
            continue

        # run the 4 different optimizers - with train and test data
        # GD
        gd_optimizer = GradientDescentOptimizer(X_train, t_train, w_init, learning_rate[0], threshold, X_test, t_test,
                                                learning_rate_decay=learning_rate_decays[0])
        w_hat_gd, train_loss_list_gd, step_num_gd, runtime_gd, test_loss_list_gd  = gd_optimizer.optimize()
        train_acc_gd = compute_accuracy(X_train, t_train, w_hat_gd)
        test_acc_gd = compute_accuracy(X_test, t_test, w_hat_gd)

        # Constrained GD
        cons_gd_optimizer = ConstrainedGradientDescentOptimizer(X_train, t_train, w_init, learning_rate[1], threshold,
                                                                B, X_test, t_test,
                                                                learning_rate_decay=learning_rate_decays[1])
        w_hat_consgd, w_final_consgd, train_loss_list_consgd, step_num_consgd, \
            runtime_consgd, test_loss_list_consgd = cons_gd_optimizer.optimize()
        train_acc_consgd = compute_accuracy(X_train, t_train, w_hat_consgd)
        test_acc_consgd = compute_accuracy(X_test, t_test, w_hat_consgd)

        # Regularized GD
        reg_gd_optimizer = RegularizedGradientDescentOptimizer(X_train, t_train, w_init, learning_rate[2], threshold,
                                                               lambda_reg, X_test, t_test,
                                                               learning_rate_decay=learning_rate_decays[2])
        w_hat_reggd, train_loss_list_reggd, step_num_reggd, \
            runtime_reggd, test_loss_list_reggd = reg_gd_optimizer.optimize()
        train_acc_reggd = compute_accuracy(X_train, t_train, w_hat_reggd)
        test_acc_reggd = compute_accuracy(X_test, t_test, w_hat_reggd)

        # SGD
        sgd_optimizer = SGDOptimizer(X_train, t_train, w_init, learning_rate[3], threshold, B, X_test, t_test,
                                     learning_rate_decay=learning_rate_decays[3])
        w_hat_sgd, w_final_sgd, train_loss_list_sgd, step_num_sgd, \
            runtime_sgd, test_loss_list_sgd = sgd_optimizer.optimize()
        train_acc_sgd = compute_accuracy(X_train, t_train, w_final_sgd)
        test_acc_sgd = compute_accuracy(X_test, t_test, w_final_sgd)

        # aux lists
        train_losses = [train_loss_list_gd, train_loss_list_consgd, train_loss_list_reggd, train_loss_list_sgd]
        test_losses = [test_loss_list_gd, test_loss_list_consgd, test_loss_list_reggd, test_loss_list_sgd]
        step_nums = [step_num_gd, step_num_consgd, step_num_reggd, step_num_sgd]
        optimizer_names = ['GD', 'Constrained GD', 'Regularized GD', 'SGD']
        train_accuracies = [train_acc_gd, train_acc_consgd, train_acc_reggd, train_acc_sgd]
        test_accuracies = [test_acc_gd, test_acc_consgd, test_acc_reggd, test_acc_sgd]
        w_hats = [w_hat_gd, w_hat_consgd, w_hat_reggd, w_hat_sgd]
        runtimes = [runtime_gd, runtime_consgd, runtime_reggd, runtime_sgd]

        # file name
        figure_filename = 'ATILT_project_' + get_datetime()

        # print log
        for step_num, train_acc, test_acc, w_hat, optimizer_name,\
                runtime, learning_rate_decay, test_loss in zip(step_nums,
                                                    train_accuracies,
                                                    test_accuracies,
                                                    w_hats,
                                                    optimizer_names,
                                                    runtimes,
                                                    learning_rate_decays, test_losses):
            log_str = "**** Part B - MNIST binary classification results ***\n" \
                      "Figure filename: {9}\n" \
                      "Dataset: {4}\n" \
                      "Optimizer: {6}\n" \
                      "Hyper-parameters :\n" \
                      "   learning rate = {2}\n" \
                      "   learning rate decay = {12}\n" \
                      "   threshold = {3}\n" \
                      "   B = {7}\n" \
                      "   lambda = {8}\n" \
                      "Runtime : {10}\n" \
                      "L2 norm of the minimizer w_hat = {5}\n" \
                      "Final training accuracy = {1}\n" \
                      "Number of steps until convergence = {0} steps\n" \
                      "Test loss = {13}\n" \
                      "Final test accuracy = {11}\n\n\n".format(
                        step_num, train_acc, learning_rate, threshold, dataset_name, np.linalg.norm(w_hat),
                        optimizer_name, B, lambda_reg, figure_filename, runtime, test_acc, learning_rate_decay,
                        test_loss[-1])

            # print to console
            print(log_str)

            # save results to log file
            if log_to_file:
                with open(log_file_path + "\log.txt", 'a') as log_file:
                    log_file.write(log_str)

        # TODO: normalize the different w_hats to norm 1 and examine the distance between them (meed to have approx.
        #  the same angle)
        # plot the optimization error vs iteration for each optimizer
        # TODO: set the x-axis of all the plots to the same max value so the
        #  difference in convergence time will be visible

        # 4 plots with the same x limits for comparison
        # x_limit = max(step_nums)
        # fig = plt.figure()
        # if COMBINED_PLOT:
        #     fig = plt.figure()
        #     for step_num, test_loss, optimizer_name, plt_color, i in zip(step_nums,
        #                                                                  test_losses,
        #                                                                  optimizer_names,
        #                                                                  ['C0', 'C2', 'C3', 'C6'],
        #                                                                  range(221, 225)):
        #         # fig = plt.figure()
        #         ax = fig.add_subplot(i)
        #         if THEORETICAL_HYPER_PARAMS:
        #             ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
        #                          '\n  Test loss per Iteration - m = ' + str(X_train.shape[0]) +
        #                          '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        #         else:
        #             ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
        #                          '\n  Test loss per Iteration - m = ' + str(X_train.shape[0]) +
        #                          '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        #         # ax.set(title='Part B - ' + dataset_name + ' - ' + optimizer_name +
        #         #              '\n  Test loss per iteration', xlabel='Iteration', ylabel='E(w)')
        #         ax.plot(np.arange(0, step_num + 1, 1), test_loss, linewidth=2, c=plt_color)
        #         plt.legend([optimizer_name + ' - Test loss'])
        #         # ax.set_xlim(1, x_limit)
        #     fig.tight_layout()
        #     plt.savefig('plots/' + figure_filename + '_' + optimizer_name + '.png')
        #     plt.show()
        # else:
        #     for step_num, test_loss, optimizer_name in zip(step_nums,
        #                                                               test_losses,
        #                                                               optimizer_names):
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)
        #         if THEORETICAL_HYPER_PARAMS:
        #             ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
        #                              '\n  Test loss per Iteration - m = ' + str(X_train.shape[0]) +
        #                              '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        #         else:
        #             ax.set(title='Part B - ' + optimizer_name + ' Optimizer - ' + dataset_name +
        #                          '\n  Test loss per Iteration - m = ' + str(X_train.shape[0]) +
        #                          '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        #         # ax.set(title='Part B - ' + dataset_name + ' - ' + optimizer_name +
        #         #              '\n  Test loss per iteration', xlabel='Iteration', ylabel='E(w)')
        #         ax.plot(np.arange(0, step_num + 1, 1), test_loss, linewidth=2, c='C3')
        #         plt.legend([optimizer_name + ' - Test loss'])
        #         # ax.set_xlim(1, x_limit)
        #         fig.tight_layout()
        #         plt.savefig('plots/' + figure_filename + '_' + optimizer_name + '.png')
        #         plt.show()


        # combined plot of all optimizers
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if THEORETICAL_HYPER_PARAMS:
            ax.set(title='Part B - ' + dataset_name + ' - Combined plot of all optimizers' +
                     '\n  Test loss per iteration, m =  ' + str(X_train.shape[0]) + '\n Theoretical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        else:
            ax.set(title='Part B - ' + dataset_name + ' - Combined plot of all optimizers' +
                     '\n  Test loss per iteration,  m =  ' + str(X_train.shape[0]) + '\n Empirical Hyper-Parameters', xlabel='Iteration', ylabel='E(w)')
        ax.plot(np.arange(0, step_nums[0]+1, 1), test_losses[0], linewidth=2, c='C0')
        ax.plot(np.arange(0, step_nums[1]+1, 1), test_losses[1], linewidth=2, c='C2')
        ax.plot(np.arange(0, step_nums[2]+1, 1), test_losses[2], linewidth=2, c='C3')
        ax.plot(np.arange(0, step_nums[3]+1, 1), test_losses[3], linewidth=2, c='C6')
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.set_xlim(1, 50)
        plt.legend(optimizer_names)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        fig.tight_layout()
        plt.savefig('plots/' + figure_filename + '_2.png')
        plt.show()
