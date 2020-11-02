from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import fun
from keras import layers
from keras import models
from keras import optimizers
import pickle

path = '.../R0/'

#folder = '1802201656'
folder = '30Jan20'
#folder = '1502201536'

f = open(path+'dict_'+folder+'_.pkl','rb')
D = pickle.load(f)
f.close()

H = fun.distance_grid_loc(18, 18, True)
phi = 3 / 5

index_test_s1 = fun.create_index(m=496, samples=10, k=100)
index_test_s2 = fun.create_index(m=504, samples=10, k=100)

accuracy_array = np.zeros(shape=(45, 10))
accuracy_array[0:5, 0:10] = np.repeat(0.5, 50).reshape(5,10)

voi_array = np.zeros(shape=(45,10))
voi_array[0:5, 0:10] = np.repeat(0, 50).reshape(5,10)

gp_array = np.zeros(shape=(1600,1))
probas_gp_array = np.zeros(shape=(1600,10))
knn_array = np.zeros(shape=(1600,1))
probas_knn_array = np.zeros(shape=(1600,10))
rf_array = np.zeros(shape=(1600,1))
probas_rf_array = np.zeros(shape=(1600,10))
mlp_array = np.zeros(shape=(1600,1))
probas_mlp_array = np.zeros(shape=(1600,10))
cnn_array = np.zeros(shape=(1600,1))
probas_cnn_array = np.zeros(shape=(1600,10))

times = [2, 5, 8, 11, 14, 17, 20, 23]

for j in range(10):
    accuracy_dict = {}
    voi_dict = {}

    for t in range(len(D)):
        l_scen1_train, l_scen1_test = fun.split_by_index(D['t' + str(t)]['s1'], index_test=index_test_s1[j])
        l_scen2_train, l_scen2_test = fun.split_by_index(D['t' + str(t)]['s2'], index_test=index_test_s2[j])
        m_scen1_train, ld_scen1, il_scen1 = fun.arguments_gp_R0(H, phi, l_scen1_train)
        m_scen2_train, ld_scen2, il_scen2 = fun.arguments_gp_R0(H, phi, l_scen2_train)

        X_train = fun.ltodf(l_scen1_train + l_scen2_train)
        X_test = fun.ltodf(l_scen1_test + l_scen2_test)
        y_train = np.concatenate((np.repeat(0, len(l_scen1_train)), np.repeat(1, len(l_scen2_train))), axis=None)
        y_test = np.concatenate((np.repeat(0, len(l_scen1_test)), np.repeat(1, len(l_scen2_test))), axis=None)

        list_probs_gp = []
        pred_gp = np.zeros(shape=(200,))
        for i in range(200):
            list_probs_gp.append(fun.probabilities_gp([fun.LogCP2(X_test.iloc[i].values, m_scen1_train, ld_scen1, il_scen1, 0.5).value(),
                                                       fun.LogCP2(X_test.iloc[i].values, m_scen2_train, ld_scen2, il_scen2, 0.5).value()]))
            pred_gp[i] = fun.classif(list_probs_gp[-1])
        proba_gp = np.asarray(list_probs_gp)

        classifier_knn = KNeighborsClassifier(n_neighbors=10)
        classifier_knn.fit(X_train, y_train)
        proba_knn = classifier_knn.predict_proba(X_test)
        pred_knn = classifier_knn.predict(X_test)

        classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        classifier_rf.fit(X_train, y_train)
        proba_rf = classifier_rf.predict_proba(X_test)
        pred_rf = classifier_rf.predict(X_test)

        classifier_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)
        classifier_mlp.fit(X_train, y_train)
        proba_mlp = classifier_mlp.predict_proba(X_test)
        pred_mlp = classifier_mlp.predict(X_test)

        # CNN: keras
        train_images = X_train.values.reshape((800, 18, 18, 1))
        test_images = X_test.values.reshape((200, 18, 18, 1))
        train_labels = y_train
        test_labels = y_test

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(18, 18, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
        model.fit(train_images, train_labels, epochs=60, batch_size=16)
        proba_cnn = model.predict(test_images)
        pred_cnn = model.predict_classes(test_images).reshape(200)

        gp_array[t*200:(t+1)*200] = proba_gp[...,1].reshape(200,1)
        knn_array[t*200:(t+1)*200] = proba_knn[...,1].reshape(200,1)
        rf_array[t*200:(t+1)*200] = proba_rf[...,1].reshape(200,1)
        mlp_array[t*200:(t+1)*200] = proba_mlp[...,1].reshape(200,1)
        cnn_array[t*200:(t+1)*200] = proba_cnn

        values = {'v(x=0,a=0)': -25 + 0.6*times[t], 'v(x=1,a=0)': -27 - 0.6*times[t], 'v(x=0,a=1)': -10, 'v(x=1,a=1)': -42}

        voi_dict['t' + str(t) + '_voi_' + 'GP'] = fun.voi_mc(proba_gp, values)
        voi_dict['t' + str(t) + '_voi_' + 'knn'] = fun.voi_mc(proba_knn, values)
        voi_dict['t' + str(t) + '_voi_' + 'rf'] = fun.voi_mc(proba_rf, values)
        voi_dict['t' + str(t) + '_voi_' + 'mlp'] = fun.voi_mc(proba_mlp, values)
        voi_dict['t' + str(t) + '_voi_' + 'cnn'] = fun.voi_mc(np.concatenate((np.array(1-proba_cnn), proba_cnn), axis=1), values)

        accuracy_dict['t' + str(t) + '_accuracy_' + 'GP'] = accuracy_score(y_test, pred_gp)
        accuracy_dict['t' + str(t) + '_accuracy_' + 'knn'] = accuracy_score(y_test, pred_knn)
        accuracy_dict['t' + str(t) + '_accuracy_' + 'rf'] = accuracy_score(y_test, pred_rf)
        accuracy_dict['t' + str(t) + '_accuracy_' + 'mlp'] = accuracy_score(y_test, pred_mlp)
        accuracy_dict['t' + str(t) + '_accuracy_' + 'cnn'] = accuracy_score(y_test, pred_cnn)

    probas_gp_array[..., j] = gp_array.reshape(1600)
    probas_knn_array[..., j] = knn_array.reshape(1600)
    probas_rf_array[..., j] = rf_array.reshape(1600)
    probas_mlp_array[..., j] = mlp_array.reshape(1600)
    probas_cnn_array[..., j] = cnn_array.reshape(1600)

    voi_array[5:, j]=np.asarray([x for x in voi_dict.values()]).reshape(40,)
    accuracy_array[5:, j]=np.asarray([x for x in accuracy_dict.values()]).reshape(40,)