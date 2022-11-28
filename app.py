from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from pickle import dump
from pickle import load
from sklearn.metrics import r2_score
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/dataset/", methods=["GET","POST"])
def dataset():
    if request.method == "POST":
        global Loan
        cek = 1
        link = request.form['link']
        # Membaca dataset
        Loan = pd.read_csv(link)
        return render_template("dataset.html", cek=cek, a=link, b=Loan.shape, 
            dataset_Loan=[Loan.to_html(justify='center',index=False)], 
            dataset_Loan_isi=[''], coba=Loan.to_numpy())
    else:
        cek=0
        link=""
        tampilan=""
        return render_template("dataset.html", a=link, b=tampilan, cek=cek)

@app.route("/preprocessing/", methods=["GET","POST"])
def preprocessing():
    if request.method == "POST":
        global normalisasi_request, label_encoder
        cek = 1
        transformasi = request.form['transformasi']
        normalisasi_request = request.form['normalisasi']
        # Menghapus missing value
        Loan2 = Loan.dropna(axis=0)
        # Transformasi data dengan Label Encoder
        label_encoder = LabelEncoder()
        Loan2["Gender"] = label_encoder.fit_transform(Loan2[["Gender"]])
        Loan2["Married"] = label_encoder.fit_transform(Loan2[["Married"]])
        Loan2["Education"] = label_encoder.fit_transform(Loan2[["Education"]])
        Loan2["Self_Employed"] = label_encoder.fit_transform(Loan2[["Self_Employed"]])
        Loan2["Property_Area"] = label_encoder.fit_transform(Loan2[["Property_Area"]])
        Loan2["Loan_Status"] = label_encoder.fit_transform(Loan2[["Loan_Status"]])
        data_label_encoder = Loan2.head()
        # Mengambil kolom selain kode_kontrak dan risk_rating dan melakukan normalisasi data
        normalisasi = Loan2.drop(["Loan_ID", "Loan_Status"], axis=1)
        if normalisasi_request == '1':
            global Loan_Zscore
            # Melakukan normalisasi dengan Z Score atau StandardScaler
            scaler = StandardScaler()
            scaler.fit(normalisasi)
            scale_data = (scaler.transform(normalisasi))
            # Menampilkan data normalisasi dari Z score
            dataZScale = pd.DataFrame(scale_data, columns=normalisasi.columns.values)
            dataZScale1 = dataZScale.head()
            # Proses mengubah dataframe menjadi array
            data_Loan_Status_array = Loan2["Loan_Status"].to_numpy()
            # Mengubah array menjadi dataframe perkolom kembali
            data_Loan_Status = pd.DataFrame(data_Loan_Status_array, columns = ["Loan_Status"])
            # Menggabungkan kolom yang sudah dinormalisasi Min Max dan data Loan_Status
            Loan_Zscore = pd.concat([dataZScale, data_Loan_Status], axis=1)
            Loan_Zscore1 = Loan_Zscore.head()
            # save the scaler
            dump(scaler, open('scaler_ZScore.pkl', 'wb'))
            if transformasi == 'no':
                cek2 = 0
                pesan = "Anda harus mengaktifkan Label Encoder agar bisa menampilkan hasil"
                return render_template("preprocesing.html", cek=cek, a=transformasi,  b=normalisasi_request, pesan=pesan, cek2=cek2)
            else:
                return render_template("preprocesing.html", cek=cek, a=transformasi,  b=normalisasi_request, 
                                data_label_encoder=[data_label_encoder.to_html(justify='center',index=False)], data_label_encoder_isi=[''],
                                dataZScale1=[dataZScale1.to_html(justify='center',index=False)], dataZScale1_isi=[''],
                                Loan_Zscore1=[Loan_Zscore1.to_html(justify='center',index=False)], Loan_Zscore1_isi=[''])
        else:
            global Loan_min_max
            # melakukan skala fitur
            scaler = MinMaxScaler()
            model = scaler.fit(normalisasi)
            scaled_data=model.transform(normalisasi)
            # menampilkan data normalisasi dari Min Max
            namakolom = normalisasi.columns.values
            dataMinMax = pd.DataFrame(scaled_data, columns=namakolom)
            dataMinMax1 = dataMinMax.head()
            # Proses mengubah dataframe menjadi array
            data_Loan_Status_array = Loan2["Loan_Status"].to_numpy()
            # Mengubah array menjadi dataframe perkolom kembali
            data_Loan_Status = pd.DataFrame(data_Loan_Status_array, columns = ["Loan_Status"])
            # Menggabungkan kolom yang sudah dinormalisasi Min Max dan data Loan_Status
            Loan_min_max = pd.concat([dataMinMax, data_Loan_Status], axis=1)
            Loan_min_max1 = Loan_min_max.head()
            # save the scaler
            dump(scaler, open('scaler_MinMax.pkl', 'wb'))
            if transformasi == 'no':
                cek2 = 0
                pesan = "Anda harus mengaktifkan Label Encoder agar bisa menampilkan hasil"
                return render_template("preprocesing.html", cek=cek, a=transformasi, b=normalisasi_request, pesan=pesan, cek2=cek2)
            else:
                return render_template("preprocesing.html", cek=cek, a=transformasi, b=normalisasi_request,
                                data_label_encoder=[data_label_encoder.to_html(justify='center',index=False)], data_label_encoder_isi=[''],
                                dataMinMax1=[dataMinMax1.to_html(justify='center',index=False)], dataMinMax1_isi=[''],
                                Loan_min_max1=[Loan_min_max1.to_html(justify='center',index=False)], Loan_min_max1_isi=[''])
    else:
        cek = 0
        transformasi = ""
        normalisasi_request = ""
        return render_template("preprocesing.html", cek=cek, a=transformasi,  b=normalisasi_request)

@app.route("/modelling/", methods=["GET","POST"])
def modelling():
    if request.method == "POST":
        global modeling, bagging
        cek = 1
        modeling = request.form['modelling']
        bagging = request.form['bagging']
        if modeling == '1':
            if normalisasi_request == '1':
                # Mengambil kelas dan fitur dari dataset pada Z score
                # fiturnya
                X_Zscore = Loan_Zscore.iloc[:,1:13].values
                # classnya
                y_Zscore = Loan_Zscore.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_Zscore, X_testn_Zscore, y_trainn_Zscore, y_testn_Zscore = train_test_split(X_Zscore, y_Zscore, test_size=0.30, random_state=0, stratify=y_Zscore)
                # Mengatasi eror pada data label dengan labeling
                y_trainn_Zscore = label_encoder.fit_transform(y_trainn_Zscore)
                y_testn_Zscore = label_encoder.fit_transform(y_testn_Zscore)
                # Menghitung akurasi dari naive bayes dengan normalisasi Z score
                gaussian = GaussianNB()
                gaussian.fit(X_trainn_Zscore, y_trainn_Zscore)
                Y_predn_Zscore = gaussian.predict(X_testn_Zscore)
                accuracy_n_Zscore = round(accuracy_score(y_testn_Zscore,Y_predn_Zscore)* 100, 2)
                hasil_akurasi = 'Akurasi Naive Bayes dengan normalisasi Z Score : %.3f' %accuracy_n_Zscore
                # save the model
                dump(gaussian, open('model_ZScore_Gaussian.pkl', 'wb'))
                if bagging == 'yes':
                    bagging_naive = BaggingClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=0).fit(X_trainn_Zscore, y_trainn_Zscore)
                    rsc = bagging_naive.predict(X_testn_Zscore)
                    c = ['Naive Bayes']
                    Bayes = pd.DataFrame(rsc,columns = c)
                    bagging_accuracy1 = round(100 * accuracy_score(y_testn_Zscore, Bayes), 2)
                    hasil_bagging = 'Akurasi Naive Bayes dengan normalisasi Z Score dengan bagging : %.3f'%bagging_accuracy1
                    # save the model
                    dump(bagging_naive, open('model_ZScore_Gaussian_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)

                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
            else:
                # Mengambil kelas dan fitur dari dataset pada Min Max
                # fiturnya
                X_min_max = Loan_min_max.iloc[:,1:13].values
                # classnya
                y_min_max = Loan_min_max.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_min_max, X_testn_min_max, y_trainn_min_max, y_testn_min_max = train_test_split(X_min_max, y_min_max, test_size=0.30, random_state=0, stratify=y_min_max)
                # Menghitung akurasi dari naive bayes dengan normalisasi min max
                gaussian = GaussianNB()
                gaussian.fit(X_trainn_min_max, y_trainn_min_max)
                Y_predn_min_max = gaussian.predict(X_testn_min_max) 
                accuracy_n_min_max=round(accuracy_score(y_testn_min_max,Y_predn_min_max)* 100, 2)
                hasil_akurasi = 'Akurasi Naive Bayes dengan normalisasi Min Max : %.3f' %accuracy_n_min_max
                # save the model
                dump(gaussian, open('model_MinMax_Gaussian.pkl', 'wb'))
                if bagging == 'yes':
                    bagging_naive = BaggingClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=0).fit(X_trainn_min_max, y_trainn_min_max)
                    rsc = bagging_naive.predict(X_testn_min_max)
                    c = ['Naive Bayes']
                    Bayes = pd.DataFrame(rsc,columns = c)
                    bagging_accuracy1 = round(100 * accuracy_score(y_testn_min_max, Bayes), 2)
                    hasil_bagging = 'Akurasi Naive Bayes dengan normalisasi Min Max dengan bagging : %.3f'%bagging_accuracy1
                    # save the model
                    dump(bagging_naive, open('model_MinMax_Gaussian_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)
                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
        elif modeling== '2':
            if normalisasi_request == '1':
                # Mengambil kelas dan fitur dari dataset pada Z score
                # fiturnya
                X_Zscore = Loan_Zscore.iloc[:,1:13].values
                # classnya
                y_Zscore = Loan_Zscore.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_Zscore, X_testn_Zscore, y_trainn_Zscore, y_testn_Zscore = train_test_split(X_Zscore, y_Zscore, test_size=0.30, random_state=0, stratify=y_Zscore)
                # Mengatasi eror pada data label dengan labeling
                y_trainn_Zscore = label_encoder.fit_transform(y_trainn_Zscore)
                y_testn_Zscore = label_encoder.fit_transform(y_testn_Zscore)
                # Menghitung akurasi dengan menggunakan gini indek dengan normalisasi Z score
                d_tree = tree.DecisionTreeClassifier(criterion="gini")
                d_tree.fit(X_trainn_Zscore, y_trainn_Zscore)
                acc_tree = round(d_tree.score(X_trainn_Zscore, y_trainn_Zscore) * 100, 2)
                hasil_akurasi = 'Akurasi Decision Tree dengan normalisasi Z Score : %.3f'%acc_tree
                dump(d_tree, open('model_ZScore_Tree.pkl', 'wb'))
                if bagging == 'yes':
                    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_trainn_Zscore, y_trainn_Zscore)
                    rsb = clf.predict(X_testn_Zscore)
                    b = ['Decision Tree']
                    Tree = pd.DataFrame(rsb,columns = b)
                    bagging_accuracy3 = round(100 * accuracy_score(y_testn_Zscore, Tree), 2)
                    hasil_bagging = 'Akurasi Decision Tree dengan normalisasi Z Score dengan bagging : %.3f'%bagging_accuracy3
                    # save the model
                    dump(clf, open('model_ZScore_Tree_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)
                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
            else:
                # Mengambil kelas dan fitur dari dataset pada Min Max
                # fiturnya
                X_min_max = Loan_min_max.iloc[:,1:13].values
                # classnya
                y_min_max = Loan_min_max.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_min_max, X_testn_min_max, y_trainn_min_max, y_testn_min_max = train_test_split(X_min_max, y_min_max, test_size=0.30, random_state=0, stratify=y_min_max)
                # Menghitung akurasi dengan menggunakan gini indek dengan normalisasi min max
                d_tree = tree.DecisionTreeClassifier(criterion="gini")
                d_tree.fit(X_trainn_min_max, y_trainn_min_max)
                acc_tree = round(d_tree.score(X_trainn_min_max, y_trainn_min_max) * 100, 2)
                hasil_akurasi = 'Akurasi Decision Tree dengan normalisasi Min Max : %.3f'%acc_tree
                # save the model
                dump(d_tree, open('model_MinMax_Tree.pkl', 'wb'))
                if bagging == 'yes':
                    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_trainn_min_max, y_trainn_min_max)
                    rsb = clf.predict(X_testn_min_max)
                    b = ['Decision Tree']
                    Tree = pd.DataFrame(rsb,columns = b)
                    bagging_accuracy3 = round(100 * accuracy_score(y_testn_min_max, Tree), 2)
                    hasil_bagging = 'Akurasi Decision Tree dengan normalisasi Min Max dengan bagging : %.3f'%bagging_accuracy3
                    # save the model
                    dump(clf, open('model_MinMax_Tree_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)
                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
        elif modeling == "3":
            if normalisasi_request == '1':
                # Mengambil kelas dan fitur dari dataset pada Z score
                # fiturnya
                X_Zscore = Loan_Zscore.iloc[:,1:13].values
                # classnya
                y_Zscore = Loan_Zscore.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_Zscore, X_testn_Zscore, y_trainn_Zscore, y_testn_Zscore = train_test_split(X_Zscore, y_Zscore, test_size=0.30, random_state=0, stratify=y_Zscore)
                # Mengatasi eror pada data label dengan labeling
                y_trainn_Zscore = label_encoder.fit_transform(y_trainn_Zscore)
                y_testn_Zscore = label_encoder.fit_transform(y_testn_Zscore)
                # Menghitung akurasi dari KNN dengan normalisasi Z score
                neigh = KNeighborsClassifier(n_neighbors=3)
                neigh.fit(X_trainn_Zscore, y_trainn_Zscore)
                acc_knn = round(neigh.score(X_trainn_Zscore, y_trainn_Zscore) * 100, 2)
                hasil_akurasi = 'Akurasi KNN dengan normalisasi Z Score : %.3f'%acc_knn
                # save the model
                dump(neigh, open('model_ZScore_KNN.pkl', 'wb'))
                if bagging == 'yes':
                    K = 3
                    knn = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors = K),n_estimators=10, random_state=0).fit(X_trainn_Zscore, y_trainn_Zscore)
                    rsa = knn.predict(X_testn_Zscore)
                    a = ['KNN']
                    KNN = pd.DataFrame(rsa,columns = a)
                    bagging_accuracy2 = round(100 * accuracy_score(y_testn_Zscore, KNN), 2)
                    hasil_bagging = 'Akurasi KNN dengan normalisasi Z Score dengan bagging : %.3f'%bagging_accuracy2
                    # save the model
                    dump(knn, open('model_ZScore_KNN_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)
                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
            else:
                # Mengambil kelas dan fitur dari dataset pada Min Max
                # fiturnya
                X_min_max = Loan_min_max.iloc[:,1:13].values
                # classnya
                y_min_max = Loan_min_max.iloc[:,0].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_min_max, X_testn_min_max, y_trainn_min_max, y_testn_min_max = train_test_split(X_min_max, y_min_max, test_size=0.30, random_state=0, stratify=y_min_max)
                # Menghitung akurasi dari KNN dengan normalisasi min max
                neigh = KNeighborsClassifier(n_neighbors=3)
                neigh.fit(X_trainn_min_max, y_trainn_min_max)
                acc_knn = round(neigh.score(X_trainn_min_max, y_trainn_min_max) * 100, 2)
                hasil_akurasi = 'Akurasi KNN dengan normalisasi Min Max : %.3f'%acc_knn
                # save the model
                dump(neigh, open('model_MinMax_KNN.pkl', 'wb'))
                if bagging == 'yes':
                    K = 3
                    knn = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors = K),n_estimators=10, random_state=0).fit(X_trainn_min_max, y_trainn_min_max)
                    rsa = knn.predict(X_testn_min_max)
                    a = ['KNN']
                    KNN = pd.DataFrame(rsa,columns = a)
                    bagging_accuracy2 = round(100 * accuracy_score(y_testn_min_max, KNN), 2)
                    hasil_bagging = 'Akurasi KNN dengan normalisasi Min Max dengan bagging : %.3f'%bagging_accuracy2
                    # save the model
                    dump(knn, open('model_MinMax_KNN_bagging.pkl', 'wb'))
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi, bagging1=hasil_bagging)
                else:
                    return render_template("modeling.html", cek=cek, a=modeling, b=bagging, akurasi=hasil_akurasi)
    else:
        cek=0
        modelling = ""
        bagging = ""
    return render_template("modeling.html", cek=cek, a=modelling, b=bagging)

@app.route("/prediksi/", methods=["GET","POST"])
def prediksi():
    if request.method == "POST":
        Gender = request.form['Gender']
        Married = request.form['Married']
        Dependents = request.form['Dependents']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        ApplicantIncome = request.form['ApplicantIncome']
        CoapplicantIncome = request.form['CoapplicantIncome']
        LoanAmount = request.form['LoanAmount']
        Loan_Amount_Term = request.form['Loan_Amount_Term']
        Credit_History = request.form['Credit_History']
        Property_Area = request.form['Property_Area']
        cek = 1
        # load the scaler
        scaler_ZScore = load(open('scaler_ZScore.pkl', 'rb'))
        scaler_MinMax = load(open('scaler_MinMax.pkl', 'rb'))
        # load the model
        model_ZScore_Gaussian = load(open('model_ZScore_Gaussian.pkl', 'rb'))
        model_MinMax_Gaussian = load(open('model_MinMax_Gaussian.pkl', 'rb'))
        model_ZScore_Gaussian_bagging = load(open('model_ZScore_Gaussian_bagging.pkl', 'rb'))
        model_MinMax_Gaussian_bagging = load(open('model_MinMax_Gaussian_bagging.pkl', 'rb'))
        model_ZScore_Tree = load(open('model_ZScore_Tree.pkl', 'rb'))
        model_MinMax_Tree = load(open('model_MinMax_Tree.pkl', 'rb'))
        model_ZScore_Tree_bagging = load(open('model_ZScore_Tree_bagging.pkl', 'rb'))
        model_MinMax_Tree_bagging = load(open('model_MinMax_Tree_bagging.pkl', 'rb'))
        model_ZScore_KNN = load(open('model_ZScore_KNN.pkl', 'rb'))
        model_MinMax_KNN = load(open('model_MinMax_KNN.pkl', 'rb'))
        model_ZScore_KNN_bagging = load(open('model_ZScore_KNN_bagging.pkl', 'rb'))
        model_MinMax_KNN_bagging = load(open('model_MinMax_KNN_bagging.pkl', 'rb'))
        # Data Ditampung Dalam Bentuk Array
        dataArray = [[int(Gender), int(Married), int(Dependents), int(Education), int(Self_Employed), int(ApplicantIncome), int(CoapplicantIncome), int(LoanAmount), int(Loan_Amount_Term), int(Credit_History), int(Property_Area)]]
        if (normalisasi_request == '1'):
            # Data Dinormalisasi
            hasil_scale_ZScore = (scaler_ZScore.transform(dataArray))
            if (modeling == '0'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_ZScore_Gaussian_bagging.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan Naive Bayes kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_ZScore_Gaussian.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan Naive Bayes."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
            elif (modeling == '1' or modeling == '2' or modeling == '3'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_ZScore_Tree_bagging.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan Decission Tree kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_ZScore_Tree.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan Decission Tree."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
            elif (modeling == '0'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_ZScore_KNN_bagging.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan KNN kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_ZScore_KNN.predict(hasil_scale_ZScore)
                    pesan = "Anda menggunakan normalisasi Z Score dan pemodelan KNN."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
        elif (normalisasi_request == '2'):
            # Data Dinormalisasi
            hasil_scale_MinMax = (scaler_MinMax.transform(dataArray))
            if (modeling == '0'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_MinMax_Gaussian_bagging.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan Naive Bayes kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_MinMax_Gaussian.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan Naive Bayes."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
            elif (modeling == '1' or modeling == '2' or modeling == '3'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_MinMax_Tree_bagging.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan Decission Tree kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_MinMax_Tree.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan Decission Tree."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
            elif (modeling == '0'):
                if (bagging == 'yes'):
                    # Data dimodelkan
                    hasil = model_MinMax_KNN_bagging.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan KNN kombinasi bagging."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
                else:
                    # Data dimodelkan
                    hasil = model_MinMax_KNN.predict(hasil_scale_MinMax)
                    pesan = "Anda menggunakan normalisasi Min Max dan pemodelan KNN."
                    if (hasil == 1):
                        prediksine = "DITERIMA"
                    else:
                        prediksine = "DITOLAK"
                    return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, 
                            e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome, h=LoanAmount, i=Loan_Amount_Term, 
                            j=Credit_History, k=Property_Area, prediksi=prediksine, pesan=pesan)
    else:
        Gender = ''
        Married = ''
        Dependents = ''
        Education = ''
        Self_Employed = ''
        ApplicantIncome = ''
        CoapplicantIncome = ''
        LoanAmount = ''
        Loan_Amount_Term = ''
        Credit_History = ''
        Property_Area = ''
        hasil = ''
        pesan = ''
        cek = 0
        return render_template("prediksi.html", cek=cek, a=Gender, b=Married, c=Dependents, d=Education, e=Self_Employed, f=ApplicantIncome, g=CoapplicantIncome,
                            h=LoanAmount, i=Loan_Amount_Term, j=Credit_History, k=Property_Area, prediksi=hasil, pesan=pesan)
