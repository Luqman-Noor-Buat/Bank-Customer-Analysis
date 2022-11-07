from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/hasil/", methods=["POST"])
def hasil():
    kode_kontrak = request.form['kode_kontrak']
    pendapatan_setahun_juta = request.form['pendapatan_setahun_juta']
    durasi_pinjaman_bulan = request.form['durasi_pinjaman_bulan']
    jumlah_tanggungan = request.form['jumlah_tanggungan']
    kpr_aktif = request.form['kpr_aktif']
    rata_rata_overdue = request.form['rata_rata_overdue']

    # mengumpulkan data menjadi satu
    normalisasi = "TIDAK DIKETAHUI"
    return render_template("hasil.html", a=kode_kontrak, b=pendapatan_setahun_juta, c=durasi_pinjaman_bulan, d=jumlah_tanggungan, e=kpr_aktif, f=rata_rata_overdue, g=normalisasi)