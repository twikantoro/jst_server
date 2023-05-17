import sqlite3


def get_models_all():
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    c.execute("SELECT * FROM model ORDER BY id DESC")
    models = c.fetchall()
    conn.close()
    return models


def create_model(data):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    res = c.execute(
        """
    INSERT INTO model(nama,latih_mulai,latih_akhir,latih_pakai_london,uji_mulai,uji_akhir,uji_pakai_london,susunan_neuron)
    VALUES(?,?,?,?,?,?,?,?)
    """,
        data,
    )
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return rowid


def get_model_by_id(id):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    c.execute("SELECT * FROM model WHERE id=" + str(id))
    res = c.fetchone()
    conn.close()
    return res


def get_latest_iteration_by_model_id(id):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    c.execute("SELECT * FROM iterasi WHERE id_model=" + str(id) + " ORDER BY id DESC")
    res = c.fetchone()
    conn.close()
    return res


# def get_latest_iteration_by_model_id(id):
#     iterations = get_all_iteration_by_model_id(id)
#     return iterations[-1]


def get_all_iteration_by_model_id(id):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM iterasi WHERE id_model=" + str(id))
        res = c.fetchall()
    except:
        res = []
    conn.close()
    return res


def delete_model(id):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    c.execute("DELETE FROM model WHERE id="+str(id))
    conn.commit()
    conn.close()
    return "ok"


def iterasi_add(data):
    conn = sqlite3.connect("db/jst_server.db")
    c = conn.cursor()
    res = c.execute(
        """
    INSERT INTO iterasi(id_model,ke,loss_train,akurasi_train,loss,akurasi)
    VALUES(?,?,?,?,?,?)
    """,
        data,
    )
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return rowid