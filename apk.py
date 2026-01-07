from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox,
    QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsLineItem, QGraphicsTextItem,
    QTabWidget,QDoubleSpinBox, QFormLayout, QTextEdit
)
from PySide6.QtGui import QBrush, QFont, QPen,QPainterPath,QColor
from PySide6.QtCore import Qt, QRectF

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import sys
import time

import mrowa2
import genetic


# =========================
# CityItem
# =========================

class CityItem(QGraphicsItem):
    def __init__(self, x, y, index, radius=5):
        super().__init__()
        self.index = index
        self.radius = radius
        self.pack_letters = []
        self.color = Qt.red
        self.setZValue(1)
        self.setPos(x, y)

    def set_color(self, color):
        self.color = color
        self.update()

    def add_package(self, letter, type_):
        self.pack_letters.append(letter + type_)
        self.update()

    def boundingRect(self):
        r = self.radius
        return QRectF(-r-30, -r-40, 2*r+60, 2*r+40)
    
    def shape(self):
        path = QPainterPath()
        r = self.radius
        path.addEllipse(-r, -r, 2*r, 2*r)
        return path

    def paint(self, painter, option, widget):
        r = self.radius
        painter.setBrush(QBrush(self.color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-r, -r, 2*r, 2*r)

        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.white)
        painter.drawText(-r,-r-5, str(self.index))

        if self.pack_letters:
            painter.setPen(Qt.darkGreen)
            painter.drawText(-r, -r-20, ",".join(self.pack_letters))

# =========================
# EdgeItem (nieskierowana)
# =========================
class EdgeItem(QGraphicsLineItem):
    def __init__(self, city_a, city_b):
        super().__init__()
        self.city_a = city_a
        self.city_b = city_b
        self.dist = self.calc_dist(city_a.x(), city_a.y(), city_b.x(), city_b.y())
        self.rob_prop = int(100*(1-1/(self.dist*0.01+1)))
        self.key = frozenset({city_a.index, city_b.index})

        self.pen = QPen(Qt.black, 2)
        self.setPen(self.pen)
        self.setZValue(0)

        self.label = QGraphicsTextItem(f"{self.dist}km p: {self.rob_prop}%", self)
        self.label.setDefaultTextColor(Qt.white)
        self.label.setZValue(1)
        self.update_position()

    def calc_dist(self,x1,y1,x2,y2): return round((((x1-x2)**2+(y1-y2)**2)**0.5)/20,2)

    def update_position(self):
        a = self.city_a.pos()
        b = self.city_b.pos()
        self.setLine(a.x(), a.y(), b.x(), b.y())
        mx = (a.x() + b.x()) / 2
        my = (a.y() + b.y()) / 2
        rect = self.label.boundingRect()
        self.label.setPos(mx - rect.width() / 2, my - rect.height() / 2)

    def set_default_style(self):
        self.pen.setColor(Qt.black)
        self.pen.setWidth(2)
        self.setPen(self.pen)
        
    def set_highlight_style(self):
        self.pen.setColor(Qt.blue)
        self.pen.setWidth(4)
        self.setPen(self.pen)
# =========================
# MapView
# =========================
class MapView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0,0,1000, 1000)
        self.setScene(self.scene)
        self.setFixedSize(1050, 1050)
        self.draw_grid()

        self.set_variables()
        self.parcels_panel = None

    def draw_grid(self, step=20, size=1000):
        pen = QPen(QColor(100, 100, 100))
        for i in range(0, size + 1, step):
            self.scene.addLine(i, 0, i, size, pen)
            self.scene.addLine(0, i, size, i, pen)

    def set_variables(self):                    #Dane grafu
        self.cities = [] #lista obiektow miast
        self.edges = {} #slownik pary indeksow na obiekt krawedzi
        self.base = None #obiekt bazy
        self.parcels = [] #lista zamowien (p_ind,d_ind,val)
        self.parcels_letters = [] #lista liter zamowien
        self.dist_mat = [] #macierz odleglosci (lista list)
        self.prop_mat = [] #macierz prawdopodobienstw napadu (lista list)

        self.mode = None
        self.temp_pickup = None
        self.temp_edge_city = None
        self.current_value = None
        self.next_letter_index = 0

    def set_mode(self, mode, value=None):
        self.mode = mode
        self.current_value = value
        self.temp_pickup = None
        self.temp_edge_city = None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton: return

        pos = self.mapToScene(event.position().toPoint())
        items = self.scene.items(pos)
        clicked_city = None

        for item in items:
            if isinstance(item, CityItem):
                clicked_city = item
                break

        if self.mode == "add_city":
            if not((pos.x() > 0 and pos.x() < 1000) and (pos.y() > 0 and pos.y() < 1000)): return
            if clicked_city != None: return
            self.add_city(pos.x(),pos.y())

        elif self.mode == "add_parcel_pickup" and clicked_city:
            self.temp_pickup = clicked_city
            self.mode = "add_parcel_delivery"

        elif self.mode == "add_parcel_delivery" and clicked_city:
            self.add_parcell(self.temp_pickup,clicked_city,self.current_value)

        elif self.mode == "add_edge" and clicked_city:
            if self.temp_edge_city is None:
                self.temp_edge_city = clicked_city
            else:
                self.add_edge(clicked_city,self.temp_edge_city)
                self.temp_edge_city = clicked_city
                self.mode = "add_edge"
        
        elif self.mode == "add_base" and clicked_city:
            self.add_base(clicked_city)

    def add_city(self,x,y):
        index = len(self.cities)
        city = CityItem(x,y, index)
        self.scene.addItem(city)
        self.cities.append(city)
        self.update_mat()

    def add_edge(self,city_a,city_b):
        if city_a == city_b: return
        key = frozenset({
            city_b.index,
            city_a.index
        })
        if key not in self.edges:
            edge = EdgeItem(city_b, city_a)
            self.scene.addItem(edge)
            self.edges[key] = edge
            self.update_mat()

    def get_next_letter(self):
        letter = chr(ord('A') + self.next_letter_index)
        self.next_letter_index += 1
        return letter

    def add_parcell(self,city_p,city_d,value):
        if city_d == city_p: return
        letter = self.get_next_letter()
        city_p.add_package(letter, "p")
        city_d.add_package(letter, "d")
        self.parcels.append((city_p.index,city_d.index,value))
        self.parcels_letters.append(letter)
        self.parcels_panel.add_parcel(letter,city_p.index,city_d.index,value)
        self.mode = None

    def add_base(self,city):
        if self.base != None: self.base.set_color(Qt.red)
        self.base = city
        self.base.set_color(Qt.yellow)

    def clear(self):
        for edge in self.edges.values(): self.scene.removeItem(edge)
        for city in self.cities: self.scene.removeItem(city)
        self.parcels_panel.clear_table()
        self.set_variables()

    def draw_path(self,path):
        if path == None or path == []: return
        if self.edges != None:
            for edge in self.edges.values():
                edge.set_default_style()
            for i in range(len(path)-1):
                key = frozenset({path[i],path[i+1]})
                edge = self.edges[key]
                edge.set_highlight_style()

    def update_mat(self):
        if self.edges != None:
            self.dist_mat = []
            self.prop_mat = []
            n = len(self.cities)
            self.dist_mat = [[None for _ in range(n)] for _ in range(n)]
            self.prop_mat = [[None for _ in range(n)] for _ in range(n)]
            for edge in self.edges.values():
                self.dist_mat[edge.city_a.index][edge.city_b.index] = edge.dist
                self.dist_mat[edge.city_b.index][edge.city_a.index] = edge.dist
                self.prop_mat[edge.city_a.index][edge.city_b.index] = edge.rob_prop
                self.prop_mat[edge.city_b.index][edge.city_a.index] = edge.rob_prop

    def is_all_connected(self):
        n = len(self.dist_mat)
        visited = [False]*n
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if self.dist_mat[node][neighbor] is not None and not visited[neighbor]:
                    dfs(neighbor)
        dfs(0)
        return all(visited)

    def upload(self,path):
        self.clear()
        with open(path, "r", encoding="utf-8") as f:
            data_type = None
            for line in f:
                line = line.strip()
                if line in ["cords","edges","parcells","base"]: 
                    data_type = line
                    continue
                dane = list(map(int, line.split()))

                if data_type == "cords":
                    self.add_city(dane[0],dane[1])
                elif data_type == "edges":
                    city_a = self.cities[dane[0]]
                    city_b = self.cities[dane[1]]
                    self.add_edge(city_a,city_b)          
                elif data_type == "parcells":
                    city_p = self.cities[dane[0]]
                    city_b = self.cities[dane[1]]
                    val = dane[2]
                    self.add_parcell(city_p,city_b,val)
                elif data_type == "base":
                    self.add_base(self.cities[dane[0]])

    def download(self):
        txt = ""
        txt += "cords\n"
        if self.cities != None:
            for city in self.cities:
                txt += str(int(city.x()))+" "+str(int(city.y()))+"\n"
        txt += "edges\n"
        if self.edges != None:
            for edge in self.edges.values():
                txt += str(edge.city_a.index)+" "+str(edge.city_b.index)+"\n"
        txt += "parcells\n"
        if  self.parcels != None:
            for parcel in self.parcels:
                txt += str(parcel[0])+" "+str(parcel[1])+" "+str(parcel[2])+"\n"
        txt += "base\n"
        if self.base != None:
            txt += str(self.base.index)

        with open("mapa.txt", "w", encoding="utf-8") as f:
            f.write(txt)

# =========================
# ParcelsPanel
# =========================
class ParcelsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Zlecenia"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Paczka", "Pickup", "Delivery", "Zysk"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_parcel(self, letter, pickup, delivery, value):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(letter))
        self.table.setItem(row, 1, QTableWidgetItem(str(pickup)))
        self.table.setItem(row, 2, QTableWidgetItem(str(delivery)))
        self.table.setItem(row, 3, QTableWidgetItem(str(value)))

    def clear_table(self):
        self.table.setRowCount(0)

# =========================
# Plot
# =========================
class SimplePlot(FigureCanvas):
    def __init__(self, title, ylabel):
        fig = Figure(figsize=(5, 3))
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.title = title
        self.ylabel = ylabel
        self.set_params()

    def set_params(self):
        self.fig.patch.set_facecolor("#2D2D2D")
        self.ax.set_facecolor("#3a3a3a")
        self.ax.set_title(self.title, color="white", loc="left")
        self.ax.set_xlabel("Iteracje", color="white")
        self.ax.set_ylabel(self.ylabel, color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, color="#555555")

    def set_data(self, data):
        self.ax.clear()
        self.set_params()
        self.ax.plot(range(len(data)), data, marker=".", color="#4fc3f7")
        self.draw_idle()

# =========================
# MainWindow
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Problem karawany - algorytm mrówkowy")
        self.resize(1700, 1000)

        central = QWidget()
        main_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # ---------- MAPA ----------
        map_tab = QWidget()
        map_tab_layout = QHBoxLayout(map_tab)
        self.map_view = MapView()

        # Lewy layout
        left_layout = QVBoxLayout()

        map_title_label = QLabel("Mapa miast i połączeń 50km x 50km (1 kratka = 1km)")
        map_title_label.setAlignment(Qt.AlignCenter)
        map_title_label.setStyleSheet("font-size: 16px; color: white;")

        self.path_txt_label = QTextEdit()
        self.path_txt_label.setReadOnly(True)
        self.path_txt_label.setStyleSheet("font-size: 16px; color: white; background-color: #2b2b2b;")
        self.path_txt_label.setLineWrapMode(QTextEdit.NoWrap)
        self.path_txt_label.setAlignment(Qt.AlignLeft)
        self.path_txt_label.setMaximumWidth(1050)

        left_layout.addWidget(map_title_label)
        left_layout.addWidget(self.map_view)
        left_layout.addWidget(self.path_txt_label)
        map_tab_layout.addLayout(left_layout, 3)
        
        # Prawy layout
        right_layout = QVBoxLayout()
        
        self.add_city_btn = QPushButton("Dodawaj miasta")
        self.add_city_btn.clicked.connect(lambda: self.map_view.set_mode("add_city"))

        self.add_edge_btn = QPushButton("Dodaj połączenie")
        self.add_edge_btn.clicked.connect(lambda: self.map_view.set_mode("add_edge"))

        self.add_base_btn = QPushButton("Ustaw bazę")
        self.add_base_btn.clicked.connect(lambda: self.map_view.set_mode("add_base"))

        order_layout = QHBoxLayout()
        self.value_box = QSpinBox()
        self.value_box.setRange(1, 10000)
        self.value_box.setValue(100)

        self.add_parcel_btn = QPushButton("Ustaw zlecenie")
        self.add_parcel_btn.clicked.connect(lambda: self.map_view.set_mode("add_parcel_pickup",self.value_box.value()))

        order_layout.addWidget(QLabel("Zysk ze zlecenia: "))
        order_layout.addWidget(self.value_box)
        order_layout.addWidget(self.add_parcel_btn)

        self.parcels_panel = ParcelsPanel()
        self.map_view.parcels_panel = self.parcels_panel

        clear_btn = QPushButton("Wyczyść mapę")
        clear_btn.clicked.connect(self.map_view.clear)

        self.add_upload_btn = QPushButton("Załaduj z pliku .txt")
        self.add_upload_btn.clicked.connect(lambda: self.map_view.upload("mapa.txt"))

        self.add_download_btn = QPushButton("Wygeneruj plik .txt") 
        self.add_download_btn.clicked.connect(lambda: self.map_view.download())

        # Parametry algorytmu mrówkowego

        ant_info_label = QLabel("Parametry algorytmu mrówkowego")
        ant_info_label.setAlignment(Qt.AlignCenter)
        ant_info_label.setStyleSheet("font-size: 16px; color: white;")

        ant_params_form = QFormLayout()
        it_box = QSpinBox(); it_box.setRange(1,1000)
        ants_box = QSpinBox(); ants_box.setRange(1,200) 
        alpha_box = QDoubleSpinBox(); alpha_box.setRange(0.1,5.0)
        beta_box = QDoubleSpinBox(); beta_box.setRange(0.1, 10.0)
        evap_box = QDoubleSpinBox(); evap_box.setRange(0.01, 0.9); evap_box.setSingleStep(0.01)

        def set_default_ant_params():
            it_box.setValue(100)
            ants_box.setValue(50)
            alpha_box.setValue(1.0)
            evap_box.setValue(0.5)
            beta_box.setValue(2.0)
        set_default_ant_params()

        ant_params_form.addRow("Iteracje", it_box)
        ant_params_form.addRow("Mrówki", ants_box)
        ant_params_form.addRow("Alpha", alpha_box)
        ant_params_form.addRow("Beta", beta_box)
        ant_params_form.addRow("Parowanie", evap_box)

        default_params_ant_btn = QPushButton("Przywróć domyślne parametry")
        default_params_ant_btn.clicked.connect(set_default_ant_params)

        #Parametry algorytmu genetycznego

        gen_info_label = QLabel("Parametry algorytmu mrówkowego")
        gen_info_label.setAlignment(Qt.AlignCenter)
        gen_info_label.setStyleSheet("font-size: 16px; color: white;")

        gen_params_form = QFormLayout()
        pop_box = QSpinBox(); pop_box.setRange(1,500)
        gen_box = QSpinBox(); gen_box.setRange(1,500)
        mut_box = QDoubleSpinBox(); mut_box.setRange(0.01,0.9); mut_box.setSingleStep(0.01)

        def set_default_gen_params():
            pop_box.setValue(100)
            gen_box.setValue(200)
            mut_box.setValue(0.05)
        set_default_gen_params()

        gen_params_form.addRow("Wielkosc populaji", pop_box)
        gen_params_form.addRow("Generacje", gen_box)
        gen_params_form.addRow("Mutacja", mut_box)

        default_params_gen_btn = QPushButton("Przywróć domyślne parametry")
        default_params_gen_btn.clicked.connect(set_default_gen_params)

        compute_path_btn = QPushButton("Oblicz najkrótszą trasę z optymalnymi kupnami ochrony")
        compute_path_btn.clicked.connect(
            lambda: self.compute_path(
                it_box.value(),
                ants_box.value(),
                alpha_box.value(),
                beta_box.value(),
                evap_box.value(),
                pop_box.value(),
                gen_box.value(),
                mut_box.value()

            )
        )
        # Komunikaty
        info_txt = QLabel("Info:")
        info_txt.setStyleSheet("font-size: 16px; color: white;")
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 16px; color: red;")
        
        right_layout.addWidget(self.add_city_btn)
        right_layout.addWidget(self.add_edge_btn)
        right_layout.addWidget(self.add_base_btn)
        right_layout.addLayout(order_layout)
        right_layout.addWidget(self.parcels_panel)
        right_layout.addWidget(clear_btn)
        right_layout.addWidget(self.add_download_btn)
        right_layout.addWidget(self.add_upload_btn)
        right_layout.addWidget(ant_info_label)
        right_layout.addLayout(ant_params_form)
        right_layout.addWidget(default_params_ant_btn)
        right_layout.addWidget(gen_info_label)
        right_layout.addLayout(gen_params_form)
        right_layout.addWidget(default_params_gen_btn)
        right_layout.addWidget(compute_path_btn)
        right_layout.addWidget(info_txt)
        right_layout.addWidget(self.info_label)
        right_layout.addStretch()

        map_tab_layout.addLayout(right_layout, 1)
        tabs.addTab(map_tab, "Mapa grafu")

        # ----------  WYNIKI ----------
        results_tab = QWidget()

        results_layout = QVBoxLayout(results_tab)
        self.distance_plot = SimplePlot("Dystans w kolejnych iteracjach", "Dystans")
        self.profit_plot = SimplePlot("Zysk w kolejnych iteracjach", "Zysk")

        self.results_label = QTextEdit("Tu pojawią się wyniki programu:")
        self.results_label.setReadOnly(True)
        self.results_label.setStyleSheet("font-size: 16px; color: white; background-color: #2b2b2b;")
        self.results_label.setLineWrapMode(QTextEdit.NoWrap)
        self.results_label.setAlignment(Qt.AlignLeft)
        self.results_label.setMaximumWidth(1050)
        
        results_layout.addWidget(self.distance_plot)
        results_layout.addWidget(self.profit_plot)
        results_layout.addWidget(self.results_label)

        tabs.addTab(results_tab, "Wyniki")
        self.setCentralWidget(central)

        # ----------  MACIERZE ----------
        matrix_tab = QWidget()
        matrix_layout = QHBoxLayout(matrix_tab)

        self.matrix_table = QTableWidget()
        self.matrix_table.setEditTriggers(QTableWidget.NoEditTriggers)
        matrix_layout.addWidget(self.matrix_table)

        self.showing_distance = True
        self.matrix_btn = QPushButton("Pokaż macierz odległości")
        self.matrix_btn.clicked.connect(self.toggle_matrix)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.matrix_btn)
        right_layout.addStretch()

        matrix_layout.addLayout(right_layout)

        tabs.addTab(matrix_tab, "Macierze")
        
        # ----------  INFO ----------
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        app_info_label = QLabel("O programie:")
        app_info_label.setStyleSheet("font-size: 32px; color: white;")
        app_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        info_layout.addWidget(app_info_label)

        tabs.addTab(info_tab, "Informacje o programie")
        self.setCentralWidget(central)
        
    def toggle_matrix(self):
        if self.showing_distance:
            matrix = self.map_view.dist_mat
            self.show_matrix(matrix, self.matrix_table)
            self.matrix_btn.setText("Pokaż macierz prawdopodobieństw")
        else:
            matrix = self.map_view.prop_mat
            self.show_matrix(matrix, self.matrix_table)
            self.matrix_btn.setText("Pokaż macierz odległości")
        self.showing_distance = not self.showing_distance
    
    def show_matrix(self, matrix, table):
        n = len(matrix)
        table.setRowCount(n)
        table.setColumnCount(n)
        headers = [str(i) for i in range(n)]
        table.setHorizontalHeaderLabels(headers)
        table.setVerticalHeaderLabels(headers)
        unit = "m" if self.showing_distance else "%"
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                table.setItem(i, j, QTableWidgetItem("—" if val is None else (str(val)+unit))
                )
    
    def compute_path(self,iter,ants,alfa,beta,evap,pop,gen,mut):
        
        self.info_label.setStyleSheet("font-size: 16px; color: red;")
        if self.map_view.base == None: self.info_label.setText("Nie wybrałeś bazowego wierzchołka"); return
        if len(self.map_view.cities)<2: self.info_label.setText("Dodaj przynajmniej jeszcze jedno miasto"); return
        if not self.map_view.is_all_connected(): self.info_label.setText("Graf nie jest spójny!"); return
        if len(self.map_view.parcels)==0: self.info_label.setText("Brak zamówień, nie trzeba ruszać z bazy"); return

        start_timer = time.time()

        alg_dist_mat = np.array(self.map_view.dist_mat,dtype=object)
        parcels = self.map_view.parcels
        base = self.map_view.base.index
        ant_params = [iter,ants,alfa,beta,evap]
        
        alg = mrowa2.AntColonyOptimization(alg_dist_mat,parcels,base,ant_params)
        best_path, best_dist, ant_history, orders_sequence = alg.solve()
        best_path = [int(x) for x in best_path] 
        ant_history = [int(x) for x in ant_history]

        self.map_view.draw_path(best_path)

        trasa_input = []
        for i in range(len(best_path)-1): #Generowanie trasy jako wejscie do ag
            ind = best_path [i]
            next_ind = best_path[i+1]
            edge = self.map_view.edges[frozenset({ind,next_ind})]
            rob_prop = (edge.rob_prop)/100
            city_props = [x for x in self.map_view.prop_mat[ind] if x is not None]
            avg_prop_from_city_cost = sum(city_props)/len(city_props)/100
            const_cost = 100
            prot_cost = int(const_cost*avg_prop_from_city_cost)
            trasa_input.append((ind,rob_prop,prot_cost))

        ga = genetic.SingleCargoGA(trasa_input,parcels,pop,gen,mut)
        buy_protect, ga_history = ga.run()
        best_score = ga_history[-1]
        
        self.distance_plot.set_data(ant_history)
        self.profit_plot.set_data(ga_history)
        self.toggle_matrix()
        self.info_label.setStyleSheet("font-size: 16px; color: white;")
        self.info_label.setText('Obliczono Optmalną trase która minimalizuje dystans dla zadanych zleceń ' \
        '                      \noraz maxymalizuje z nich zarobek' \
        '                      \nDokładne informacje o trasie w zakładce "wyniki"')

        path_txt = "Trasa: "
        path_list_txt = "Lista kroków: | krok | z kąd |do kąd | paczka | wartość paczki | czy ochrona | koszt ochrony \n"

        has_parcel = False
        parcel_letters = self.map_view.parcels_letters
        cities_protected = []
        seq_idx = 0

        for step_idx in range(len(best_path)-1):
            node = best_path[step_idx]
            next_node = best_path[step_idx+1]

            path_txt += f"{node} "

            if has_parcel: # Paczka
                pickup, delivery, _ = parcels[parcel_id]
                if node == delivery:
                    has_parcel = False
                    path_txt += f"{parcel_letter}d "
                    seq_idx += 1
            elif seq_idx < len(orders_sequence):
                parcel_id = orders_sequence[seq_idx]
                parcel_letter = parcel_letters[parcel_id]
                pickup, delivery, _ = parcels[parcel_id]
                if node == pickup:
                    has_parcel = True
                    path_txt += f"{parcel_letter}p "
            
            path_txt += "--"
            if buy_protect[step_idx]: 
                cities_protected.append(node)
                path_txt += "$" # Ochrona
            path_txt += "-> "  # Strzałka

            path_list_txt += f"{step_idx} | {node} | {next_node} "
            if has_parcel: 
                path_list_txt += f"| {parcel_letter} | {parcels[parcel_id][2]} "
                if buy_protect[step_idx]: path_list_txt += f"| Tak | {trasa_input[step_idx][2]}"
                else: path_list_txt += f"| Nie | ---"
            path_list_txt += "\n"

        path_txt += f"{base}"
        path_txt += "\n Oznaczenia: Np - Odbiór paczki N, Nd - Dostawa paczki, $ - Wykupiona ochrona na odcinku"
        self.path_txt_label.setText(path_txt)

        letters_order = []
        for i in orders_sequence: letters_order.append(parcel_letters[i])

        results_txt = "Wyniki:\n"
        results_txt += f"Kolejność odwiedzanych wierzchołków: {best_path}\n"
        results_txt += f"Dystans trasy: {best_dist:.3f} km\n"
        results_txt += f"Kolejność wykonywanych zleceń: {letters_order}\n"
        results_txt += f"Wierchołki w których kupiono ochronę: {cities_protected}\n"
        results_txt += f"Przewidywany zarobek: {best_score:.0f}\n"
        results_txt += f"Czas wykonywania algorytmu: {time.time()-start_timer:.3f}s\n"
        results_txt += path_list_txt
        self.results_label.setText(results_txt)



# =========================
# START
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
