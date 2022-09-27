import os

import PIL
import plotly
import plotly.express as px
from fpdf import FPDF



class PDF(FPDF):
    def lines(self):
        self.set_line_width(0.0)
        self.line(5.0, 5.0, 205.0, 5.0)  # top one
        self.line(5.0, 292.0, 205.0, 292.0)  # bottom one
        self.line(5.0, 5.0, 5.0, 292.0)  # left one
        self.line(205.0, 5.0, 205.0, 292.0)  # right one

    def titles(self, title):
        self.set_xy(0.0, 0.0)
        self.set_font("Arial", "B", 26)
        self.set_text_color(0, 0, 0)
        self.cell(w=210.0, h=40.0, align="C", txt=title, border=0)


def BACKTEST_REPORT(output):

    font = "Arial"

    SYMBOL = output["SYMBOL"]
    SPREAD = output["SPREAD"]
    N_CLUSTERS = output["N_CLUSTERS"]
    PREDICTIVE_CLUSTERS = output["PREDICTIVE_CLUSTERS"]
    MIN_PIPS = output["MIN_PIPS"]
    RETURN = output["RETURN"]
    SHARPE = output["SHARPE"]
    N_TRADE = output["N_TRADE"]
    WIN_RATE = output["WIN_RATE"]
    MAX_DRAWDOWN = output["MAX_DRAWDOWN"]
    path = output["PATH"]
    BENCHMARK_RETURN = output["BENCHMARK_RETURN"]
    START_DATE = output["START_DATE"]
    END_DATE = output["END_DATE"]
    return_path = path + "\\temp_return.png"
    leverage_path = path + "\\temp_leverage.png"
    orders_path = path + "\\temp_orders.png"

    pdf = PDF()
    pdf.add_page()

    # TITRE
    pdf.titles("BACKTEST REVIEW")

    # PARTIE I
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(font, "B", 14)
    pdf.set_xy(10, 45)
    pdf.multi_cell(0, 5, "PARAMETERS")
    pdf.set_line_width(0.5)
    pdf.line(10, 50, 205, 50)  # top one
    pdf.set_xy(10, 55)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Simulation dates ")
    pdf.set_xy(100, 55)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, START_DATE + " to " + END_DATE)
    pdf.set_xy(10, 60)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Symbol ")
    pdf.set_xy(100, 60)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, SYMBOL)
    pdf.set_xy(10, 65)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Spread ")
    pdf.set_xy(100, 65)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, str(SPREAD * 100) + "%")

    # PARTIE II
    pdf.set_xy(10, 80)
    pdf.set_font(font, "B", 14)
    pdf.multi_cell(0, 5, "ENCODER")
    pdf.line(10, 85, 205, 85)  # top one
    pdf.set_xy(10, 90)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Encoder ")
    pdf.set_xy(100, 90)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, "LSTM")
    pdf.set_xy(10, 95)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Input dimension ")
    pdf.set_xy(100, 95)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, str(8))
    pdf.set_xy(10, 100)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Latent dimension ")
    pdf.set_xy(100, 100)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, str(3))

    # PARTIE III
    pdf.set_xy(10, 110)
    pdf.set_font(font, "B", 14)
    pdf.multi_cell(0, 5, "CLUSTER")
    pdf.line(10, 115, 205, 115)
    pdf.set_xy(10, 120)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Clusterizer ")
    pdf.set_xy(100, 120)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, "KMEANS")
    pdf.set_xy(10, 125)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Number of clusters ")
    pdf.set_xy(100, 125)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, str(N_CLUSTERS))
    pdf.set_xy(10, 130)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "Number of predictive clusters")
    pdf.set_xy(100, 130)
    pdf.set_font(font, "B", 12)
    pdf.multi_cell(0, 5, str(PREDICTIVE_CLUSTERS))

    # PARTIE IV
    pdf.set_xy(140, 150)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "RETURN  ")
    pdf.set_xy(175, 150)
    pdf.set_font(font, "B", 20)
    if RETURN < 0:
        pdf.set_text_color(255, 0, 0)
    else:
        pdf.set_text_color(0, 255, 0)
    pdf.multi_cell(0, 5, str(round(RETURN, 1)) + "%")
    pdf.set_text_color(0, 0, 0)

    pdf.set_xy(140, 160)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "BENCHMARK ")
    pdf.set_xy(175, 160)
    pdf.set_font(font, "B", 20)
    if BENCHMARK_RETURN < 0:
        pdf.set_text_color(255, 0, 0)
    else:
        pdf.set_text_color(0, 255, 0)
    pdf.multi_cell(0, 5, str(round(BENCHMARK_RETURN, 1)) + "%")
    pdf.set_text_color(0, 0, 0)

    pdf.set_xy(140, 170)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "SHARPE ")
    pdf.set_xy(180, 170)
    pdf.set_font(font, "B", 20)
    pdf.multi_cell(0, 5, str(round(SHARPE, 1)))
    pdf.set_xy(140, 180)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "TRADES")
    pdf.set_xy(180, 180)
    pdf.set_font(font, "B", 20)
    pdf.multi_cell(0, 5, str(N_TRADE))
    pdf.set_xy(140, 190)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "WINNING RATE ")
    pdf.set_xy(175, 190)
    pdf.set_font(font, "B", 20)
    pdf.multi_cell(0, 5, str(round(WIN_RATE, 2)) + "%")
    pdf.set_xy(140, 200)
    pdf.set_font(font, "", 12)
    pdf.multi_cell(0, 5, "MAX DRAWDOWN ")
    pdf.set_xy(180, 200)
    pdf.set_font(font, "B", 20)
    pdf.multi_cell(0, 5, str(round(MAX_DRAWDOWN * 100)) + "%")

    pdf.image(return_path, x=10, y=150, w=120, h=90)
    pdf.image(leverage_path, x=10, y=245, w=120, h=50)
    pdf.image(orders_path, x=130, y=245, w=65, h=50)

    # SAVE
    pdf.output(path + "\\BACKTEST.pdf", "F")
    return