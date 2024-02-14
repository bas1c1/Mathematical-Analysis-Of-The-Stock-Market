import yfinance as yf
import pandas

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    data = data.reset_index()
    return data

def rsi(df, periods = 14, ema = True):
    close_delta = df.diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def calculate_atr(data, period=14):
    highs = data['High']
    lows = data['Low']
    closes = data['Close']
    
    trs = []
    for i in range(1, len(data)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range = max(high_low, high_close, low_close)
        trs.append(true_range)
    
    atr = np.mean(trs[:period])
    
    for i in range(period, len(trs)):
        atr = ((atr * (period - 1)) + trs[i]) / period
    
    return atr

def supertrend(data, period=14, multiplier=3.0):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    atr = calculate_atr(data, period)
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend = close.copy()
    supertrend[close > upper_band] = lower_band
    supertrend[close < lower_band] = upper_band
    
    return supertrend

def tsi(data, short_period=9, long_period=25):
    prices = data['Close']
    close_diff = prices.diff()
    close_diff = close_diff[~np.isnan(close_diff)]

    tsi = close_diff.ewm(span=short_period, adjust=False).mean() / \
            close_diff.ewm(span=long_period, adjust=False).mean()
    tsi = tsi / 100
    
    return tsi

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')

def draw_figure(canvas, figure):
   tkcanvas = FigureCanvasTkAgg(figure, canvas)
   tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=0)
   return tkcanvas

def calc_data(symbol, start_date, end_date):
	data = get_stock_data(symbol, start_date, end_date)
	data['MA'] = data['Close'].rolling(window=20).mean()
	data['RSI'] = rsi(data['Close'])
	data['supertrend'] = supertrend(data)
	data['tsi'] = tsi(data)
	k = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
	d = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
	macd = k - d
	macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
	macd_h = macd - macd_s
	data['macd'] = data.index.map(macd)
	data['macd_h'] = data.index.map(macd_h)
	data['macd_s'] = data.index.map(macd_s)
	return data

layout = [
			[sg.Text("Введите название компании (Например: INTC, MSFT)"), sg.InputText()],
			[sg.Text("Введите стартовую дату показа акций (Например: 2021-01-01)"), sg.InputText()],
			[sg.Text("Введите конечную дату показа акций (Например: 2024-01-01)"), sg.InputText()],
			[sg.Checkbox("RSI", key='rsi'), sg.Checkbox("MACD", key='macd'), sg.Checkbox("MA", key='ma'), sg.Checkbox("SuperTrend", key='supertrend'), sg.Checkbox("TSI", key='tsi')],
			[sg.Canvas(key='-CANVAS-')],
			[sg.Text("Введите за какой период показать скользящее среднее (Например: 20)"), sg.InputText()],
			[sg.Text("Введите диапозон цены"), sg.InputText(), sg.InputText()],
			[sg.Text("Введите диапозон даты"), sg.InputText(), sg.InputText()],
			[sg.Button('Update')]
			]
window = sg.Window('Анализ состояния рынка акции', layout, size=(1300, 1000), finalize=True, element_justification='center', font='Helvetica 18')

fig, ax = plt.subplots()
fig.set_figwidth(640)
fig.set_figheight(320)
fig.set_size_inches(12.0, 6.0)
draw_figure(window['-CANVAS-'].TKCanvas, fig)

def create_collection(df):
    l = len(df)

    grid = []
    height = []
    colors = []
    
    for i in range(l):
        if df.loc[i, 'Close'] > df.loc[i, 'Open']:
            grid.append([i, df.loc[i, 'Open']])
            height.append(df.loc[i, 'Close'] - df.loc[i, 'Open'])
            colors.append('green')
        elif df.loc[i, 'Close'] < df.loc[i, 'Open']:
            grid.append([i, df.loc[i, 'Close']])
            height.append(df.loc[i, 'Open'] - df.loc[i, 'Close'])
            colors.append('red')
        else:
            grid.append([i, df.loc[i, 'Close']])
            height.append(df.loc[i, 'Open'] - df.loc[i, 'Close'])
            colors.append('black')
    grid = np.array(grid)

    patches = []
    lines = []
    width = 0.5
    
    for i in range(l):
        
        rect = mpatches.Rectangle(grid[i]-[width/2, 0.0], width, height[i])
        patches.append(rect)
        line = mlines.Line2D([i, i], [df.loc[i, 'Low'], df.loc[i, 'High']], lw=0.75, color=colors[i])
        lines.append(line)

    collection = PatchCollection(patches, cmap=plt.cm.hsv)
    collection.set_facecolors([colors[i] for i in range(l)])
    collection.set_linewidth(1.0)
    collection.set_edgecolors([colors[i] for i in range(l)])
    
    return collection, lines

while True:
	event, values = window.read()
	if event == sg.WIN_CLOSED or event == 'Cancel':
		break

	data = calc_data(values[0], values[1], values[2])
	print(data)
	
	collection, lines = create_collection(data)
	ax.cla()

	ax.add_collection(collection)
	[ax.add_line(lines[i]) for i in range(len(data))]

	if values["ma"] == True:
		data['MA'] = data['Close'].rolling(window=int(values[3])).mean()
		plt.plot(data['MA'], label=f'Скользящее среднее {int(values[3])} дней')
	if values["rsi"] == True:
		plt.plot(data['RSI'], label='Индикатор RSI')
	if values["macd"] == True:
		plt.plot(data['macd'], label='Индикатор MACD')
	if values["supertrend"] == True:
		plt.plot(data['supertrend'], label='Индикатор SuperTrend')
	if values["tsi"] == True:
		plt.plot(data['tsi'], label='Индикатор TSI')
	plt.xlabel('Дата')
	plt.ylabel('Цена (USD)')
	plt.title('Анализ состояния рынка акции')
	#plt.axis([0, 100])
	if values[6] != "":
		ax.set_xlim(int(values[6]), int(values[7]))
	elif values["ma"] != True and values["rsi"] != True and values["macd"] != True and values["tsi"] != True and values["supertrend"] != True:
		ax.set_xlim(0, len(data))
	if values[4] != "":
		ax.set_ylim(int(values[4]), int(values[5]))
	elif values["ma"] != True and values["rsi"] != True and values["macd"] != True and values["tsi"] != True and values["supertrend"] != True:
		ax.set_ylim(min(list(map(int, data['Close']))), max(list(map(int, data['Close']))))
		#print(data['Close'][0])
		#print(type(data['Close'][0]))
		#print(min(list(map(int, data['Close']))))
	plt.legend()
	data.to_csv(f"{values[0]}.csv", sep=',', index=False, encoding='utf-8')
	fig.canvas.draw()

window.close()