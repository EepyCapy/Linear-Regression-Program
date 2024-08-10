import dearpygui.dearpygui as dpg
import DearPyGui_DragAndDrop as dpg_dnd
import pandas as pd
from LinearRegression import *

dpg.create_context()
dpg_dnd.initialize()
dpg.create_viewport(title='Linear Regression', width=800, height=800)

tableData: pd.DataFrame = None
tableKeys: list[str] = []
availableVariables: list[str] = []
independentVariables: list[str] = []
dependentVariable: str = ''
linearRegressor: LinearRegression = None

def createTable():
    global tableData, tableKeys
    with dpg.table(tag='table', parent='dataWindow'):
        # Columns
        for key in tableKeys:
            dpg.add_table_column(label=key,)
        # Rows
        for i in range(len(tableData[tableKeys[0]])):
            with dpg.table_row():
               for key in tableKeys:
                   dpg.add_text(f'{tableData[key][i]}')

def isNum(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False

def getValidKeys():
    global tableData, tableKeys, availableVariables
    for key in tableKeys:
        if isNum(f'{tableData[key][1]}'):
            availableVariables.append(key)

def popupButtons2(sender, app_data, user_data):
    global dependentVariable
    print(user_data)
    dependentVariable = user_data
    availableVariables.remove(user_data)
    print(dependentVariable)

def popupButtons1(sender, app_data, user_data):
    global independentVariables
    independentVariables.append(user_data)
    availableVariables.remove(user_data)
    print(independentVariables)

def createPopUp1():
    global availableVariables
    with dpg.popup(parent='addVar', tag='VarPopUp', mousebutton=dpg.mvMouseButton_Left, modal=False):
            #dpg.add_text(label='---------')
            for var in availableVariables:
                dpg.add_button(label=var, parent='VarPopUp', callback=popupButtons1, user_data=var)
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("VarPopUp", show=False))

def createPopUp2():
    global availableVariables
    with dpg.popup(parent='addInVar', tag='VarPopUp2', mousebutton=dpg.mvMouseButton_Left, modal=False):
            #dpg.add_text(label='---------')
            for var in availableVariables:
                dpg.add_button(label=var, parent='VarPopUp2', callback=popupButtons2, user_data=var)
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("VarPopUp2", show=False))

def createPlot():
    global linearRegressor
    x = linearRegressor.xData.flatten().tolist()
    y = linearRegressor.yData.flatten().tolist()
    yPred = linearRegressor.plotPrediction().flatten().tolist()

    #with dpg.theme(tag='plotTheme'):
    #   with dpg.theme_component(dpg.mvScatterSeries):
    #        #dpg.add_theme_color(dpg.mvPlotCol_Line, (60, 150, 200), category=dpg.mvThemeCat_Plots)
    #        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Square, category=dpg.mvThemeCat_Plots)

    with dpg.plot(parent='graphWindow', width=800, height=700):
        dpg.add_plot_axis(dpg.mvXAxis, label='x')
        dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis')
        dpg.add_scatter_series(x, y, parent='y_axis', tag='dataPoints')
        dpg.add_line_series(x, yPred, parent='y_axis')

    

def analize(sender, app_data):
    global tableData, independentVariables, dependentVariable, linearRegressor
    yData = tableData[dependentVariable].to_numpy()
    xData = tableData[independentVariables].to_numpy()
    linearRegressor = LinearRegression(xData, yData)
    linearRegressor.multidLS()
    print(linearRegressor.parameters)
    createPlot()

def drop(data, keys):
    global tableData, tableKeys, availableVariables
    tableData = pd.read_csv(data[0])
    tableKeys = tableData.keys()
    getValidKeys()
    dpg.delete_item('table')
    dpg.delete_item('VarPopUp')
    createPopUp1()
    createPopUp2()
    createTable()

with dpg.window(label='Analysis Window', tag='analWindow', width=600, height=200, pos=(0, 0), no_resize=True, no_move=True, no_close=True):
    dpg.add_button(label='Add Independent Variable (+)', tag='addVar')
    dpg.add_button(label='Choose Dependent Variable (+)', tag='addInVar')
    dpg.add_button(label='Analize', tag='analButton', callback=analize)

with dpg.window(label='Graph', tag='graphWindow', width=600, height=200, pos=(600, 0)):
    dpg.add_button(label='Button')

with dpg.window(label='Data Window', tag='dataWindow', width=600, height=700, pos=(0, 500)):
    dpg.add_button(label='Button')

dpg_dnd.set_drop(drop)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
