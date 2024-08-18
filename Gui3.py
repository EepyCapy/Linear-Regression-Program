import dearpygui.dearpygui as dpg
import DearPyGui_DragAndDrop as dpg_dnd
import pandas as pd
from LinearRegression import *

dpg.create_context()
dpg_dnd.initialize()
dpg.create_viewport(title='Linear Regression', width=1800, height=1000)

tableData: pd.DataFrame = None
tableKeys: list[str] = []
variables: list[str] = []
independentVariables: list[str] = []
dependentVariable: str = ''
linearModel: LinearRegression = None

def deletePreviousWidgets():
    dpg.delete_item('analButton')
    dpg.delete_item('table')
    dpg.delete_item('selectIndVars')
    dpg.delete_item('selectDepVar')
    dpg.delete_item('text1')
    dpg.delete_item('text2')
    for var in variables:
        dpg.delete_item('ind_'+var)
        dpg.delete_item('dep_'+var)



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
    global tableData, tableKeys, variables
    variables = []
    for key in tableKeys:
        if isNum(f'{tableData[key][1]}'):
            variables.append(key)

def createVariableSelection():
    with dpg.table(parent='analWindow', tag='selectIndVars'):
        dpg.add_table_column(label='Select Independent Variables')
        for var in variables:
            with dpg.table_row():
                dpg.add_selectable(label=var, tag='ind_'+var, callback=updateIndependentVariables, user_data='ind_'+var)
    with dpg.table(parent='analWindow', tag='selectDepVar'):
        dpg.add_table_column(label='Select Dependent Variable')
        for var in variables:
            with dpg.table_row():
                dpg.add_selectable(label=var, tag='dep_'+var, callback=updateDependetVariable, user_data='dep_'+var)

def updateIndependentVariables(sender, app_data, user_data):
    global independentVariables, dependentVariable
    if len(independentVariables) == len(variables) - 1:
        print('At least one varibla must be dependent!')
    elif dpg.get_value(user_data):
        independentVariables.append(user_data.replace('ind_', ''))
    else:
        independentVariables.remove(user_data.replace('ind_', ''))
    if dependentVariable in independentVariables:
        print('Careful: The last independent variable you chose is already the dependent variable')
    print(independentVariables)

def updateDependetVariable(sender, app_data, user_data):
    global independentVariables, dependentVariable
    for var in variables:
        dpg.set_value('dep_'+var, False)
    dpg.set_value(user_data, True)
    dependentVariable = user_data.replace('dep_', '')
    if dependentVariable in independentVariables:
        print('Careful: Your chosen dependent variable is already an indendent variable')
    print(dependentVariable)

def createLinearModel():
    global tableData, independentVariables, dependentVariable, linearModel
    yData = tableData[dependentVariable].to_numpy()
    xData = tableData[independentVariables].to_numpy()
    linearModel = LinearRegression(xData, yData)
    print(linearModel)

def refreshPlot():
    global independentVariables
    dpg.delete_item('indVarList')
    dpg.delete_item('predPlot')
    dpg.delete_item('dataPoints')
    dpg.delete_item('x_axis')
    dpg.delete_item('y_axis')
    dpg.delete_item('graph')
    dpg.delete_item('graphWindow')
    with dpg.window(label='Graph', tag='graphWindow', width=900, height=900, pos=(600, 0), no_resize=True, no_move=True, no_close=True):
        with dpg.plot(parent='graphWindow', width=800, height=800):
            dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis')
            dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis')
        dpg.add_listbox(items=independentVariables, parent='graphWindow', tag='indVarList')

def scatterPlot(sender, add_data, user_data):
    global linearModel
    createLinearModel()
    x = linearModel.xData[:, user_data].flatten().tolist()
    y = linearModel.yData.flatten().tolist()
    refreshPlot()
    dpg.add_scatter_series(x, y, parent='y_axis', tag='dataPoints')

def predictionPlot(sender, app_data, user_data):
    global linearModel
    scatterPlot(sender, app_data, user_data)
    linearModel.multidLS()
    print(linearModel.parameters)
    x = linearModel.xData[:, user_data].flatten().tolist()
    yPred = linearModel.predictedValues(user_data).flatten().tolist()
    print(len(x), len(yPred))
    dpg.add_line_series(x, yPred, parent='y_axis', tag='predPlot')

def createPlot():
    global tableData, independentVariables, dependentVariable, linearModel
    x = linearModel.xData.flatten().tolist()
    y = linearModel.yData.flatten().tolist()
    yPred = linearModel.predictedValues().flatten().tolist()

def analize(sender, app_data):
    global tableData, independentVariables, dependentVariable, linearModel
    yData = tableData[dependentVariable].to_numpy()
    xData = tableData[independentVariables].to_numpy()
    linearModel = LinearRegression(xData, yData)
    linearModel.multidLS()
    print(linearModel.parameters)
    createPlot()

def drop(data, key):
    global tableData, tableKeys, variables
    deletePreviousWidgets()
    tableData = pd.read_csv(data[0])
    tableKeys = tableData.keys()
    getValidKeys()
    dpg.add_button(label='Plot Data', tag='plotData', parent='analWindow', callback=scatterPlot, user_data=0)
    dpg.add_button(label='Fit Data', tag='analButton', parent='analWindow', callback=predictionPlot, user_data=0)
    createVariableSelection()
    createTable()

with dpg.window(label='Analysis Window', tag='analWindow', width=600, height=600, pos=(0, 0), no_resize=True, no_move=True, no_close=True):
   pass

with dpg.window(label='Graph', tag='graphWindow', width=900, height=900, pos=(600, 0), no_resize=True, no_move=True, no_close=True):
    with dpg.plot(parent='graphWindow', tag='graph', width=850, height=850):
        dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis')
        dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis')
    

with dpg.window(label='Data Window', tag='dataWindow', width=600, height=300, pos=(0, 600)):
    pass

dpg_dnd.set_drop(drop)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
