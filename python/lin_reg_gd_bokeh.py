#run from terminal with: 
#$ bokeh serve --show lin_reg_gd_bokeh.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import random

from bokeh.core.properties import field
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (
    ColumnDataSource, HoverTool, SingleIntervalTicker, Slider, Button, Label,
    CategoricalColorMapper,
)
from bokeh.palettes import Spectral6
from bokeh.plotting import figure


def cost_func(X, Y, B):
    m = len(Y)
    theta = np.sum((X.dot(B) - Y)**2) / (2*m)
    return theta

def gradient_descent(X, Y, B, alpha, n_iter):
    cost_history = []
    beta_history = []
    beta_history.append(B)
    m = len(Y)
    for i in range(n_iter):
        # predicted value
        y_hat = np.dot(X,B)
        # error
        loss = y_hat - Y
        # Gradient Calculation
        gradient = np.dot(X.T, loss)/m
        # adjust value of beta(s) based on gradient
        B = B - alpha * gradient
        beta_history.append(B)
        # Calculate cost
        cost = cost_func(X, Y, B)
        cost_history.append(cost)
    return B, cost_history, beta_history

# Create data
x = np.arange(0.0, 20, 0.1)
y = [i*0.75 + np.random.normal() for i in x]
# add column of 1s for intercept evaluation
x0 = np.ones(len(x))
predictor = np.array([x0, x]).T

# set learning rate and initial values for slope and intercept
beta_init = np.array([0, 0])
alpha = 0.01
iterations = 25
# call function to get beta values and cost fun values for each iteration
beta, cost_hist, beta_hist = gradient_descent(predictor, y, beta_init, alpha, iterations)

###
#Create bokeh plot that shows how the fitted line changes over the iterations
###

# bokeh code used was adapted from the famous gapminder example:
# https://github.com/bokeh/bokeh/blob/master/examples/app/gapminder/main.py

data_bokeh = {}
iter_list = list(range(iterations))

for i, b in enumerate(beta_hist):
    y_pred = np.dot(predictor, b)
    df = pd.DataFrame({'x':x, 'y':y_pred})
    data_bokeh[i]= df.to_dict('series')

source = ColumnDataSource(data=data_bokeh[iter_list[0]])

plot = figure(x_range=(-2, 22), y_range=(-2, 22), 
              title='Linear Regression Batch Gradient Descent', plot_height=300)
plot.xaxis.ticker = SingleIntervalTicker(interval=1)
plot.xaxis.axis_label = "X"
plot.yaxis.ticker = SingleIntervalTicker(interval=1)
plot.yaxis.axis_label = "Y"

plot.line(
    x='x',
    y='y',
    source=source,
    line_color=Spectral6[4],
    line_width=1.5,
)
plot.circle(
    x=x,
    y=y,
    size=7,
    fill_color=Spectral6[0],
    fill_alpha=0.8,
    line_color=Spectral6[0],
    line_width=0.5,
    line_alpha=0.5,
)

def slider_update(attrname, old, new):
    position = slider.value
    source.data = data_bokeh[position]

slider = Slider(start=iter_list[0], end=iter_list[-1], value=iter_list[0],
                 step=1, title="iteration")
slider.on_change('value', slider_update)

def animate_update():
    iter_n = slider.value + 1
    if iter_n > iter_list[-1]:
        iter_n = iter_list[0]
    slider.value = iter_n

def animate():
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(animate_update, 500)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(animate_update)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot],
    [slider, button],
], sizing_mode='scale_width')

curdoc().add_root(layout)
curdoc().title = "Teston"
