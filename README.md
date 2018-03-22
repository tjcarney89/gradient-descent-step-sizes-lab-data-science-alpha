
### Introduction

In this lab, we'll practice applying gradient descent.  As we know gradient descent begins with an initial regression line, and moves to a "best fit" regression line by changing values of $m$ and $b$ and evaluating the RSS.  So far, we have illustrated this technique by changing the values of $b$ and evaluating the RSS.  In this lab, we will work through applying our technique by changing the value of $m$.  We'll have access to our [graph library](https://github.com/learn-co-curriculum/gradient-descent-step-sizes-lab/blob/master/graph.py), [linear equations library](https://github.com/learn-co-curriculum/gradient-descent-step-sizes-lab/blob/master/linear_equations.py), and [error library](https://github.com/learn-co-curriculum/gradient-descent-step-sizes-lab/blob/master/error.py) in completing this lab.

### Setting up our initial regression line

Once again, we'll take take a look at revenues of movies to predict revenue. 


```python
first_show = {'budget': 100, 'revenue': 275}
second_show = {'budget': 200, 'revenue': 300}
third_show = {'budget': 400, 'revenue': 700}

shows = [first_show, second_show, third_show]
```

Using our data, and our `build_regression_line`, we get some values for an initial regression line.


```python
from linear_equations import build_regression_line

budgets = list(map(lambda show: show['budget'], shows))
revenues = list(map(lambda show: show['revenue'], shows))

build_regression_line(budgets, revenues)
```




    {'b': 133.33333333333326, 'm': 1.4166666666666667}




```python
def regression_line(x):
    return 1.417*x + 133.33
```

Now using the `residual_sum_squares`, function, we calculate the RSS.  Let's take another look at it here: 

```python 
def residual_sum_squares(x_values, y_values, m, b):
    return sum(squared_errors(x_values, y_values, m, b)) 
```

### Building a cost curve

Now let's use the RSS to build a cost curve.  Keeping the $b$ value fixed at $133.33$, write a function called `rss_values` that takes `x_values` and `y_values` to pass through our dataset, and various values of $m$, an initial $b$ value.  It outputs a dictionary with keys of `m_values` and `rss_values`, with each key pointing to a list of the corresponding values.


```python
from error import residual_sum_squares
def rss_values(x_values, y_values, m_values, b):
    pass
```


```python
budgets = list(map(lambda show: show['budget'] ,shows))
revenues = list(map(lambda show: show['revenue'] ,shows))
initial_m_values = list(range(8, 19, 1))
scaled_m_values = list(map(lambda initial_m_value: initial_m_value/10,initial_m_values))
b_value = 133.33
rss_values(budgets, revenues, scaled_m_values, b_value)

# {'m_values': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
#  'rss_values': [64693.76669999998,
#   45559.96669999998,
#   30626.166699999987,
#   19892.36669999999,
#   13358.5667,
#   11024.766700000004,
#   12890.96670000001,
#   18957.166700000016,
#   29223.36670000002,
#   43689.566700000025,
#   62355.76670000004]}

```

Plotly provides for us a table chart, and we can pass the values generated from our `rss_values` function to create a table.


```python
from plotly.offline import iplot, init_notebook_mode
from graph import plot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

def plot_table(headers, columns):
    trace_cost_chart = go.Table(
        header=dict(values=headers,
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'] * 5),
        cells=dict(values=columns,
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['left'] * 5))
    plot([trace_cost_chart])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
cost_chart = rss_values(budgets, revenues, scaled_m_values, b_value) or {}
column_values = list(cost_chart.values())
plot_table(headers = ['M values', 'RSS values'], columns=column_values)
```


<div id="9ac67c50-26a0-4881-be25-7fe41ffd5388" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("9ac67c50-26a0-4881-be25-7fe41ffd5388", [{"type": "table", "header": {"values": ["M values", "RSS values"], "line": {"color": "#7D7F80"}, "fill": {"color": "#a1c3d1"}, "align": ["left", "left", "left", "left", "left"]}, "cells": {"values": [], "line": {"color": "#7D7F80"}, "fill": {"color": "#EDFAFF"}, "align": ["left", "left", "left", "left", "left"]}}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


And let's plot this out using a a line chart.


```python
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from graph import plot, trace_values

initial_m_values = list(range(8, 19, 1))
scaled_m_values = list(map(lambda initial_m_value: initial_m_value/10,initial_m_values))
cost_values = rss_values(budgets, revenues, scaled_m_values, 133.33)
rss_trace = trace_values(cost_values['m_values'], cost_values['rss_values'], mode = 'line')
plot([rss_trace])
```

### Changing our step size

In this section, we'll work up to building a gradient descent function that automatically changes our step size.  To get you started, we'll provide a function called `slope_at` that calculates the slope of the cost curve at a given point.  Here it is in action:


```python
from helper import slope_at
slope_at(budgets, revenues, 1.7, 133.33333333333326)
```




    {'m': 1.7, 'slope': 165687.66666649026}




```python
slope_at(budgets, revenues, 1.3, 133.33333333333326)
```




    {'m': 1.3, 'slope': -2312.3333333387563}



As you can see, it seems pretty accurate.  When the curve is steeper at $m = 1.7$, the slope is over 165,000.  When we near our flatline of our cost curve with our $m = 1.3$, our slope has a much smaller magnitude with a value of $-2312.3$. 

Ok, now we're ready to write a function called `updated_m`.  The function will allow us to move along our cost curve more efficiently, by taking a more efficient step size.  The `updated_m` function takes as arguments an initial value of $m$, a learning rate, and the `slope` of the cost curve at that value of $m$.  It returns an integer that equals the next value of `m`. 


```python
from error import residual_sum_squares

def updated_m(m, learning_rate, cost_curve_slope):
    pass
```


```python
current_slope = slope_at(budgets, revenues, 1.7, 133.33333333333326)['slope']
updated_m(1.7, .000001, current_slope)
# 1.5343123333335096

current_slope = slope_at(budgets, revenues, 1.534, 133.33333333333326)['slope']
updated_m(1.534, .000001, current_slope)
# 1.43803233333338

current_slope = slope_at(budgets, revenues, 1.438, 133.33333333333326)['slope']
updated_m(1.438, .000001, current_slope)
# 1.3823523333332086
```

Take a careful look at how we use the `updated_m` function.  By using our updated value of $m$ we are quickly converging towards an optimal value of $m$.   

Now let's write another function called `gradient_descent_values`.  Similar to our `rss_values` function it outputs keys of `m_values` and `rss_values` each returning a list of corresponding values.  However, the inputs are now `x_values`, `y_values`, `number_of_steps`, and `b`.  The `number_of_steps` arguments represents the number of steps the function will take before the function stops.  It is the number of steps that are taken.


```python
def gradient_descent(x_values, y_values, steps, b, learning_rate, current_m):
    pass
```


```python
descent_steps = gradient_descent(budgets, revenues, 12, 133.33, learning_rate = .000001, current_m = 0) or []
```


```python
m_values = list(map(lambda step: step['m'],descent_steps))
rss_result_values = list(map(lambda step: step['rss'], descent_steps))
text_values = list(map(lambda step: 'cost curve slope: ' + str(step['slope']), descent_steps))
gradient_trace = trace_values(m_values, rss_result_values, text=text_values)
plot([gradient_trace])
```


<div id="94e846e3-e299-415e-8adb-cf9ef7d2cb05" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("94e846e3-e299-415e-8adb-cf9ef7d2cb05", [{"x": [], "y": [], "mode": "markers", "name": "data", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


Taking a look at a plot of our trace, you can get a nice visualization of how our gradient descent function works.  It starts far away with $m = 0$, and the step size is relatively large, as is the slope of the cost curve.  As the $m$ value updates such that it approaches a minimumm of the RSS, and the slope of the cost curve decreases, the size of the step also decreases.     
