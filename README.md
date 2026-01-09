# Finding the correlation

## 📌 Problem Description

Given a set of data points:

1. Extract the ((x, y)) coordinates of each point
2. Calculate **Pearson’s correlation coefficient**
3. Interpret the result
4. Visualize the data distribution

---

## 📊 Pearson Correlation Formula

[
r =
\frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}
]

Where:

* (x_i, y_i) — individual data points
* (\bar{x}, \bar{y}) — mean values
* (n) — number of observations

---

## 📁 Project Structure

```
.
├── correlation.py     # Main Python script
├── README.md           # Project documentation
└── scatter.png         # Generated scatter plot (optional)
```

---

## 🧮 Data Used

The following data points were extracted from the graph:

| Point |   x |   y |
| ----: | --: | --: |
|     1 | -10 | -10 |
|     2 |  -5 |  -5 |
|     3 |  -3 |  -1 |
|     4 |  -5 |   2 |
|     5 |  -1 |   1 |
|     6 |   3 |   1 |
|     7 |   1 |  -2 |
|     8 |   5 |  -3 |
|     9 |   7 |  -2 |

---

## Run Script


Run the script:

```bash
python correlation.py
```

This version also generates a scatter plot saved as `scatter.png`.

---

## Output

```
Data points: [(-10, -10), (-5, -5), (-3, -1), (-5, 2), (-1, 1), (3, 1), (1, -2), (5, -3), (7, -2)]
Pearson r (manual): 0.448991
Pearson r (NumPy) : 0.448991
```

---

## Scatter Plot Visualization

The following scatter plot visualizes the distribution of the extracted data points
used to calculate Pearson’s correlation coefficient.

![Scatter Plot of Data Points](Finding the correlation/scatter.png)

