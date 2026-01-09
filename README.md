# Finding the correlation

## 📌 Problem Description

Given a set of data points:

1. Extract the ((x, y)) coordinates of each point
2. Calculate **Pearson’s correlation coefficient**
3. Interpret the result
4. Visualize the data distribution

---

## 📐 Pearson Correlation Coefficient

The Pearson correlation coefficient is defined as:

r = Σ((xᵢ − x̄)(yᵢ − ȳ)) / √( Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)² )

where:
- xᵢ, yᵢ are individual data points
- x̄, ȳ are the mean values
- Σ denotes summation
---

## 📁 Project Structure

```
.
├──Finding the correlation
    ├── correlation.py     # Main Python script
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

![Scatter Plot of Data Points](Finding%20the%20correlation/scatter.png)

