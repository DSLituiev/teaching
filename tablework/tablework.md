# Table work beyond Excel:
## Tools in Shell, Python, and R
Dima Lituiev, PhD, postdoc @ IHG, UCSF

---
# Why use scripting languages?

+ Scaling (big vs small)
+ Flexibility (trickier operations)
+ Reproducibility

---

# Problem size

## Number of entries:

        < 1000 -- small
    1e3 -- 1e8 -- medium
         > 1e8 -- big

# Programmatic tools for tables

|Standard terminal tools: |		grep, awk, sed, tr, head, tail, cat |
| --- | --- |
|R:| data.frame, data.table, dplyr + tydr |
|Python:| pandas |
|Advanced special tools: | SQL (general), bedtools (genomics)

--- 
# Table manipulation in terminal

Fast and simple

+ Estimate size / number of lines of the file: `wc -l`
+ Split / Subset a file into parts 
    `head, tail (-n #lines)`
+ pattern matching [`grep 'pattern' file`]
+ Combine files row-wise with same column order [`cat`]
+ Text replacement [`sed, tr`]
+ Column subsetting [`cut`] and reordering [`awk`]

---

# Table manipulation in R or Python

## Simple and advanced operations on medium-sized files

+ Subset rows and columns by a mask
+ Transform data entrywise (replace text, take sqrt etc.)
+ Pivot/stack and unstack tables
+ Aggregate rows by a key [take mean, variance etc]
+ Combine tables columns-wise by an index column

---

# Subset rows

```
data
   A   B      C
0  foo small  1
1  foo large  2
2  foo large  2
5  bar large  4
6  bar small  5
```

```
mask = data["B"] == "small"
data[mask]

   A   B      C
0  foo small  1
6  bar small  5

data.loc[mask, "C"] = 10

data
   A   B      C
0  foo small  10
1  foo large  2
...
```


---
# Pivot
```
data
   A   B      C
0  foo small  1
1  foo large  2
2  foo large  2
5  bar large  4
6  bar small  5
```

```
pd.pivot_table(data, values='C', index='A',
               columns=['B'], aggfunc=np.sum)

      small  large
foo   1      4
      6      NaN
bar   5      4
      6      7
```

---

# Unstack (inverse of pivoting)

```
data
   A     B    C
A  1.0  0.5  0.2
B  0.5  1.0  0.3
C  0.2  0.3  1.0

pd.unstack(data)
A  A   1.0  
A  B   0.5  
A  C   0.2  
B  B   1.0
...
```

### Group and Aggregate

```
pandas:
data.groupby("index").agg(funct)
dplyr:
data %>% 
	group_by(index) %>%
	summarise(field1 = funct(),
    		  field2 = funct() )
```

---

# Column-wise join / merge

Join two tables column-wise based on a pair of index columns
Example: join a gene expression table with a table of genomic locations of genes

Several flavours:
+ Inner join: only unique matching entries
+ Outer join: all matching entries, indices may be non-unique)
+ Cross join: all combinations, indices do not have to match

```
# pandas:
pd.concat([t1, t2],axis=1)
pd.merge(t1, t2, ...)

# dplyr:
inner_join(t1, t2, by = col)
... 
```

---
# R vs Python
It is not quite politically correct question, but â€¦
 
## Pros of pandas:
+ Allows to create and retrieve data by named hierarchical index and hierarchical columns 
+ Built-in grouping and merge capabilities (doable in R with dplyr)
 
---
# If not sure how to proceed ...

<p align="center">
  <img src=
https://s-media-cache-ak0.pinimg.com/736x/92/5a/48/925a487c310d90ff0b712816e21b2fbf.jpg alt="small_car" width="40%" height="40%">
</p>

<p align="center" fontsize=26>
<font size="3">
source: Daisy Chaussee, LA Times
</font>
</p>

---
# If not sure how to proceed ...

+ Formulate what you want to get
+ Try it on a small test table
+ Keep a piece of table before your very eyes
+ Paste a line as a comment
+ Split terminal window
+ Use a search engine and stackoverflow.com 
+ Breath and exercise

---

# Try on a small table!

 
<p align="center">
  <img src=
https://il9.picdn.net/shutterstock/videos/4530842/thumb/6.jpg alt="small_car" width="40%" height="40%">
</p>

Use head command to generate 
a small test table of 100 -- 1000 entries

```
R>     head(data, numlines)
pd>>>  data.head(numlines)
bash$  head -n $numlines data.tab
```

Or pick random lines:

```
dplyr> tbl_df(iris) %>% sample_n(numlines)
pd>>> data.sample(numlines)
bash$ shuf data | head -n $numlines
```

---

# Where to go next?

+ [pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
+ [dplyr / CRAN](https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html)
+ [dplyr cheetsheet](http://stat545.com/bit001_dplyr-cheatsheet.html)
+ [bedtools](http://bedtools.readthedocs.io/en/latest/content/tools/intersect.html)


