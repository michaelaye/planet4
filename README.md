# Planet Four

Software to support the analyis of planetfour data

## Data reduction

### Format
The target format is the Hierarchical Data Format (HDF) in version 5, a well established data format
with good reading routines for Python, Matlab and IDL.

### Parsing
The first step is just a straight forward parsing of the CSV output of the Mongo database dump.
While parsing, values of 'null' are being replaced by numpy.NaN.
I made the conscious decision to _NOT_ replace `None` in the `marking` column by NaN because that
detail is in itself useable data.

The `acquisition_date` column is currently being parsed to a python datetime, if that is a problem
for someone absolutely requiring it to be a string, I could make this optional.

### Filtering / Cleaning

### Reduction levels
I produce different versions of the reduced dataset, increasing in reduction, resulting in smaller
and faster to read files.

#### Level 1
All data is included apart from what was removed in above filtering step.
The product file name scheme is xxxxxxxx

#### Level 2
This product is reduced to the data records that are finished in Planet4 terms, which is currently
defined has having 30 individual analyses performed on a specific Planet4 subframe.

#### Level 3
This product is reduced further from Level 2 by only including data records with `markings!='None'`.
In other words, each data record of this data product has marking data in it.

## Clustering analysis
TBW