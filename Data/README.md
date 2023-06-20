
# Dataset of multi-type GCT flows

### Raw Dataset
The original multi-type GCT flows data, including Vehicle (i.e., `Vehicle_All.csv`), Pedestrian (i.e.,`Pedestrian_All.csv`), and Stationary (i.e.,`Stationary_All.csv`), are available at
```
../Data
```

These files are shared with a similar format.

Here is an example from `Vehicle_All.csv`:

|                     | Road ID 1 | Road ID 6 | Road ID 11 | ... | Road ID 202202 |
|:-------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 2022/08/28 00:05    |   8.0        |   3.0        |   12.0        |    ...         |    12.0         |
|         ...         |    ...         |    ...         |    ...         |    ...         |    ...         |
| 2022/09/14 17:00 |   62.0        |   10.0        |   35.0        |    ...         |    36.0         |
| 2022/09/14 17:05 |   127.0        |   12.0        |   34.0        |    ...         |    66.0         |
| 2022/09/14 17:10 |   82.0        |   12.0         |   36.0        |    ...         |    44.0         |
|         ...         |    ...         |    ...         |    ...         |    ...         |    ...         |
| 2022-03-31 23:55:00 |   8.0        |   0.0        |   7.0        |    ...         |    1.0         |



### Graph Construction

#### How to Create Graph Construction

As the implementation is based on pre-calculated distances between road segments, we provide the following files:
- `coordinates.csv`: Contains road segment IDs and their GPS coordinates.
- `distances.txt`: Contains road section distances.


Run the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py) to generate the Graph Structure used in the experiments.

#### Graph Construction Provided
The processed Graph Structure of Road Segment Network, i.e., `adj_mat_2022_hsin_21_locs.pkl`, are also available at:
```
https://drive.google.com/drive/folders/1mOuSkPpbGeq_Q-jVwKaq_qk-YU_mlwn4?usp=sharing
```
