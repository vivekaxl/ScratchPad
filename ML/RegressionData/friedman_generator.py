from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3


def make_file(X, y, filename):
    content = []
    for i,j in zip(X, y):
        temp = list(i)
        temp.append(j)
        content.append(temp)
    import csv
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(content)


X, y = make_friedman3(1000, 10)
make_file(X, y, "friedman3.csv")
