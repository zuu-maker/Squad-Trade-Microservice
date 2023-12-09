def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


columns = [
    "Name",
    "Team",
    "Pos",
    "G",
    "Last year",
    "COMP",
    "PATT",
    "PYDS",
    "PTD",
    "INT",
    "RATT",
    "RYDS",
    "RTD",
    "REC",
    "TAR",
    "REYDS",
    "RETD",
    "TOT",
    "LOST"
]

arranged_cols_1 = [
    "Name",
    "Team",
    "Pos",
    "COMP",
    "PATT",
    "PYDS",
    "PTD",
    "INT",
    "RATT",
    "RYDS",
    "RTD",
    "REC",
    "TAR",
    "REYDS",
    "RETD",
    "TOT",
    "LOST",
    "G",
    "Last year",
    "Two years ago",
]

arranged_cols_2 = [
    "Name",
    "Team",
    "Pos",
    "COMP",
    "PATT",
    "PYDS",
    "PTD",
    "INT",
    "RATT",
    "RYDS",
    "RTD",
    "REC",
    "TAR",
    "REYDS",
    "RETD",
    "TOT",
    "LOST",
    "G",
    "Last year",
    "Two years ago",
    "Value",
]

data_columns = ['Name', 'Team', 'Pos', 'COMP', 'PATT', 'PYDS', 'PTD', 'INT', 'RATT',
                'RYDS', 'RTD', 'REC', 'TAR', 'REYDS', 'RETD', 'TOT', 'LOST', 'G',
                'Last year', 'Two years ago']
