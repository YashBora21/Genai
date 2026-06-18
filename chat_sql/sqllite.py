import sqlite3
connection=sqlite3.connect('studentdb')
cursor=connection.cursor()

##table creation
table_info="""
CREATE TABLE IF NOT EXISTS students(name VARCHAR(25),class varchar(25),section varchar(25),marks int)
"""

cursor.execute(table_info)

cursor.execute("""
INSERT INTO students VALUES
('Alice', '10', 'A', 85),
('Bob', '10', 'B', 78),
('Charlie', '9', 'A', 92),
('David', '9', 'C', 67),
('Emma', '10', 'A', 88),
('Frank', '11', 'B', 74),
('Grace', '11', 'A', 95),
('Henry', '12', 'C', 81),
('Isabella', '12', 'B', 89),
('Jack', '10', 'C', 72),
('Kate', '9', 'B', 90),
('Liam', '11', 'A', 84),
('Mia', '12', 'A', 93),
('Noah', '10', 'B', 76),
('Olivia', '9', 'C', 87)
""")

connection.commit()

cursor.execute("SELECT * FROM students")
connection.commit()

cursor.execute("SELECT * FROM students")

data = cursor.fetchall()
print(data)

connection.close()