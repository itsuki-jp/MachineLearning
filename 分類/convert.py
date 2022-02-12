# setosa-versicolor.txtの最後の行を数字に変換するためだけのやつ
reading_file = open('setosa-versicolor.txt', 'r')
datalist = reading_file.readlines()
data = []
for i in datalist:
    line = i.split(",")
    y = 1 if line[-1] == "Iris-versicolor\n" else 0
    temp = f'{",".join(line[:-1])},{y}\n'
    data.append(temp)
reading_file.close()

writing_file = open("data.txt", "w")
writing_file.writelines(data)
writing_file.close()
print()
