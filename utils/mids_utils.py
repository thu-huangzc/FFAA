
def flatten_tuple_list(tuple_list):
    new_list = []
    for col in range(len(tuple_list[0])):
        for row in range(len(tuple_list)):
            new_list.append(tuple_list[row][col])
    return new_list
