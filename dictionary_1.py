#  dictionary - it is a collection which is unordered.In python dictionaries are written with curly brackets {}.

#  dictionary is key and value pair.

d={
    'name': 'Python',
    'fees':9000,
    'duration': '2Months'
}

# print(type(d)) # for checking that it is dict or not
# print(d)

f=d['fees']
# print(f)

# iterating
for n in d:
    print(n)
    print(d[n])