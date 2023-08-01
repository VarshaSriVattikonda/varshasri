# get() getting data from dict using key
# keys()
# values()
# items()

d={
    'name':'python',
    'fees': '8000',
    'duration':'2months'
}

# get()
n = d.get('name')
print(n)

# keys()
for a in d.keys():
    print(a)

# values()
for a in d.values():
    print(a)

# items()
for a,b in d.items():
    print(a,b)
    # we use a,b because in a keys store and in b values store.

# delete ke liye del keyword
del d['duration']
print(d) 

# pop keyword is also use for delete but written the value
print(d.pop('fees'))
print(d)

# dict
d=dict(name='python',fees=8000)
print(d)

# update
d={
    'name':'python',
    'fees':'9000',
    'duration':'2months'

}
d.update({'fees':100000})
print(d)

# update second way
d['fees']=2000
print(d)

# clear
# d.clear()
# print(d)

# insert
d['desc'] = "This is python"
print(d)
# if the key is already exist in dict so it update value of that key.
