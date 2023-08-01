# nested dict - it means putting a dictionary inside another dictionary.
# it is a collection of dictionaries into single dictionary.

course={
    'php':{'duration':'2 Months','fees':'15000'},
    'python':{'duration':'2 Months','fees':'12000'},
    'java':{'duration':'2 Months','fees':'13000'}
}
print(course)
# update
course['java']['fees'] = 20000
print(course['php']['fees'])
print(course['java']['fees'])

for k,v in course.items():
    print(k,v)
# here k store sub name and v store content
    print(k,v['duration'],v['fees'])
