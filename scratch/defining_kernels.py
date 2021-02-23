from inspect import signature

def F(v1,v2,v3, cell=9, phi=0, region=0, c=20):
    """
    """
    return v1 + v2 + v3 - cell + phi - region + c

cell = 1
phi = 2
region = 3
dofs = 4

args = (1,2,3)
pkwargs = {'region':1}
print(F(*args, phi=2, **pkwargs))


kwargs = {'cell':1, 'phi':2, 'region':3, 'dofs':2}

for kwarg in kwargs:
    print(kwarg)
    
print(kwargs)
kwargs.pop('dofs')
print(kwargs)

s = signature(F)
ba = s.bind_partial(**kwargs)
print(ba.args, ba.kwargs)
ba.arguments['cell'] = -1

print(ba.args, ba.kwargs)


print(F(*args,**kwargs,c=-2))

print(ba.kwargs)

for arg in ['v1','v2','v3']:
    print('arg',arg, arg in s.parameters)

pars = s.parameters
   
g = lambda args, kwargs: F(*args, **kwargs)
kwargs = {'phi': None, 'region': None}
print(F(1,2,3,cell=9, **kwargs))

def g(F, *args, **kwargs):
    return 

print(g(1,2,3,cell=9))

for par in s.parameters.values():
    print('--', par.default is par.empty)
    
print('dofs' in s.parameters)
b = s.bind(1,2,3,cell=9)
print(b.arguments)
print('v1' in b.arguments)
#print('dofs' in b.kwargs)
print(F(*b.args, **b.kwargs))

