def tree(x, af, bf):
  return np.sin(af*(x+bf))

def fit(x, const):

  return np.sqrt((tree(x, const_ref)-tree(x, const))**2)

def grad_desc(input, const_start, it, fun, x, alpha):

  new_fit = fit(input, float(a_start.name), float(b_start.name))
  pdaf = lambdify(x, diff(fun, a_start), "numpy")
  pda = pdaf(input)
  pdbf = lambdify(x, diff(fun, b_start), "numpy")
  pdb = pdbf(input)
  new_a = float(a_start.name) - alpha*(new_fit*pda)
  new_b = float(b_start.name) - alpha*(new_fit*pdb)

  for _ in range(it):

    a, b = symbols('{} {}'.format(new_a, new_b))
    f = sin(a * (b + x))
    new_fit = fit(input, float(a.name), float(b.name))
    pdaf = lambdify(x, diff(fun, a), "numpy")
    pda = pdaf(input)
    pdbf = lambdify(x, diff(fun, b), "numpy")
    pdb = pdbf(input)
    new_a = float(a.name) - alpha*(new_fit*pda)
    new_b = float(b.name) - alpha*(new_fit*pdb)

    del a, b, fun
  return new_a, new_b

alpha = 0.21
start_a = 2.5
start_b = 4
x,a,b = symbols('x {} {}'.format(start_a, start_b))
f = sin(a*(b+x))
out1, out2 = grad_desc(1.5, a, b, 200, f, x, alpha)
print(out1, out2)