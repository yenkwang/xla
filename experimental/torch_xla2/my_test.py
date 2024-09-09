import jax
import torch
import torch_xla2
import jax.numpy as jnp

#a = jnp.array([1.0, 2.0, 3.0, 4.0])
#b = a.reshape(2,2)
#c = jnp.linalg.slogdet(b)

#x = torch.tensor([])
#y = x.reshape(0,0)
#z = torch.slogdet(y)

torch.return_types.slogdet(sequence = (torch.tensor(1.), torch.tensor(0.)))

env = torch_xla2.default_env()
env.config.debug_print_each_op = True
env.config.debug_accuracy_for_each_op = True

with env:
  m = torch.tensor(-8.47)
  print(torch.median(m, 0, True))

  print('-'*10)

  x = torch.tensor([1.0, 2.0, 3.0, 4.0])
  y = x.reshape(2,2)
  print(torch.linalg.slogdet(y))
