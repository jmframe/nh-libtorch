def forward(self,
    input: Tensor,
    future: Tensor) -> Tensor:
  outputs = annotate(List[Tensor], [])
  if torch.eq(self.init_func, "ones"):
    h_t0 = torch.ones([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    c_t0 = torch.ones([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    h_t20 = torch.ones([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    c_t20 = torch.ones([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    output0 = torch.ones([1, 1], dtype=None, layout=None, device=None, pin_memory=None)
    c_t, c_t2, h_t, h_t2, output = c_t0, c_t20, h_t0, h_t20, output0
  else:
    h_t1 = torch.zeros([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    c_t1 = torch.zeros([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    h_t21 = torch.zeros([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    c_t21 = torch.zeros([torch.size(input, 0), 51], dtype=6, layout=None, device=None, pin_memory=None)
    output1 = torch.zeros([1, 1], dtype=None, layout=None, device=None, pin_memory=None)
    c_t, c_t2, h_t, h_t2, output = c_t1, c_t21, h_t1, h_t21, output1
  _0 = torch.chunk(input, torch.size(input, 1), 1)
  _1 = ops.prim.min([9223372036854775807, torch.len(_0)])
  outputs0 = outputs
  c_t3 = c_t
  c_t24 = c_t2
  h_t3 = h_t
  h_t24 = h_t2
  output2 = output
  for i in range(_1):
    input_t = _0[i]
    _2 = (self.lstm1).forward(input_t, (h_t3, c_t3), )
    h_t5, c_t5, = _2
    _3 = (self.lstm2).forward(h_t5, (h_t24, c_t24), )
    h_t26, c_t26, = _3
    output3 = (self.linear).forward(h_t26, )
    outputs0, c_t3, c_t24, h_t3, h_t24, output2 = torch.add_(outputs0, [output3]), c_t5, c_t26, h_t5, h_t26, output3
  outputs1 = outputs0
  c_t4 = c_t3
  c_t27 = c_t24
  h_t4 = h_t3
  h_t27 = h_t24
  output4 = output2
  for i0 in range(int(future)):
    _4 = (self.lstm1).forward(output4, (h_t4, c_t4), )
    h_t8, c_t8, = _4
    _5 = (self.lstm2).forward(h_t8, (h_t27, c_t27), )
    h_t29, c_t29, = _5
    output5 = (self.linear).forward(h_t29, )
    outputs1, c_t4, c_t27, h_t4, h_t27, output4 = torch.add_(outputs1, [output5]), c_t8, c_t29, h_t8, h_t29, output5
  outputs2 = torch.squeeze(torch.stack(outputs1, 1), 2)
  return outputs2

