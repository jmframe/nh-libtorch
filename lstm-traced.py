def forward(self,
    input_layer: Tensor,
    h_t: Tensor,
    c_t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
  h_t0 = torch.to(h_t, 6, False, False, None)
  c_t0 = torch.to(c_t, 6, False, False, None)
  input_layer0 = torch.to(input_layer, 6, False, False, None)
  input_view = torch.view(input_layer0, [1, 1, -1])
  _0 = (self.lstm).forward__0(input_view, (h_t0, c_t0), )
  output, _1, = _0
  c_t1, h_t1, = _1
  prediction = (self.linear).forward(h_t1, )
  return (prediction, h_t1, c_t1)

