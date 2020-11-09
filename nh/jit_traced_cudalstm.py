{'experiment_name': 'nwmv3_test_run', 'initial_forget_bias': 3, 'hidden_size': 64, 'dynamic_inputs': ['RAINRATE', 'Q2D', 'T2D', 'LWDOWN', 'SWDOWN', 'PSFC', 'U2D', 'V2D'], 'embedding_hiddens': [30, 20, 64], 'camels_attributes': ['lat', 'lon', 'area_sqkm'], 'static_inputs': None, 'hydroatlas_attributes': None, 'number_of_basins': 8, 'use_basin_id_encoding': False, 'predict_last_n': 1, 'head': 'regression', 'target_variables': 'obs', 'embedding_dropout': 0.0, 'output_dropout': 0.4, 'embedding_activation': 'tanh', 'output_activation': 'linear'}
def forward(self,
    data: Dict[str, Tensor]) -> Dict[str, Tensor]:
  x_d = torch.transpose(data["x_d"], 0, 1)
  if torch.__contains__(data, "x_s"):
    _1 = torch.__contains__(data, "x_one_hot")
    _0 = _1
  else:
    _0 = False
  if _0:
    x_s = torch.repeat(torch.unsqueeze(data["x_s"], 0), [(torch.size(x_d))[0], 1, 1])
    _2 = torch.unsqueeze(data["x_one_hot"], 0)
    x_one_hot = torch.repeat(_2, [(torch.size(x_d))[0], 1, 1])
    x_d1 = torch.cat([x_d, x_s, x_one_hot], -1)
    x_d0 = x_d1
  else:
    if torch.__contains__(data, "x_s"):
      x_s0 = torch.repeat(torch.unsqueeze(data["x_s"], 0), [(torch.size(x_d))[0], 1, 1])
      x_d2 = torch.cat([x_d, x_s0], -1)
    else:
      _3 = torch.__contains__(data, "x_one_hot")
      if _3:
        _4 = torch.unsqueeze(data["x_one_hot"], 0)
        x_one_hot0 = torch.repeat(_4, [(torch.size(x_d))[0], 1, 1])
        x_d4 = torch.cat([x_d, x_one_hot0], -1)
        x_d3 = x_d4
      else:
        x_d3 = x_d
      x_d2 = x_d3
    x_d0 = x_d2
  lstm_output, _5, = (self.lstm).forward__0(x_d0, None, )
  h_n, c_n, = _5
  h_n0 = torch.transpose(h_n, 0, 1)
  c_n0 = torch.transpose(c_n, 0, 1)
  pred = {"h_n": h_n0, "c_n": c_n0}
  _6 = self.head
  _7 = (self.dropout).forward(torch.transpose(lstm_output, 0, 1), )
  torch.update(pred, (_6).forward(_7, ))
  return pred

