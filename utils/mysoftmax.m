function y = mysoftmax(m_x_s)
% x has to have the structure of Ndims x Nsamples

[M,~] = size(m_x_s);

m_expx_s = exp(m_x_s);
y = m_expx_s ./ repmat( sum(m_expx_s,1), [M 1] );   

