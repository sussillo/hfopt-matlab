function ho = hotone_encoding(string, code)

ho = zeros(length(code), length(string));
for i = 1:length(code)
    c = code(i);   
    cidxs = find(string == c);   
    ho(i, cidxs) = ones(1, length(cidxs));
end



    