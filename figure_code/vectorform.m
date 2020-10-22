function Z = vectorform(Y)

[~, n] = size(Y);

Z = Y(tril(true(n),-1));
Z = Z(:)';                 % force to a row vector, even if empty

end

