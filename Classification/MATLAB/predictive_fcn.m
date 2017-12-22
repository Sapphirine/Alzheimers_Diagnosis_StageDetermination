function prob = predictive_fcn(Xhat, X, a, b)
% Note: assume that the relevant X are given to us:
% Each column is a feature, each row is a new data point

sum_X = sum(X,1);
[num_sam, num_dim] = size(X);
%c1 = log10(gamma(sum_X + Xhat + a)) - log10(gamma(Xhat + 1) ) - log10( gamma( sum_X + a) );
%c1 = gamma(sum_X + Xhat + a) ./ ( gamma(Xhat + 1) .* gamma( sum_X + a));
c1 = ones(1, num_dim);
for i = 1:num_dim
	vec1 = (sum_X(i) + a):(sum_X(i) + a + Xhat(i) - 1);
	vec2 = 1:(Xhat(i));

	c1(i) = sum(log(vec1)) - sum(log(vec2)) ;
	c1(i) = exp( c1(i) );

end

c2 = ((b + num_sam) / (b + num_sam + 1)) .^ ( sum_X + a );
c3 = double(1 / (b+ num_sam + 1)) .^ Xhat;

total = log(c1) + log(c2) + log(c3);

prob = exp( sum(total) );