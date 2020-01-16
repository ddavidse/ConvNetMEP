function [pointsOut] = returnFromNormal(Npoints,scaler)
% function [pointsOut] = returnFromNormal(Npoints);
% for odd number of points includes peak of the distribution
% for even number of point does not
% sepstep = 0.01; by default

% Dmytro Kolenov, 17th April 2019

% try different X here ?
span = (0.005:0.01:0.995);
% span = scale.*(0.005:0.01:0.995);
x = gaminv(span, length(span),10);

y_norm = normpdf(x, 1000, length(span));
% 28/10/2019
y_norm = interp(y_norm,2);

% plot(x,y_norm)
[val,ref] = max(y_norm);
weuse = (1/val).*y_norm;

% figure(), stem(x,y_norm)
% hold on, plot(x(ref),val,'r*','MarkerSize',15)

% plot(x,weuse)
% vline(x(ref),'g','The peak')

% if the amount of lines is odd
if rem(Npoints,2) 
    indices = ( (-1*floor(Npoints/2)*scaler):scaler:(floor(Npoints/2))*scaler ) + ref;
else % false is even
   % viscinity = ( (-1*floor(Npoints/2)):(floor(Npoints/2)) );
   % 7th of June this part of code is updated: 
   viscinity = ( (-1*floor(Npoints/2)*scaler):scaler:(floor(Npoints/2))*scaler );
   indices = viscinity(viscinity ~= 0) + ref;
end

% In principal better function should be written ot introduce "proper"
% scaling
% Check if there are negative values
a = 1123;
pointsOut = weuse(indices);
